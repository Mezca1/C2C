# 方向 A/B 挑战分析

---

## 方向 A：KV 投机生成

**做法**: Sharer 跑 prefill → Projector 猜 Receiver 全量 KV → Receiver 不跑 prefill 直接 decode

**省什么**: Receiver 整个 prefill（大模型 + 长序列占总推理 30-70%）

---

### 挑战 1（🔴 致命）：Receiver 从没"见过" prompt

```
C2C:
  Receiver 自己跑 prefill → 有真实 KV（prompt 信息完整）
  Projector 只补充 teacher 视角
  Gate=0 → 退化为纯 receiver → 安全兜底

方向 A:
  Receiver 完全没见过 prompt 原文
  Tokenizer、Embedding、所有 28 层 QKV 投影 → 全没参与
  Projector 必须把整个 prompt 语义"翻译"进 KV 空间
  没有兜底：投影错了就全错了
```

Projector 的输出是 `[B, N, H*D]`，需要从 teacher 的 KV 编码完整重建 receiver 对 prompt 的理解。信息论难度比 C2C 高一量级——C2C 只需编码"差异"，方向 A 需要编码"全部"。

---

### 挑战 2（🔴 致命）：逐层误差累积

```
真实 prefill:
  Layer 0: h₀ → Q₀K₀V₀ → h'₀ → 缓存 K₀V₀
  Layer 1: h'₁ → Q₁K₁V₁ → h'₁ → 缓存 K₁V₁   ← h'₁ 是真实计算的
  ...

方向 A:
  Layer 0: 没有 h₀ → 直接用 projected KV₀（误差 ε₀）
  Layer 1: projected KV₁ + 前面 projected KV₀ 做注意力 → ε₁ > ε₀
  Layer 2: ε₂ > ε₁ ...
  Layer 27: ε₂₇ 可能已经发散
```

C2C 为什么没这问题：receiver 自己的 KV 是真实的，projector 只往上加东西。方向 A 是从零造 KV，每层误差污染后续层注意力。

**DroidSpeak (2025) 佐证**: 跨模型复用超过 10% 层就质量崩溃。必须精心选择哪些层重新计算。

---

### 挑战 3（🟡 严重）：训练信号设计

```
方案 1: MSE(projected_KV, real_receiver_KV)
  问题: KV 接近 ≠ decode 质量好
  隐含假设 "KV 空间线性" → 不成立

方案 2: CE(receiver_decode_with_projected_KV)
  问题: 梯度穿 28 层 receiver → 极难优化
  receiver 权重冻结 → projector 学习 "如何骗过 receiver"

方案 3: 联合 (MSE + CE)
  问题: 两个 loss 尺度和收敛速度难平衡
```

---

### 挑战 4（🟡 严重）：Distribution Shift

```
训练: Projector 学 teacher_KV → receiver_KV
推理: Receiver 收到 projected_KV（有训练误差 + 泛化误差）

Receiver 训练时从未见过 "假的 KV"
→ 后续层对 KV 微小偏差可能敏感
→ 小误差在 FFN 非线性变换后被放大
```

---

### 挑战 5（🟡 严重）：Gate 失去兜底

```
C2C 的 gate:
  gate → 1: 用 teacher 投影（激进）
  gate → 0: 用 receiver 自己（安全，有真 KV）

方向 A 的 gate:
  gate → 1: 用投影（唯一选择）
  gate → 0: 没东西可 fallback → 层输出全零 → 崩溃
```

---

### 挑战 6（🟡 严重）：最优分界点未知

```
全投影 (方向A):  28 层全投影 → 最快，最不准
C2C (baseline):  28 层全融合 → 中等
全真实 (oracle):  receiver 自己跑 → 最准，最慢

最优解可能在中间:
  浅层 (0-13):  投影（关注局部语法，跨模型差异小）
  深层 (14-27): 真实计算（关注语义，跨模型差异大）
```

但分界点怎么定？DroidSpeak 用 O(L²) 逐层敏感度分析，3 小时跑一个 32 层模型。

---

### 挑战 7（🟡 严重）：Position/RoPE 不匹配

```
Teacher 和 Receiver 的 RoPE base frequency 可能不同:
  Qwen3:      θ = 1000000
  Llama-3.2:  θ = 500000   ← 跨系列不一致

Projector 投影的是已加位置编码的 KV
→ 两个模型 RoPE 不同 → 投影会破坏位置信息
→ 需在投影前去 RoPE、投影后重新加 RoPE（增加复杂度）
```

---

## 方向 B：分块投机 Prefill

**做法**: Prompt 分块 → 前一块投影 KV 推测下一块 → Pipeline 化 → 定期验证-修正

**省什么**: Prefill 串行变流水线 → TTFT 大幅降低

---

### 挑战 8（🔴 致命）：误差链式放大

```
真实 prefill (串行):
  Chunk1 → 真实 KV₁ → Chunk2 注意力基于真实 KV₁
  Chunk2 → 真实 KV₂ → Chunk3 注意力基于真实 KV₁+KV₂  ← 完美

方向 B (投机):
  Chunk1 → 真实 KV₁
  Chunk2 → 投影 KV̂₂ → 注意力: 真实KV₁ + 投影KV̂₂        ← 一级误差
  Chunk3 → 投影 KV̂₃ → 注意力: 真实KV₁ + 投影KV̂₂ + 投影KV̂₃  ← 二级!
  Chunk4 → 三级误差...
```

Chunk n 的注意力不仅承受自己的投影误差，还承受前 n-1 个 chunk 的累积误差。到第 k 个 chunk，是 k-1 级误差在注意力矩阵中的非线性混合。

---

### 挑战 9（🟡 严重）：验证-修正策略

```
方案 A「定期全量修正」:
  每 K 个 chunk，receiver 真实 prefill 修正所有 KV
  K=1 → 真实 prefill，零加速
  K=∞ → 方向 A（无修正），无限累积
  最优 K 依赖投影精度，且可能随 prompt 变化

方案 B「增量验证」:
  每个 chunk，receiver 只验证当前 chunk 的注意力输出
  修正当前 KV，但历史 chunk 误差仍在
  问题: 历史误差通过注意力持续污染当前 chunk

方案 C「置信度触发」:
  Projector 输出置信度 → 低于阈值自动修正
  问题: 置信度校准难 → projector 可能 "自信地犯错"
```

---

### 挑战 10（🟢 可解决）：Chunk 边界效应

```
切在句号后:
  "1+1=2。3+3=" ← 语义完整 → 投影容易
  "6。5+5="      ← 语义完整

切在中间:
  "1+1=2。3+"   ← "3+" 语义不完整 → 投影质量差
  "3=6。5+5="   ← projector 收到断裂语义
```

均匀切简单但忽略语义边界。语义切需额外开销。

---

### 挑战 11（🟢 可解决）：Pipeline 调度复杂度

```
理想 pipeline:
  t₀: Sharer Chunk1
  t₁: Sharer Chunk2 | Projector Chunk1→2
  t₂: Sharer Chunk3 | Projector Chunk2→3 | Receiver 验证 Chunk1

但验证失败 → 回退 → 打断 pipeline → 重新调度
最坏: 全部投影被拒 → 退化成串行 → 比不做还慢（overhead）
```

---

### 挑战 12（🟡 严重）：小模型上收益有限

```
Prefill 耗时占比:
  Qwen3-0.6B, seq=2048: Prefill ~50ms, Decode ~500ms → prefill 占 ~10%
  Qwen3-8B,   seq=8192: Prefill ~800ms, Decode ~800ms → prefill 占 ~50%

省 100% prefill → 小模型总加速 ~10%，大模型 ~50%
省 50% prefill  → 小模型总加速 ~5%，大模型 ~25%
```

方向 A/B 需要**大模型 (8B+) 且长 prompt (8K+)** 才能体现价值。

---

## 两个方向共有

| 挑战 | 说明 |
|------|------|
| 训练数据构造 | 每个样本需跑两份模型的全量 KV → 50万样本 × 2 × 2048token → 巨量存储 |
| Baseline 对比 | vs 纯 receiver（精度上界）、vs C2C（速度+精度 baseline）、vs DroidSpeak（KV 复用 SOTA）、vs chunked prefill（工程 baseline） |
| 论文叙事 | 需与 C2C 区分：操作阶段 (decode→prefill) + 操作方式 (融合→投机/流水线) |

---

## 致命度排序

```
🔴 致命 (必须先验证):
  挑战 1:  Receiver 未见过 prompt   → 信息论上限在哪
  挑战 2:  逐层误差累积            → DroidSpeak 说 >10% 层就崩
  挑战 8:  误差链式放大            → 注意力机制非线性放大

🟡 严重 (致命挑战通过后才能处理):
  挑战 4:  Distribution Shift     → 解决不了方向不可行
  挑战 7:  最优分界点              → 需要逐层 profiling
  挑战 12: 小模型收益有限           → 需要 8B+ 验证

🟢 可解决 (工程问题):
  挑战 3/5/9/10/11                → 通过实验迭代
```

---

## 第一步验证（1 天，不改训练）

```python
# 直接用 teacher KV 替换 receiver KV 的某些层，不经过 projector
# 测信息论上限
for num_layers in range(1, 29):
    # 替换前 num_layers 层，其余真实计算
    quality = test_replacement(num_layers)
    
# 结果:
#   <5 层可替换   → 方向 A 基本不可行，需要更强的残差架构
#   5-15 层可替换 → 走 Partial Projection 路线
#   >15 层可替换  → 方向 A 可行
```
