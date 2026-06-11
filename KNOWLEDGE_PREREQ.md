# LLM 推理前置知识

## 一、Transformer 注意力机制

### QKV 角色

```
Attention(Q, K, V) = softmax(Q × K^T) × V

Q (Query):  "我想查什么"  — 当前 token 发出的查询
K (Key):    "我是什么"    — 每个 token 的索引标签  
V (Value):  "我知道什么"  — 每个 token 的实际内容
```

类比数据库检索：Q 是查询、K 是索引、V 是要读取的数据。

### 为什么只缓存 KV，不缓存 Q

| | 能否缓存 | 原因 |
|--|:---:|------|
| **K** | ✅ | token 生成后，"被查者"的索引固定 |
| **V** | ✅ | token 生成后，"被查者"的内容固定 |
| **Q** | ❌ | 每一步要查的事情不同，且 Q 来自当前 token 的 embedding，而当前 token 由上一步输出决定 |

后面的 token 需要用前面的 K 和 V 做检索，但永远不需要前面的 Q——前面的 Q 是给前面的 token 查别人用的，使命完成就扔了。

### 因果注意力 Mask

```
         K₀  K₁  K₂  K₃  K₄
    Q₀ [ ✓   ✗   ✗   ✗   ✗ ]
    Q₁ [ ✓   ✓   ✗   ✗   ✗ ]
    Q₂ [ ✓   ✓   ✓   ✗   ✗ ]
    Q₃ [ ✓   ✓   ✓   ✓   ✗ ]
    Q₄ [ ✓   ✓   ✓   ✓   ✓ ]

每个 token 只能看到自己及前面的 token，看不到未来的。
```

### 多头注意力

把一个大注意力拆成 8 个小的并行做，各头自然分化：

```
输入 h [1, 5, 1024] → 切成 8 片 [1, 5, 128]
每片独立做 QKV 投影 + 注意力 → concat → Wo 线性变换
```

头间分化来自随机初始化 + 梯度竞争，无需人工指定。

### GQA（分组查询注意力）

```
Qwen3-0.6B: 8 Q heads, 8 KV heads  → 标准 MHA (1:1)
Qwen3-8B:  32 Q heads, 8 KV heads  → 4 个 Q 头共享 1 组 KV

KV 头数更少 → KV Cache 更小 → 显存省 4×
```

---

## 二、Prefill vs Decode

### Prefill

```
输入: 整个 prompt 的所有 token 一次进
计算: 所有 token 的 QKV 投影 + 注意力 + FFN（compute-bound）
产出: 28 层 × N 个 token 的 KV Cache + 第一个输出 token
扔掉: 所有 Q
```

### Decode

```
输入: 每次 1 个新 token（上一步的输出）
计算: 只算当前 token 的 QKV 投影 + 与历史 KV 的注意力（memory-bound）
产出: 1 组 K,V 追加到缓存 + 下一个 token
关键: KV Cache 省的是重复投影 K,V，不省 Q×K 注意力矩阵
```

### 为什么 Decode 必须一步步来

每一步的输入 token 由上一步的输出决定，无法提前知道。这个因果链打不破。

```
Step 1: 输入 "2"   → Q₅K₅V₅ → 输出 ","      ← Step 2 的输入
Step 2: 输入 ","   → Q₆K₆V₆ → 输出 "答"      ← Step 3 的输入
...
```

位置编号从 prefill 结束位置接着排，是连续的绝对位置。

---

## 三、KV Cache 省了什么

```
无缓存: 每步重新计算所有历史 token 的 K,V → 总计算量 O(n²)
有缓存: 每步只算 1 个新 token 的 Q,K,V → 读历史 KV → 总计算量 O(n)

KV Cache 省了: 历史 K,V 投影 (大矩阵乘法)
不省: Q_new × 所有历史 K (注意力点积，本身开销小)
```

瓶颈在 memory-bound——大部分时间花在从显存搬运 KV Cache。

---

## 四、投机解码

### 核心思想

不改变"每步依赖上一步"的约束，而是用**小模型提前猜多个 token**，大模型一次验证。

```
不用投机: 100 token = 100 次大模型 forward
用投机:   100 token ≈ 35 次大模型 forward (接受率 ~3/step)

加速 2-3×
```

### 对 KV Cache 的影响

验证阶段大模型一次处理多个候选 token → KV Cache 需增删验证失败的分支 → KV Cache 管理变复杂（SwiftSpec、EXSpec 都在解决这个）。

### 对 Prefill 的影响

投机解码不碰 Prefill，只加速 Decode。

---

## 五、EAGLE：预测隐状态 = 间接预测 Q

### 为什么直接预测 Q 不行

Q 来自当前 token 的 embedding，当前 token 由上一步决定 → 预测 Q ≈ 预测下一个 token ≈ 需要跑整个 LLM。

### EAGLE 的绕路

```
h₄ (当前隐状态) → [Draft Model] → ĥ₅ (预测的未来隐状态)
                                      ↓
                                Wq × ĥ₅ = Q̂₅  ← 免费得到
                                Wk × ĥ₅ = K̂₅
                                Wv × ĥ₅ = V̂₅
```

预测隐状态而非 Q。**EAGLE-3 (2025)** 融合浅、中、深三层特征 + 训练时模拟推理分布。

---

## 六、C2C 核心概念

### 做法

两个模型同时跑 prefill → Projector 将 teacher KV 投影融合到 receiver KV → receiver 用融合 KV decode。

### kv_cache_index

逐 token 控制用谁的信息：
- prompt 阶段：用 teacher 投影融合（`[1, 0]`）
- decode 阶段：receiver 自己生成（`[-1, 0]`）

### Projector 公式

```
output = target_KV + gate × sigmoid(scalar) × projected_KV
```

可学习组件：gate（层开关）+ scalar（token/head 级权重）+ projection MLP。

### 训练

两个 LLM 冻结，只训练 Projector (~530M)。温度退火 1.0→0.001 确定哪些层需要跨模型融合。

---

## 七、关键论文速查

| 论文 | 核心贡献 |
|------|---------|
| **C2C** (ICLR 2026) | 跨模型 KV Cache 投影融合，decode 阶段 |
| **EAGLE-3** (NeurIPS 2025) | 预测未来隐状态，间接得到 QKV |
| **P-EAGLE** (2025) | 并行预测多个未来隐状态 |
| **DroidSpeak** (2025) | 跨模型 KV 复用，逐层敏感度分析 |
| **SwiftSpec** (2025) | 异步投机解码 + KV Cache 一致性 |
| **FastKV** (2026) | Prefill-Decode 解耦 KV 压缩 |
| **SpeCache** (ICML 2025) | CPU 卸载 + 投机 KV 预取 |
| **QuantSpec** (ICML 2025) | 量化 KV 自投机解码 |

---

## 八、术语速查

| 术语 | 一句话 |
|------|--------|
| Prefill | 一次性处理 prompt，算完所有 token 的 KV 缓存 |
| Decode | 逐 token 生成，每次只算 1 个新 token |
| KV Cache | 历史 token 的 K,V 矩阵，省掉重复投影 |
| O(n²) | 注意力矩阵的计算复杂度 |
| Memory-bound | Decode 瓶颈是等显存搬运数据 |
| Compute-bound | Prefill 瓶颈是算力不够 |
| 投机解码 | 小模型猜多 token，大模型一次验证，加速 Decode |
| Gate | 学习到的层开关，控制是否启用投影 |
| RoPE | 旋转位置编码，每个位置的 token 有不同角度 |
| GQA | 分组查询注意力，Q 头多于 KV 头 |
