# 残差投影方案

## 动机

两个模型处理同一个 prompt，KV 是**相似多于相异**的——都在编码 "1+1=?" 的语义。Projector 不需要从零生成 receiver KV，只需预测差异量。

---

## 一、C2C 已有的残差范式

```python
# projector.py line 1009
output = target_KV + gate × sigmoid(scalar) × projected_KV
#         ↑ 残差基        ↑ 残差修正
```

天然残差结构：target_KV 是残差基（receiver 真实 KV），projected_KV 是修正量。

方向 A 的问题是 **target_KV 不存在了**——receiver 没跑 prefill。核心问题变成：**用什么替代残差基？**

---

## 二、四层残差架构

### 层次 1：Sharer KV 作残差基

```python
# 当前 C2C
output = receiver_KV + gate × scalar × projected_KV

# 方向 A 残差版
output = Align(sharer_KV) + gate × scalar × Residual(sharer_KV)
#         ↑ 残差基              ↑ 只学差异
```

**直觉**：同一个 prompt 下，两个模型的 KV 相似度：

```
  余弦相似度
  1.0 ┤
      │ ██ 浅层 (0-5):   0.7-0.9   ← sharer KV 已经很接近
  0.8 ┤ ██
      │   ██
  0.6 ┤   ██ 中层 (6-15): 0.5-0.7  ← 需要中等修正
      │     ██
  0.4 ┤     ██ 深层 (16-27): 0.3-0.6 ← 需要较大修正
      │       ██
  0.2 ┤        ██
      │
    0 ┤─────────────────────────────
        0    5    10   15   20   27  Layer
```

残差基已有 40-90% 正确信息，Projector 只需补 10-60%。

**实现**：

```python
class ResidualProjector(nn.Module):
    def __init__(self, source_dim, target_dim, source_heads, target_heads, hidden_dim=256):
        super().__init__()
        
        # 维度对齐：只做线性投影，不做语义翻译
        self.dim_align = nn.Linear(
            source_dim * source_heads, 
            target_dim * target_heads
        )
        
        # 残差 MLP：容量小（只学差异）
        self.residual = nn.Sequential(
            nn.Linear(source_dim * source_heads, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim * target_heads)
        )
        
        # Gate：控制哪些层需要残差
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, sharer_kv, receiver_kv=None):
        source_key, source_value = sharer_kv
        
        # 维度对齐
        aligned_key = self.dim_align(source_key)
        aligned_value = self.dim_align(source_value)
        
        # 学习残差
        residual_key = self.residual(source_key)
        residual_value = self.residual(source_value)
        
        # Gate
        gate = torch.sigmoid(self.gate_logit / self.temperature)
        
        # 输出 = 残差基 + 门控残差
        output_key = aligned_key + gate * residual_key
        output_value = aligned_value + gate * residual_value
        
        return output_key, output_value
```

**6 个好处**：

| # | 好处 | 说明 |
|---|------|------|
| 1 | **降级问题** | 从"生成"降为"修正"——预测差异熵更低 |
| 2 | **天然下界** | residual≈0 → output≈aligned_sharer → 最差不低于 sharer 水平 |
| 3 | **训练稳定** | 残差头零初始化 → 初始行为≈aligned_sharer → loss 起点低 |
| 4 | **参数大幅减少** | 530M → 6-84M（缩小 6-80×）|
| 5 | **误差截断** | 每层地基是稳定的 sharer KV，非前层误差的废墟 |
| 6 | **天然置信度** | CosSim(aligned_sharer, real_KV) = 方向 B 的验证触发信号 |

---

### 层次 2：多层残差精炼

```
KV₀ = Align(sharer_KV)               # 初始猜测
KV₁ = KV₀ + Refiner(KV₀)             # 第一次修正
KV₂ = KV₁ + Refiner(KV₁)             # 第二次修正
...
KV_out = KVₙ                          # 精炼结果
```

**好处**：每步修正量递减 → 更容易学习；共享 Refiner 权重降低参数。

**代价**：多次 forward 增加推理开销，需要控制步数。

**适用场景**：仅在深层（16-27）对残差质量要求高时启用 2-3 步精炼。

---

### 层次 3：跨层残差修正

```
Layer i 的投影不仅看本层 sharer_KV，还看相邻层投影结果:

projected_i = Projector(sharer_KV_i)
            + λ_up   × CrossUp(projected_{i-1})     ← 上层信号
            + λ_down × CrossDown(projected_{i+1})   ← 下层信号
```

**直觉**：相邻层的 KV 强相关。Layer 2 和 Layer 4 的正确信息可以修正 Layer 3 的误差。

**实现**：

```python
class CrossLayerProjector(nn.Module):
    def __init__(self, d, num_layers):
        self.cross_up   = nn.Linear(d, d)  # 接收上层残差信号
        self.cross_down = nn.Linear(d, d)  # 接收下层残差信号
    
    def forward(self, sharer_kv, prev_output, next_output):
        base = self.dim_align(sharer_kv)
        residual = self.mlp(sharer_kv)
        # 跨层修正（权重小，防止震荡）
        cross = 0.05 * self.cross_up(prev_output) + 0.05 * self.cross_down(next_output)
        return base + residual + cross
```

**好处**：误差传播在层间被截断——单层出错不传染全栈。

**注意**：必须控制 cross 权重（λ < 0.1），否则不同层互相干扰可能震荡。

---

### 层次 4：Token/Head 粒度的残差控制

```
Per-layer:    gate ∈ R¹          → 整层全开或全关
Per-head:     gate ∈ R^H         → "第 3 头需修正，第 5 头可直接用"
Per-token:    gate ∈ R^N         → "token '=' 需修正，token '1' 可复用"
Per-channel:  gate ∈ R^(H×D)    → 最细粒度
```

方向 A 可能需要 per-token 控制，因为不同 token 的跨模型差异不同：

```
Token "1":  两个模型对数字理解几乎一致 → 残差 ≈ 0
Token "+":  算符语义类似 → 小残差
Token "=":  对"暗示结果"的理解不同 → 需大残差
Token "?":  对"问题意图"差异最大 → 需最大残差
```

**好消息**：C2C 的 `C2CProjector` 已有 per-head scalar weight（`key_scalar_head` 输出 `[B,N,H]`），可直接复用。

---

## 三、最终公式

```
output = Align(sharer_KV)                                    ← L1: 残差基
       + gate × scalar × Residual(sharer_KV)                 ← L1: 学习残差
       + 0.05 × CrossUp(prev_layer_output)                   ← L3: 跨层修正
       + 0.05 × CrossDown(next_layer_output)                 ← L3: 跨层修正
       + optional: Refine(current_output)                    ← L2: 精炼（仅深层）
```

---

## 四、集成到 C2C 代码

继承 `C2CProjector`，只改 `forward`：

```python
class ResidualC2CProjector(C2CProjector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 加维度对齐层（如果 source_dim ≠ target_dim 或 head 数不同）
        out_dim = self.target_dim * self.target_num_heads
        in_dim = self.source_dim * self.source_num_heads
        self.key_align = nn.Linear(in_dim, out_dim, dtype=kwargs.get('dtype', torch.float32))
        self.val_align = nn.Linear(in_dim, out_dim, dtype=kwargs.get('dtype', torch.float32))
        
        # 残差头零初始化
        nn.init.zeros_(self.key_proj_out.weight)
        nn.init.zeros_(self.key_proj_out.bias)
        nn.init.zeros_(self.value_proj_out.weight)
        nn.init.zeros_(self.value_proj_out.bias)
    
    def forward(self, source_kv, target_kv=None, **kwargs):
        source_key, source_value = source_kv
        
        # ====== L1: 残差基 ======
        aligned_key = self.key_align(source_key)
        aligned_value = self.val_align(source_value)
        
        # ====== L1: 学习残差（复用父类投影逻辑）=======
        projected = super().forward(source_kv, target_kv, **kwargs)
        # 但父类的 output = target + gate * proj
        # 这里改: residual = projected - target → output = aligned + gate * residual
        # 或更简单: 直接用 projected 的 delta 部分
        
        # ====== L1: 组合 ======
        output = aligned_key + gate * scalar * projected_key, ...
        
        return output
```

改动量约 100 行。

---

## 五、收益汇总

| 维度 | 纯投影 | 残差投影 | 倍率 |
|------|--------|---------|:---:|
| Projector 参数量 | 530M | 6M (共享) / 84M (独立) | **6-80×** |
| 训练显存 | ~15 GB | ~5 GB | **3×** |
| 训练稳定性 | 需精细调参 | 零初始化 → warm start | — |
| 最差性能 | 垃圾输出 | ~sharer 水平 | — |
| 误差累积 | 逐层放大 | 被残差基截断 | — |
| 方向 B 验证触发 | 需额外网络 | CosSim 即置信度 | — |
| 分析价值 | 无 | Gate 量化层间语义距离 | — |

---

## 六、信息论直觉

```
纯投影: 
  sharer_KV → [大容量 projector] → receiver_KV
  H(output) ≥ H(receiver_KV | sharer_KV) 大

残差投影:
  sharer_KV → [对齐] → aligned_KV ─┐
            → [小MLP] → residual ─┴→ output
  H(residual) = H(receiver_KV | aligned_sharer_KV) ≪ H(receiver_KV | sharer_KV)

因为 aligned 已经去掉了大部分线性可预测的差异，
residual 只需要编码非线性、不可预测的那小部分。
```

这就是为什么 256 维的隐藏层就够——残差空间比完整 KV 空间稀疏得多。
