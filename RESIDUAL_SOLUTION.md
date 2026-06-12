# 残差投影方案：实验更新版

## 结论先行

原始设想是：

```text
output = Align(sharer_KV) + gate * scalar * Residual(sharer_KV)
```

这个 **source-only 残差基** 版本已经实现并跑过 0.6B + 0.5B smoke，但生成会崩：乱码、空输出、重复 token。说明 `Align(sharer_KV)` 不能直接替代 receiver 的真实 KV 基座。

当前更可行的版本是 **hybrid 残差基**：

```text
base   = alpha * receiver_KV + (1 - alpha) * Align(sharer_KV)
output = base + gate * scalar * Residual(sharer_KV)
```

当 `alpha = 0.95` 时，0.6B + 0.5B smoke 训练稳定，生成恢复正常。继续试 `alpha = 0.8` 后，生成仍正常，但 eval loss 明显变差。这说明 receiver KV 依赖可以降低，但降低过快会损伤质量。它还不能证明比原版 C2C 更强，但说明残差方案可以作为一条稳定的 curriculum 路线：先保留 receiver KV 稳住生成，再逐步降低 `alpha`，逼近 direction A。

---

## 一、为什么直接 source-only 会失败

C2C 原本的实际结构已经是残差式：

```python
output = target_KV + gate * sigmoid(scalar) * projected_KV
#        receiver 真实 KV         source 注入修正
```

它稳定的关键是 `target_KV` 来自 receiver 自己，已经处在 receiver 模型熟悉的注意力空间里。

direction A 想省掉 receiver prefill，于是自然会尝试：

```text
output = Align(sharer_KV) + residual
```

但实测说明这个替换过猛。即使两个模型处理同一个 prompt，KV 不是只差一个小残差；它们的 head 组织、层语义、数值尺度、position 行为都可能不同。`Align(sharer_KV)` 一旦直接写进 receiver cache，会破坏 receiver 后续 decode 的注意力分布。

### Smoke 结果

配置：0.6B receiver + 0.5B sharer，100 sample OpenHermes smoke。

| 方案 | Final eval loss | 生成 |
|---|---:|---|
| source-only residual | 15.6616 | 乱码、空输出、重复 token |
| hybrid residual, alpha=0.95 | 0.0291 | 正常生成 |
| hybrid residual, alpha=0.8 | 0.5050 | 正常生成，但 loss 明显变差 |

source-only 生成样例：

```text
Prompt: Say hello in one short sentence.
Output: phantanger_null_null_null_null_...

Prompt: What is 1+1? Answer briefly.
Output: ""

Prompt: Write a Python function...
Output: KAPKAPKAPAPAPAPAP...
```

hybrid 生成样例：

```text
Prompt: Say hello in one short sentence.
Output: Hello! How can I assist you today?

Prompt: What is 1+1? Answer briefly.
Output: 1+1=2

Prompt: Write a Python function that returns the square of x.
Output:
def square(x):
    return x * x
```

`alpha=0.8` 生成样例：

```text
Prompt: Say hello in one short sentence.
Output: Hello! I'm here to help you with any questions. Let me know what you need!

Prompt: What is 1+1? Answer briefly.
Output: 2

Prompt: Write a Python function that returns the square of x.
Output:
def square(x):
    return x * x
```

---

## 二、当前实现

新增 projector 类型：

```text
ResidualC2CProjector
```

位置：

```text
rosetta/model/projector.py
```

核心公式：

```text
aligned = Align(source_kv)
base    = alpha * target_kv + (1 - alpha) * aligned
output  = base + gate * scalar * Residual(source_kv)
```

其中：

| 参数 | 作用 |
|---|---|
| `target_base_init` | `alpha` 的初始值，建议先用 `0.95` 或 `0.9` |
| `trainable_target_base` | 是否让 `alpha` 可训练 |
| `zero_init_residual` | 残差分支零初始化，避免初始随机扰动 |
| `identity_init_align` | source/target KV 维度一致时，Align 用 identity 初始化 |

为了不影响原项目，原有 `C2CProjector` 和 `AllInOneProjector` 没有改行为。新逻辑只在配置里显式使用 `ResidualC2CProjector` 时生效。

### Hybrid smoke 配置

```text
recipe/train_recipe/C2C_0.6+0.5_residual_hybrid_smoke.json
```

启动脚本：

```text
bash/train/residual_hybrid_smoke_0.6_0.5.sh
```

输出：

```text
local/checkpoints/0.6+0.5B_residual_hybrid_smoke/final
```

`alpha=0.8` smoke 配置：

```text
recipe/train_recipe/C2C_0.6+0.5_residual_hybrid_a0.8_smoke.json
```

输出：

```text
local/checkpoints/0.6+0.5B_residual_hybrid_a0.8_smoke/final
```

---

## 三、修正后的理解

原文里有一个乐观假设：

> residual≈0 → output≈aligned_sharer → 最差不低于 sharer 水平

实测后这个说法需要修正。对 receiver 来说，`aligned_sharer` 不是“sharer 水平”，而是一个被写进 receiver 内部状态空间的外来 KV。它可能比 receiver 自己的 KV 差很多，甚至直接导致解码崩溃。

更准确的下界是：

```text
alpha≈1, residual≈0 → output≈receiver_KV
```

也就是说，hybrid 残差基才有稳定下界。source-only 版本需要额外训练阶段让 `Align(source_kv)` 先学到 receiver KV 的几何结构。

---

## 四、推荐路线

### 阶段 1：Hybrid 稳定训练

先训练：

```text
alpha = 0.95
output = 0.95 * receiver_KV + 0.05 * Align(source_KV) + residual
```

目标是保证生成不崩，并观察 residual/gate 是否学到有效修正。

### 阶段 2：Alpha curriculum

逐步降低 `alpha`：

```text
0.95 → 0.9 → 0.8 → 0.5 → 0.2 → 0
```

每个点都检查：

| 指标 | 目的 |
|---|---|
| eval loss | 是否仍能训练 |
| short generation sanity | 是否乱码/空输出/重复 |
| MMLU/GSM8K 等任务指标 | 是否超过 base 和原版 C2C |
| `target_base_weight` | 是否真的学会少依赖 receiver KV |

### 阶段 3：Align 预热

如果 alpha 降低后开始崩，应先训练 Align：

```text
loss_align = MSE(Align(source_KV), receiver_KV)
```

再接 SFT loss：

```text
loss = CE(output_logits, labels) + lambda * loss_align
```

这样 source-only 方向才有机会成立。

---

## 五、是否比原版更好

目前不能下结论。

当前结果只证明：

1. source-only residual 不稳定。
2. hybrid residual 稳定。
3. residual projector 参数量更小：0.6B + 0.5B smoke 中约 27.8M 可训练参数。
4. hybrid 生成正常，但很可能主要是在保持 base 行为。
5. `alpha=0.8` 仍可生成，说明可以减少 receiver KV 依赖；但 eval loss 从 `0.0291` 升到 `0.5050`，说明减少依赖会带来质量代价。

### 当前优势

| 优势 | 说明 |
|---|---|
| 稳定训练路径 | source-only 会崩，hybrid 不崩，可以作为 curriculum 起点 |
| 参数更少 | 当前 residual projector 约 27.8M 可训练参数，显著小于 AllInOne smoke 的约 160M |
| 可控逼近 direction A | 通过 `alpha` 控制 receiver KV 依赖，从 `0.95` 逐步降到 `0` |
| 易定位失败边界 | alpha sweep 可以明确看到从稳定到崩溃的临界点 |
| 行为确实改变 | 生成比 base 更短、更直接，说明不是完全复制 base 输出 |

### 当前限制

| 限制 | 说明 |
|---|---|
| 未证明优于原版 C2C | 还没有同训练步数、同评测集的正式对比 |
| 低 alpha 质量下降 | `alpha=0.8` 不崩，但 loss 明显差于 `alpha=0.95` |
| 仍依赖 receiver prefill | hybrid 还不是 direction A 的最终无 receiver-prefill 形态 |
| smoke 太小 | 当前只是 100 sample sanity check，不能代表任务指标 |

要证明优于原版，需要同条件比较：

| 模型 | 说明 |
|---|---|
| Base 0.6B | receiver 单模型 |
| 原版 C2CProjector | 当前 C2C baseline |
| ResidualC2CProjector alpha=0.95 | 稳定 hybrid |
| ResidualC2CProjector alpha schedule | 逐步逼近 direction A |
| ResidualC2CProjector source-only | 失败对照 |

评测必须使用同数据、同训练步数、同 eval recipe。

---

## 六、下一步实验建议

优先做小规模 ablation。根据当前结果，下一步最有信息量的是 `alpha=0.9`：它比 `0.95` 更少依赖 receiver KV，又可能比 `0.8` 稳得多。

```text
alpha ∈ {0.95, 0.9, 0.8, 0.5}
num_samples = 1k 或 5k
max_length = 512
模型 = Qwen3-0.6B + Qwen2.5-0.5B-Instruct
```

当前边界：

```text
alpha=0.95: 稳，eval loss 很低，生成正常，可能主要保持 base 行为
alpha=0.8 : 生成正常，但 eval loss 明显变差
alpha=0   : 生成崩溃
```

不要马上跑 source-only 大实验。先找“生成不崩且 loss 不明显恶化的最小 alpha”，再研究如何继续降低 alpha。

如果目标是 direction A，也就是 receiver 不跑 prefill，那么目前最靠谱的路线不是一步到位，而是：

```text
Hybrid residual → alpha curriculum → Align/MSE 预热 → source-only residual
```
