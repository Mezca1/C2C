# C2C (Cache-to-Cache) 项目原理与复现计划

## 一、项目概述

**论文**: [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)  
**会议**: ICLR 2026  
**代码**: https://github.com/thu-nics/C2C  
**预训练权重**: https://huggingface.co/nics-efc/C2C_Fuser

### 核心思想

让两个 LLM 在 **KV-Cache 层面直接通信**，绕过文本生成，实现更高效的多模型协作。

```
传统文本通信:
  Model A → 生成文本 → 文本输入 → Model B    (慢，信息有损)

C2C 通信:
  Model A → KV-Cache → Projector → Model B   (快，信息更丰富)
```

### 性能

- 准确率：比单模型高 **8.5–10.5%**，比文本通信高 **3.0–5.0%**
- 速度：比文本通信快 **2.0×**
- 模型参数冻结，只训练 Projector

---

## 二、架构原理

### 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                        Prefill 阶段                         │
│                                                             │
│  Prompt ──→ [Base Model (Receiver)] ──→ Base KV Cache       │
│       │                                                     │
│       └─→ [Teacher Model (Sharer)] ──→ Teacher KV Cache     │
│                                          ↓                  │
│                                    [Projector ×28]          │
│                                          ↓                  │
│                                    Fused KV Cache           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                        Decode 阶段                          │
│                                                             │
│  Fused KV Cache → Base Model → token_1 → token_2 → ...     │
│                        (用融合后的 KV 逐 token 生成)        │
└─────────────────────────────────────────────────────────────┘
```

### Projector 结构（`C2CProjector`）

每个 projector 层处理一对 (teacher_layer, base_layer) 的 KV：

```
Source KV (teacher)  ──→ [key_in]  ──→ [MLP×N]  ──→  projected KV
Target KV (base)     ──→ [value_in] ──→ [MLP×N]  ──→  scalar weights
                                                       gate (可学习参数)
                                                              ↓
  Output = Target_KV + gate × sigmoid(scalar) × Projected_KV
```

关键机制：
- **Gate**：可学习标量参数，控制该层是否启用 C2C 投影。训练时用 Gumbel-Sigmoid + 温度退火 (1.0 → 0.001)
- **Scalar weight**：输入相关的 token 级权重，控制每个 token 的投影混入比例
- **kv_cache_index**：逐 token 控制用投影 KV 还是 base 自己的 KV；prompt 阶段融合，decode 阶段自主生成

### 层映射策略

不同模型层数不同（如 teacher 36 层 → base 28 层），用 `last_aligned` 策略：将 teacher 的后 N 层映射到 base 的全部 28 层，按归一化位置最近邻匹配。

---

## 三、代码结构

```
C2C/
├── rosetta/                    # 核心包
│   ├── model/
│   │   ├── wrapper.py          # RosettaModel — 多模型 KV 管理核心
│   │   ├── projector.py        # C2CProjector — KV 投影融合
│   │   ├── sampling.py         # 生成循环 + kv_cache_index 控制
│   │   └── aligner.py          # Token 对齐（跨 tokenizer）
│   ├── train/
│   │   ├── dataset_adapters.py # 数据集适配 + RosettaDataCollator
│   │   └── model_utils.py      # 层映射工具
│   └── utils/                  # 评估、注册等工具
├── script/
│   ├── train/SFT_train.py      # 训练入口
│   ├── evaluation/             # 评估脚本
│   └── playground/             # 交互 demo
└── recipe/
    ├── train_recipe/           # 训练配置 JSON
    └── eval_recipe/            # 评估配置 YAML
```

---

## 四、环境搭建

### 本地开发机

```bash
conda create -n rosetta python=3.10 -y
conda activate rosetta
cd C2C

# 基础包
pip install torch==2.6.0 transformers==4.52.4
pip install -e .

# 训练依赖
pip install -e ".[training]"

# 评估依赖
pip install -e ".[evaluation]"
```

### 服务器

```bash
# 方式一：直接 clone
git clone https://github.com/thu-nics/C2C.git
cd C2C
# 同上安装

# 方式二：本地打包上传
tar -czf C2C.tar.gz C2C/
scp C2C.tar.gz user@server:/path/
# 服务器解压后同上安装
```

### 模型下载

```bash
# 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com
```

需要的模型（训练时自动下载）：
- Qwen/Qwen3-0.6B（~1.2 GB）
- Qwen/Qwen2.5-0.5B-Instruct（~1.0 GB）
- teknium/OpenHermes-2.5（训练数据）

---

## 五、复现步骤

### Phase 1: 验证环境（1 GPU, 30 min）

```bash
conda activate rosetta

# 验证模型能加载
python -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.bfloat16)
print(f'Model OK. VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
"
```

### Phase 2: Smoke 测试（1 GPU, 20 min）

用小数据集跑通全流程，验证训练不报错。

```bash
# 用 smoke 配置：1000 条数据, seq=512, hidden_dim=256
export CUDA_VISIBLE_DEVICES=0
python script/train/SFT_train.py --config recipe/train_recipe/C2C_smoke.json
```

**通过标准**: 不 OOM，loss 下降，checkpoint 正常保存。

### Phase 3: 正式训练（2-4 GPU, 1-2 天）

按可用 GPU 数量调整配置。

| 可用 GPU | per_device_batch | grad_accum | 有效 batch | 命令 |
|----------|:---:|:---:|:---:|------|
| 2 | 2 | 32 | 128 | `torchrun --nproc_per_node=2 ...` |
| 3 | 2 | 21 | 126 | `torchrun --nproc_per_node=3 ...` |
| 4 | 2 | 16 | 128 | `torchrun --nproc_per_node=4 ...` |

```bash
# 示例：4 GPU 训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_4090.json
```

关键配置参数：
```json
{
    "model": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "projector": {
            "type": "C2CProjector",
            "params": { "hidden_dim": 1024, "num_layers": 3 }
        },
        "mapping": "last_aligned"
    },
    "training": {
        "learning_rate": 1e-4,
        "max_length": 2048,
        "freeze": ["teacher", "base"],   // 两个 LLM 全部冻结，只训练 Projector
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "num_epochs": 1
    },
    "data": {
        "type": "OpenHermesChatDataset",
        "kwargs": { "num_samples": 500000 }
    }
}
```

### Phase 4: 评估（1 GPU, 1-2 h）

```bash
python script/evaluation/unified_evaluator.py \
    --config recipe/eval_recipe/unified_eval.yaml
```

评估用 MMLU-redux 等 benchmark，比较 C2C vs 单独模型 vs 文本通信的准确率。

### 捷径：直接用预训练 Projector

```python
from huggingface_hub import snapshot_download
ckpt = snapshot_download("nics-efc/C2C_Fuser",
    allow_patterns=["qwen3_0.6b+qwen2.5_0.5b_Fuser/*"])
# 然后参考 README 的推理代码
```

已发布的支持模型对：

| Receiver (Base) | Sharer (Teacher) |
|-----------------|------------------|
| Qwen3-0.6B | Qwen2.5-0.5B-Instruct |
| Qwen3-0.6B | Llama-3.2-1B-Instruct |
| Qwen3-0.6B | Qwen2.5-Math-1.5B |
| Qwen3-1.7B | Qwen2.5-1.5B-Instruct |
| Qwen3-8B | Qwen2.5-7B-Instruct |

---

## 六、显存参考

### 训练时单卡显存占用估算

| 配置 | 单卡 4090 (24GB) | 备注 |
|------|:---:|------|
| 全量 Projector (530M), b=2, seq=2048 | ~15 GB | 4 卡可跑 |
| 共享 Projector (74M) | ~8 GB | 1 卡可跑 |
| Smoke 配置 (hidden=256, seq=512) | ~4 GB | 1 卡轻松 |

主要显存消耗：
1. 两个 LLM 权重（bf16）：~2.2 GB
2. Projector 优化器状态：随参数量线性增长（最大的可变开销）
3. KV Cache：随 seq_len × batch 增长
4. 中间激活：随 seq_len × hidden_dim 增长

---

## 七、后续研究方向

基于 C2C 的 KV-Cache 投影能力，两个可行的扩展方向：

### 方向 A：KV 投机生成

```
只用 Sharer 跑 Prefill → Projector 直接猜 Receiver 的 KV
→ Receiver 不跑 Prefill，直接用猜测 KV 开始 Decode
```

省掉 Receiver 整个 Prefill 阶段（占总推理时间 30-70%）。

### 方向 B：分块投机 Prefill

```
Prompt 分块 → 用前一块的投影 KV 推测下一块
→ Pipeline 化 Prefill → 定期验证-修正
```

大幅降低 Time-To-First-Token。

---

## 八、常见问题

**Q: HuggingFace 下载太慢？**  
`export HF_ENDPOINT=https://hf-mirror.com`

**Q: 显存不够？**  
降低 `max_length`（2048→512），降低 `hidden_dim`（1024→256），或减少 GPU 数提高 `grad_accum`

**Q: 不想训练只想跑推理？**  
直接用预训练 Projector（Phase 捷径），下载后跑 README 的推理代码

**Q: 没装 wandb？**  
配置里 `wandb_config.mode` 设为 `"offline"`，不联网也能跑
