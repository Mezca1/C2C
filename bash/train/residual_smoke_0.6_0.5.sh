#!/usr/bin/env bash
set -euo pipefail

cd /home/lishaowei/cache2cache/C2C

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p local/logs

/home/lishaowei/anaconda3/envs/c2c/bin/torchrun \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29511}" \
  script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_0.6+0.5_residual_smoke.json
