#!/usr/bin/env bash
# Sonnet 跨域诊断版本扩散训练：100k step（不需要 SOTA quality，只演示失败模式 + 修复）。
# 与现有 bash/sonnet_diffusion.sh 的差别：
#   - lr_anneal_steps 200k → 100k
#   - 增加日志重定向（logs/sonnet_*.log）
#   - 与 ci 保持同 in_channel=16，便于对比

set -euo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

# 激活 conda 环境，确保 improved_diffusion 等本地包可见
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

mkdir -p logs improved-diffusion/diffusion_models

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

: "${LR_STEPS:=100000}"
: "${IMAGE_SIZE:=14}"      # 14*14=196 covers max sonnet length
: "${IN_CHANNEL:=16}"
: "${BATCH_SIZE:=64}"
: "${SEED:=102}"
: "${VOCAB_SIZE:=22970}"

# 预先创建 checkpoint 目录，避免 train.py 直写文件失败
CKPT_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_pad_2000steps_${IN_CHANNEL}dim_sonnet3355"
mkdir -p "$CKPT_DIR"

LOG="logs/sonnet_diffusion_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%F %T')] sonnet diffusion 100k start  lr_steps=$LR_STEPS  image_size=$IMAGE_SIZE"
echo "  log -> $LOG"

cd improved-diffusion
python -u scripts/run_train.py \
  --diff_steps 2000 \
  --model_arch transformer \
  --lr 0.0001 \
  --lr_anneal_steps "$LR_STEPS" \
  --seed "$SEED" \
  --noise_schedule sqrt \
  --image_size "$IMAGE_SIZE" \
  --in_channel "$IN_CHANNEL" \
  --modality e2e-tgt \
  --submit no \
  --padding_mode pad \
  --bsz "$BATCH_SIZE" \
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train ../datasets/sonnet3355 --notes sonnet3355 " \
  --notes sonnet3355 \
  2>&1 | tee "../$LOG"

echo "[$(date '+%F %T')] sonnet diffusion done"
