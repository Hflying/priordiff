#!/usr/bin/env bash
# E2E NLG 跨域诊断版本扩散训练：100k step。与 sonnet/ci 同 in_channel=16, predict_xstart=True。
# image_size=8 → T=64 覆盖 E2E 长度分布的 99.99%。
# padding_mode=pad（与 sonnet 同），便于 BAD/VPD 在 pad 模式下的纵向对比。

set -euo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

mkdir -p logs improved-diffusion/diffusion_models

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

: "${LR_STEPS:=100000}"
: "${IMAGE_SIZE:=8}"        # T=64
: "${IN_CHANNEL:=16}"
: "${BATCH_SIZE:=64}"
: "${SEED:=102}"
: "${VOCAB_SIZE:=11000}"    # E2E target vocab (>10) is 840; 留 padding 余量

CKPT_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_pad_2000steps_${IN_CHANNEL}dim_e2e"
mkdir -p "$CKPT_DIR"

LOG="logs/e2e_diffusion_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%F %T')] e2e diffusion 100k start  lr_steps=$LR_STEPS  image_size=$IMAGE_SIZE  vocab=$VOCAB_SIZE"
echo "  log -> $LOG  ckpt -> $CKPT_DIR"

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
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train ../datasets/e2e_data --notes e2e " \
  --notes e2e \
  2>&1 | tee "../$LOG"

echo "[$(date '+%F %T')] e2e diffusion done"
