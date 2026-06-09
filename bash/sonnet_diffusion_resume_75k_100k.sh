#!/usr/bin/env bash
# Resume sonnet diffusion from 75k EMA → train to 100k.
# 用 run_train.py 的 --checkpoint 参数（不带 step prefix），
# 它会把 model075000.pt 当成 resume_checkpoint 传给 train.py.
# Optimizer state opt075000.pt 不存在，train_util.py 会安全 fallback。

set -uo pipefail
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
: "${IMAGE_SIZE:=14}"
: "${IN_CHANNEL:=16}"
: "${BATCH_SIZE:=64}"
: "${SEED:=102}"
: "${VOCAB_SIZE:=22970}"
: "${RESUME_STEP:=075000}"

CKPT_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_pad_2000steps_${IN_CHANNEL}dim_sonnet3355"
[ -f "${CKPT_DIR}/model${RESUME_STEP}.pt" ] || { echo "[err] missing ${CKPT_DIR}/model${RESUME_STEP}.pt"; exit 1; }
mkdir -p "$CKPT_DIR"

LOG="logs/sonnet_diffusion_resume_${RESUME_STEP}_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%F %T')] sonnet resume from step=${RESUME_STEP}, target lr_anneal=${LR_STEPS}"
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
  --checkpoint "$RESUME_STEP" \
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train ../datasets/sonnet3355 --notes sonnet3355 " \
  --notes sonnet3355 \
  2>&1 | tee "../$LOG"

echo "[$(date '+%F %T')] sonnet resume done"
