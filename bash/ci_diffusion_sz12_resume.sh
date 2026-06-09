#!/usr/bin/env bash
# 从 ema/model_200000.pt 续训扩散模型，再训 200k 步（共 400k）。
#
# 与 ci_diffusion_sz12.sh 区别：
#   - LR_ANNEAL_STEPS: 200000 -> 400000 （线性退火总步数）
#   - CKPT_STEP:       (空)   -> 200000  （从 model200000.pt resume）
#
# 续训起始 LR：1e-4 * (1 - 200000/400000) = 5e-5
# 续训结束 LR：0（在 step=400000 时）
#
# 用法（在仓库根目录执行）：
#   mkdir -p logs
#   nohup bash bash/ci_diffusion_sz12_resume.sh \
#     > logs/diffusion_sz12_resume_$(date +%Y%m%d_%H%M).log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../improved-diffusion"

CONDA_ENV="${CONDA_ENV:-priordiff}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
echo "[env] python = $(which python)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
mkdir -p "$WANDB_DIR"

IMAGE_SIZE="${IMAGE_SIZE:-12}"
DIFF_STEPS="${DIFF_STEPS:-2000}"
IN_CHANNEL="${IN_CHANNEL:-16}"
LR="${LR:-0.0001}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-400000}"
SEED="${SEED:-102}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-sqrt}"
NOTES="${NOTES:-ci_sz12_gpu}"
DATASET_DIR="${DATASET_DIR:-../datasets/ci8w}"
VOCAB_SIZE="${VOCAB_SIZE:-5049}"
CKPT_STEP="${CKPT_STEP:-200000}"

OUT_DIR="diffusion_models/diff_e2e-tgt_block_${DIFF_STEPS}steps_${IN_CHANNEL}dim_${NOTES}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$OUT_DIR/model${CKPT_STEP}.pt" ]]; then
  echo "[error] resume checkpoint not found: $OUT_DIR/model${CKPT_STEP}.pt" >&2
  exit 1
fi
echo "[ok] resume checkpoint: $OUT_DIR/model${CKPT_STEP}.pt ($(du -h "$OUT_DIR/model${CKPT_STEP}.pt" | cut -f1))"

CKPT_ARG="--checkpoint $CKPT_STEP"

START_LR=$(python -c "print(${LR}*(1-${CKPT_STEP}/${LR_ANNEAL_STEPS}))")
echo "============================================================"
echo " CI Diffusion training (RESUME from step $CKPT_STEP)"
echo "   image_size      : $IMAGE_SIZE  (seqlen = $((IMAGE_SIZE * IMAGE_SIZE)))"
echo "   diffusion_steps : $DIFF_STEPS"
echo "   in_channel      : $IN_CHANNEL"
echo "   lr (peak)       : $LR"
echo "   lr_anneal_steps : $LR_ANNEAL_STEPS"
echo "   resume_step     : $CKPT_STEP"
echo "   start LR        : $START_LR  (will linearly anneal to 0 at step=$LR_ANNEAL_STEPS)"
echo "   notes           : $NOTES"
echo "   output dir      : $OUT_DIR"
echo "============================================================"

python scripts/run_train.py \
  --diff_steps "$DIFF_STEPS" \
  --model_arch transformer \
  --lr "$LR" \
  --lr_anneal_steps "$LR_ANNEAL_STEPS" \
  --seed "$SEED" \
  --noise_schedule "$NOISE_SCHEDULE" \
  --image_size "$IMAGE_SIZE" \
  --in_channel "$IN_CHANNEL" \
  --modality e2e-tgt \
  --submit no \
  --padding_mode block \
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train $DATASET_DIR --notes $NOTES " \
  --notes "$NOTES" \
  $CKPT_ARG
