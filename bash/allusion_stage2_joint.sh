#!/usr/bin/env bash
# Stage-2：解冻 AllusionEncoder，三损失联合微调（含 L_allusion）
set -euo pipefail
cd "$(dirname "$0")/../improved-diffusion"

IMAGE_SIZE="${IMAGE_SIZE:-12}"
DIFF_STEPS="${DIFF_STEPS:-2000}"
IN_CHANNEL="${IN_CHANNEL:-16}"
LR="${LR:-0.00005}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-50000}"
SEED="${SEED:-102}"
NOTES="${NOTES:-ci_sz12_allusion_s2}"
DATASET_DIR="${DATASET_DIR:-../datasets/ci8w}"
ALLUSION_KB="${ALLUSION_KB:-../datasets/allusion_kb/allusion_v1_candidates.json}"
VOCAB_SIZE="${VOCAB_SIZE:-5049}"
CKPT_STEP="${CKPT_STEP:-}"
STAGE1_NOTES="${STAGE1_NOTES:-ci_sz12_allusion_s1}"

CKPT_ARG=""
if [[ -n "$CKPT_STEP" ]]; then
  CKPT_ARG="--checkpoint $CKPT_STEP"
fi

OUT_DIR="diffusion_models/diff_e2e-tgt_block_${DIFF_STEPS}steps_${IN_CHANNEL}dim_${NOTES}"
mkdir -p "$OUT_DIR"

echo "Stage-2: joint train (allusion_encoder unfrozen) from ${STAGE1_NOTES}"

python scripts/run_train.py \
  --diff_steps "$DIFF_STEPS" \
  --model_arch transformer \
  --lr "$LR" \
  --lr_anneal_steps "$LR_ANNEAL_STEPS" \
  --seed "$SEED" \
  --noise_schedule sqrt \
  --image_size "$IMAGE_SIZE" \
  --in_channel "$IN_CHANNEL" \
  --modality e2e-tgt \
  --submit no \
  --padding_mode block \
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train $DATASET_DIR --notes $NOTES --experiment_mode allusion_gen --allusion_kb $ALLUSION_KB --freeze_allusion False --allusion_loss_weight 0.1 " \
  --notes "$NOTES" \
  $CKPT_ARG
