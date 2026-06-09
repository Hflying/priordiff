#!/usr/bin/env bash
# Stage-1：冻结 AllusionEncoder，从已有 ci_sz12 权重续训主干（experiment_mode=allusion_gen）
set -euo pipefail
cd "$(dirname "$0")/../improved-diffusion"

IMAGE_SIZE="${IMAGE_SIZE:-12}"
DIFF_STEPS="${DIFF_STEPS:-2000}"
IN_CHANNEL="${IN_CHANNEL:-16}"
LR="${LR:-0.0001}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-200000}"
SEED="${SEED:-102}"
NOTES="${NOTES:-ci_sz12_allusion_s1}"
DATASET_DIR="${DATASET_DIR:-../datasets/ci8w}"
ALLUSION_KB="${ALLUSION_KB:-../datasets/allusion_kb/allusion_v0.json}"
VOCAB_SIZE="${VOCAB_SIZE:-5049}"
CKPT_STEP="${CKPT_STEP:-150000}"
PRETRAIN_NOTES="${PRETRAIN_NOTES:-ci_sz12_gpu}"

CKPT_ARG=""
if [[ -n "$CKPT_STEP" ]]; then
  CKPT_ARG="--checkpoint $CKPT_STEP"
fi

OUT_DIR="diffusion_models/diff_e2e-tgt_block_${DIFF_STEPS}steps_${IN_CHANNEL}dim_${NOTES}"
mkdir -p "$OUT_DIR"

echo "Stage-1: freeze allusion_encoder, init from ${PRETRAIN_NOTES} @ ${CKPT_STEP}"

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
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train $DATASET_DIR --notes $NOTES --experiment_mode allusion_gen --allusion_kb $ALLUSION_KB --freeze_allusion True --allusion_loss_weight 0.0 " \
  --notes "$NOTES" \
  $CKPT_ARG
