#!/usr/bin/env bash
# M3-v1：在 ci8w 上训练 BERT MLM（与 sz12 词表对齐），供 decode_pkl_m3_v1.py 使用。
set -euo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}/improved-diffusion
mkdir -p logs diffusion_models/m3v1_mlm_ci8w

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

: "${MAX_STEPS:=50000}"
: "${BATCH_SIZE:=32}"
: "${SAVE_EVERY:=5000}"
: "${LOG_EVERY:=100}"

echo "[$(date '+%F %T')] M3-v1 MLM train  max_steps=$MAX_STEPS  batch=$BATCH_SIZE"

python -u scripts/train_m3v1_mlm.py \
  --vocab_dir diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu \
  --corpus ../datasets/ci8w/ci_train.txt \
  --out_dir diffusion_models/m3v1_mlm_ci8w \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --save_every "$SAVE_EVERY" \
  --log_every "$LOG_EVERY" \
  2>&1 | tee "logs/m3v1_train_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] done"
