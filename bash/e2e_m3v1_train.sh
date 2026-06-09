#!/usr/bin/env bash
# M3-v1 MLM training on E2E NLG (word-level, spaCy tokenizer).
# Trains a small BERT MLM on the utterance portion of src1_train.txt
# (the part after the `||` MR/utterance separator), aligned with the
# E2E diffusion model's 821-token vocab.
set -euo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}/improved-diffusion

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

mkdir -p logs diffusion_models/m3v1_mlm_e2e
LOG="logs/e2e_m3v1_train_$(date +%Y%m%d_%H%M%S).log"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
: "${MAX_STEPS:=10000}"
: "${BATCH_SIZE:=64}"
: "${SAVE_EVERY:=2000}"
: "${LOG_EVERY:=100}"
: "${DEVICE:=auto}"

echo "[$(date '+%F %T')] E2E M3-v1 MLM train  steps=$MAX_STEPS  bsz=$BATCH_SIZE  device=$DEVICE" | tee "$LOG"

python -u scripts/train_m3v1_mlm_e2e.py \
  --vocab_dir diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_e2e \
  --corpus ../datasets/e2e_data/src1_train.txt \
  --out_dir diffusion_models/m3v1_mlm_e2e \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --save_every "$SAVE_EVERY" \
  --log_every "$LOG_EVERY" \
  --device "$DEVICE" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done" | tee -a "$LOG"
