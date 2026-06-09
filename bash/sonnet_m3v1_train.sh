#!/usr/bin/env bash
# M3-v1 MLM training on sonnet3355 (word-level, spaCy tokenizer).
# Vocab is aligned with the sonnet diffusion model so that LCVR-decoded
# logits and the MLM share the same id space (incl. <,eos,> triplet).
set -euo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}/improved-diffusion

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

mkdir -p logs diffusion_models/m3v1_mlm_sonnet3355
LOG="logs/sonnet_m3v1_train_$(date +%Y%m%d_%H%M%S).log"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
: "${MAX_STEPS:=10000}"
: "${BATCH_SIZE:=32}"
: "${SAVE_EVERY:=2000}"
: "${LOG_EVERY:=100}"
: "${DEVICE:=auto}"

echo "[$(date '+%F %T')] sonnet M3-v1 MLM train  steps=$MAX_STEPS  bsz=$BATCH_SIZE  device=$DEVICE" | tee "$LOG"

python -u scripts/train_m3v1_mlm_sonnet.py \
  --vocab_dir diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355 \
  --corpus ../datasets/sonnet3355/sonnet_train.txt \
  --out_dir diffusion_models/m3v1_mlm_sonnet3355 \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --save_every "$SAVE_EVERY" \
  --log_every "$LOG_EVERY" \
  --device "$DEVICE" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done" | tee -a "$LOG"
