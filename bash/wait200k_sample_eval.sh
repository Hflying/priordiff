#!/usr/bin/env bash
# Wait until ema_0.9999_200000.pt exists, stop the training,
# sample N poems from it, then run scripts/eval_samples.py and print
# a side-by-side comparison with the 50k checkpoint.
#
# Usage: bash bash/wait200k_sample_eval.sh

set -uo pipefail

CONDA_ENV="${CONDA_ENV:-priordiff}"
PROJ_ROOT="${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}"
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_gpu}"
TARGET_TAG="${TARGET_TAG:-200000}"
PREV_TAG="${PREV_TAG:-050000}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-50}"
TOP_P="${TOP_P:--1.0}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/priordiff_gpu_resume.log}"
SAMPLE_LOG="${SAMPLE_LOG:-/tmp/priordiff_sample_200k.log}"
EVAL_LOG="${EVAL_LOG:-/tmp/priordiff_eval_200k.log}"
POLL_SEC="${POLL_SEC:-120}"

CKPT_DIR="$PROJ_ROOT/improved-diffusion/diffusion_models/$RUN_NAME"
EMA_CKPT="$CKPT_DIR/ema_0.9999_${TARGET_TAG}.pt"
PREV_SAMPLES="$PROJ_ROOT/improved-diffusion/generation_outputs/${RUN_NAME}.ema_0.9999_${PREV_TAG}.pt.samples_${TOP_P}.json"
NEW_SAMPLES="$PROJ_ROOT/improved-diffusion/generation_outputs/${RUN_NAME}.ema_0.9999_${TARGET_TAG}.pt.samples_${TOP_P}.json"

echo "[$(date +%T)] activating conda env: $CONDA_ENV"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "[$(date +%T)] watching for checkpoint: $EMA_CKPT"
while [ ! -f "$EMA_CKPT" ]; do
  cur_step="$(awk '/^\| step / {s=$4} END{print s+0}' "$TRAIN_LOG" 2>/dev/null || echo 0)"
  cur_loss="$(awk '/^\| loss / {s=$4} END{print s+0}' "$TRAIN_LOG" 2>/dev/null || echo 0)"
  echo "[$(date +%T)]   step ~ $cur_step   loss ~ $cur_loss   (target $TARGET_TAG)"
  sleep "$POLL_SEC"
done

echo "[$(date +%T)] checkpoint found, stopping training..."
pkill -f "$RUN_NAME" 2>/dev/null || true
sleep 5
ls -lh "$CKPT_DIR" | grep -E "$TARGET_TAG|vocab|training_args|random_emb" || true

cd "$PROJ_ROOT/improved-diffusion"
mkdir -p generation_outputs

echo "[$(date +%T)] sampling $NUM_SAMPLES poems from 200k ckpt (batch=$BATCH_SIZE, top_p=$TOP_P)..."
WANDB_MODE=offline python scripts/text_sample.py \
  --model_path "diffusion_models/$RUN_NAME/ema_0.9999_${TARGET_TAG}.pt" \
  --batch_size "$BATCH_SIZE" \
  --num_samples "$NUM_SAMPLES" \
  --top_p "$TOP_P" \
  --out_dir generation_outputs 2>&1 | tee "$SAMPLE_LOG"

if [ ! -f "$NEW_SAMPLES" ]; then
  echo "[$(date +%T)] WARNING: expected samples not found: $NEW_SAMPLES"
  ls -t generation_outputs | head
  exit 1
fi

echo "[$(date +%T)] running eval_samples.py on 200k ..."
python scripts/eval_samples.py "$NEW_SAMPLES" \
  --vocab ../datasets/ci8w/ci_vocab.txt \
  --tone  ../datasets/ci8w/ci_tone.txt \
  --vowel ../datasets/ci8w/ci_vowel.txt \
  --ref   ../datasets/ci8w/ci_valid.txt 2>&1 | tee "$EVAL_LOG"

if [ -f "$PREV_SAMPLES" ]; then
  echo
  echo "===================================================="
  echo " For comparison: 50k checkpoint metrics"
  echo "===================================================="
  python scripts/eval_samples.py "$PREV_SAMPLES" \
    --vocab ../datasets/ci8w/ci_vocab.txt \
    --tone  ../datasets/ci8w/ci_tone.txt \
    --vowel ../datasets/ci8w/ci_vowel.txt \
    --ref   ../datasets/ci8w/ci_valid.txt 2>&1 | tee -a "$EVAL_LOG"
fi

echo "[$(date +%T)] done. eval saved to $EVAL_LOG"
echo "----- first 5 generated poems (200k) -----"
python - "$NEW_SAMPLES" <<'PY'
import json, sys
data = [json.loads(l) for l in open(sys.argv[1])]
for i, x in enumerate(data[:5]):
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], str):
        text = x[0]
    elif isinstance(x, list):
        text = ' '.join(map(str, x[0] if x and isinstance(x[0], list) else x))
    else:
        text = str(x)
    print(f"--- sample {i+1} ---")
    print(text)
    print()
PY
