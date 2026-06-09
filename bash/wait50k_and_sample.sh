#!/usr/bin/env bash
# Wait until the running ci_gpu training reaches step 50000, stop it,
# then sample 500 poems from the ema_0.9999_050000.pt checkpoint and
# print the first few generated samples.
#
# Usage: bash bash/wait50k_and_sample.sh
set -uo pipefail

CONDA_ENV="${CONDA_ENV:-priordiff}"
PROJ_ROOT="${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}"
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_gpu}"
CKPT_TAG="${CKPT_TAG:-050000}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-50}"
TOP_P="${TOP_P:--1.0}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/priordiff_gpu.log}"
SAMPLE_LOG="${SAMPLE_LOG:-/tmp/priordiff_sample.log}"
POLL_SEC="${POLL_SEC:-60}"

CKPT_DIR="$PROJ_ROOT/improved-diffusion/diffusion_models/$RUN_NAME"
EMA_CKPT="$CKPT_DIR/ema_0.9999_${CKPT_TAG}.pt"

echo "[$(date +%T)] activating conda env: $CONDA_ENV"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "[$(date +%T)] watching for checkpoint: $EMA_CKPT"
while [ ! -f "$EMA_CKPT" ]; do
  cur_step="$(awk '/^\| step / {s=$4} END{print s+0}' "$TRAIN_LOG" 2>/dev/null || echo 0)"
  cur_loss="$(awk '/^\| loss / {s=$4} END{print s+0}' "$TRAIN_LOG" 2>/dev/null || echo 0)"
  echo "[$(date +%T)]   step ~ $cur_step   loss ~ $cur_loss   (waiting for $CKPT_TAG)"
  sleep "$POLL_SEC"
done

echo "[$(date +%T)] checkpoint found, stopping training..."
pkill -f "$RUN_NAME" 2>/dev/null || true
sleep 3
ls -lh "$CKPT_DIR" | grep -E "$CKPT_TAG|vocab|training_args|random_emb" || true

cd "$PROJ_ROOT/improved-diffusion"
mkdir -p generation_outputs

echo "[$(date +%T)] sampling $NUM_SAMPLES poems (batch=$BATCH_SIZE, top_p=$TOP_P)..."
WANDB_MODE=offline python scripts/text_sample.py \
  --model_path "diffusion_models/$RUN_NAME/ema_0.9999_${CKPT_TAG}.pt" \
  --batch_size "$BATCH_SIZE" \
  --num_samples "$NUM_SAMPLES" \
  --top_p "$TOP_P" \
  --out_dir generation_outputs 2>&1 | tee "$SAMPLE_LOG"

SAMPLES_FILE=$(ls -t generation_outputs/${RUN_NAME}*ema_0.9999_${CKPT_TAG}*samples_${TOP_P}.json 2>/dev/null | head -1)
if [ -z "$SAMPLES_FILE" ]; then
  echo "[$(date +%T)] WARNING: samples json not found in generation_outputs/"
  ls -t generation_outputs | head
  exit 1
fi

echo "[$(date +%T)] samples written to: $SAMPLES_FILE"
echo "----- first 10 generated poems -----"
python - "$SAMPLES_FILE" <<'PY'
import json, sys
path = sys.argv[1]
data = [json.loads(l) for l in open(path)]
print(f"total samples: {len(data)}\n")
for i, x in enumerate(data[:10]):
    if isinstance(x, list):
        toks = x[0] if x and isinstance(x[0], list) else x
        text = ''.join(toks) if all(isinstance(t, str) for t in toks) else ' '.join(map(str, toks))
    else:
        text = str(x)
    print(f"--- sample {i+1} ---")
    print(text)
    print()
PY

echo "[$(date +%T)] done."
