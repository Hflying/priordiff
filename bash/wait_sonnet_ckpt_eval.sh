#!/usr/bin/env bash
# 等 sonnet 25k step ckpt 出现，自动跑：
#   1. unconditional sample + LCVR decode
#   2. baseline vs LCVR 评测
# 后台启动后可以离开终端。

set -uo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

CKPT_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355"
TARGET_STEP="${TARGET_STEP:-25000}"
SLEEP_SEC="${SLEEP_SEC:-300}"

CKPT_PATH="${CKPT_DIR}/ema_0.9999_$(printf '%06d' "$TARGET_STEP").pt"
echo "[$(date '+%F %T')] watching for $CKPT_PATH"

while [ ! -s "$CKPT_PATH" ]; do
    sleep "$SLEEP_SEC"
    if [ -s "$CKPT_PATH" ]; then
        break
    fi
    LATEST=$(ls -t ${CKPT_DIR}/ema_0.9999_*.pt 2>/dev/null | head -1 || true)
    echo "[$(date '+%F %T')] waiting; latest ckpt=$LATEST"
done

echo "[$(date '+%F %T')] ckpt detected, sleep 30s for safe write"
sleep 30

cd improved-diffusion
mkdir -p out_gen logs

OUT_JSON="out_gen/sonnet_son10_${TARGET_STEP}.json"
LOG="../logs/sonnet_eval_${TARGET_STEP}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] sample + LCVR decode -> $OUT_JSON"
TRANSFORMERS_OFFLINE=1 python -u scripts/sonnet_sample_lcvr.py \
    --ckpt "${CKPT_PATH#improved-diffusion/}" \
    --target_json control_gen/target_son10.json \
    --num_samples 50 --batch_size 25 \
    --out "$OUT_JSON" 2>&1 | tee "$LOG"

echo "[$(date '+%F %T')] eval -> control_gen/sonnet_compare_${TARGET_STEP}.md"
python -u scripts/sonnet_eval.py \
    --out_json "$OUT_JSON" \
    --target_json control_gen/target_son10.json \
    --out_md "control_gen/sonnet_compare_${TARGET_STEP}.md" \
    --out_dir "control_gen/decoded_son10_step${TARGET_STEP}" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] sonnet ckpt $TARGET_STEP eval done"
