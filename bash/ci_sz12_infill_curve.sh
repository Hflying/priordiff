#!/usr/bin/env bash
# Sequentially run ci_control_tone_vowel_sz12 infill on a list of EMA ckpts.
# Used for §6.5 from-scratch transfer-curve baseline:
#   from-scratch ci_sz12_gpu @ {25k, 50k, 75k, 100k} EMA
# Output: out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step{N}k_bfreeze_subset10.json
# (matches naming used by compare_lcvr.py)

set -uo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

# steps to evaluate (skip ones with existing .json)
STEPS=(${STEPS:-25000 50000 75000 100000})
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu}"

OUTDIR=improved-diffusion/out_gen/control_tone_vowel_length

for STEP in "${STEPS[@]}"; do
    STEP_K=$((STEP / 1000))k
    NOTES="tone_vowel_length_sz12_subset10_step${STEP_K}_bfreeze"
    OUTFILE="${OUTDIR}/infill_${NOTES}_subset10.json"
    if [ -s "$OUTFILE" ]; then
        echo "[$(date '+%F %T')] skip step=$STEP — output exists: $OUTFILE"
        continue
    fi
    echo "[$(date '+%F %T')] ==== infill step=$STEP (notes=$NOTES) ===="
    LOG="logs/ci_sz12_infill_${STEP_K}_$(date +%Y%m%d_%H%M%S).log"
    env CKPT_STEP=$(printf '%06d' "$STEP") \
        INFILL_NOTES="$NOTES" \
        BOUNDARY_FREEZE=true \
        RUN_NAME="$RUN_NAME" \
        bash bash/ci_control_tone_vowel_sz12.sh > "$LOG" 2>&1
    RC=$?
    echo "[$(date '+%F %T')] step=$STEP rc=$RC log=$LOG outfile_exists=$([ -s $OUTFILE ] && echo yes || echo no)"
done

echo "[$(date '+%F %T')] all steps done"
