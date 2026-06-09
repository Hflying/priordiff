#!/usr/bin/env bash
# §6.5 transfer-curve — from-finetune (sonnet→ci) side.
#
# Watch the ci_initFromSonnet diffusion-models dir for new EMA snapshots
# at the canonical step counts {10k, 25k, 50k}, and as each appears, run
# the same classifier-guided infill pipeline used for the from-scratch
# baseline (bash/ci_control_tone_vowel_sz12.sh) — but with RUN_NAME
# pointed at the finetuned model dir, and INFILL_NOTES tagged with
# "_initFromSonnet" so outputs do not collide with the baseline.
#
# Once each step's *.json is written, also auto-trigger the
# transfer-curve eval just like ci_transfer_curve_eval_watch.sh does
# for the from-scratch side (delegated to a final eval call at the
# end of the script).

set -uo pipefail
PROJ_ROOT="${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}"
cd "$PROJ_ROOT"

CONDA_ENV="${CONDA_ENV:-priordiff}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

STEPS_K=(${STEPS_K:-10 25 50})
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_sz12_initFromSonnet}"
RUN_DIR="improved-diffusion/diffusion_models/${RUN_NAME}"
OUT_DIR="improved-diffusion/out_gen/control_tone_vowel_length"
DONE_DIR="logs/ft_curve_done"
mkdir -p "$DONE_DIR"

run_infill_for_step() {
    local step=$1
    local step_padded=$(printf '%06d' $((step*1000)))
    local step_k=${step}k
    local notes="tone_vowel_length_sz12_subset10_step${step_k}_bfreeze_initFromSonnet"
    local outfile="${OUT_DIR}/infill_${notes}_subset10.json"
    if [ -s "$outfile" ]; then
        echo "[$(date '+%T')] [step ${step_k}] outfile exists → skip"
        return 0
    fi
    echo "[$(date '+%T')] [step ${step_k}] launching infill"
    LOG="logs/ci_ft_infill_${step_k}_$(date +%Y%m%d_%H%M%S).log"
    env CKPT_STEP="$step_padded" \
        INFILL_NOTES="$notes" \
        BOUNDARY_FREEZE=true \
        RUN_NAME="$RUN_NAME" \
        bash bash/ci_control_tone_vowel_sz12.sh > "$LOG" 2>&1
    local rc=$?
    echo "[$(date '+%T')] [step ${step_k}] rc=$rc log=$LOG outfile=$([ -s $outfile ] && echo OK || echo MISSING)"
    return $rc
}

while true; do
    pending=0
    for step in "${STEPS_K[@]}"; do
        sentinel="${DONE_DIR}/step${step}k.done"
        if [ -s "$sentinel" ]; then continue; fi
        step_padded=$(printf '%06d' $((step*1000)))
        ckpt="${RUN_DIR}/ema_0.9999_${step_padded}.pt"
        if [ ! -e "$ckpt" ]; then
            pending=$((pending+1))
            continue
        fi
        echo "[$(date '+%T')] step ${step}k EMA exists → run infill"
        run_infill_for_step "$step" || { pending=$((pending+1)); continue; }
        touch "$sentinel"
    done
    if [ "$pending" -eq 0 ]; then
        echo "[$(date '+%T')] all finetune steps infill-done."
        break
    fi
    echo "[$(date '+%T')] $pending finetune steps still pending; sleep 600s"
    sleep 600
done

echo "[$(date '+%T')] finetune infill driver done."
