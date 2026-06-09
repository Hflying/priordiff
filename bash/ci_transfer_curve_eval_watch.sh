#!/usr/bin/env bash
# Watch for ci_sz12 from-scratch infill_*.json outputs to appear, then
# (a) run decode_pkl_lcvr.py with restrict_freq={none,drop_low_zero} and
# (b) feed the {baseline, lcvr_no_freq, lcvr_drop_low_zero} triplet into
#     scripts/eval_ci_transfer_curve.py for §6.5 transfer-curve numbers.
#
# Designed to run idempotently: skips already-evaluated steps.
#
# Usage:
#   bash bash/ci_transfer_curve_eval_watch.sh                  # default 25 50 75 100
#   STEPS_K="25 50 75 100" bash bash/ci_transfer_curve_eval_watch.sh

set -uo pipefail
PROJ_ROOT="${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}"
cd "$PROJ_ROOT"

CONDA_ENV="${CONDA_ENV:-priordiff}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

STEPS_K=(${STEPS_K:-25 50 75 100})
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu}"
TGT_JSON="${TGT_JSON:-control_gen/target_ci_subset10.json}"
CORPUS="${CORPUS:-../datasets/ci8w/ci_train.txt}"

OUT_DIR="improved-diffusion/out_gen/control_tone_vowel_length"
RUN_DIR="improved-diffusion/diffusion_models/${RUN_NAME}"

# Output sentinel file — eval is "done" if this exists & is non-empty
DONE_DIR="logs/transfer_curve_done"
mkdir -p "$DONE_DIR"

run_lcvr_for_step() {
    local step=$1
    local step_k=${step}k
    local pkl_path="${OUT_DIR}/checkpoint_tone_vowel_length_sz12_subset10_step${step_k}_bfreeze.pkl"
    local model_path="${RUN_DIR}/ema_0.9999_$(printf '%06d' $((step*1000))).pt"

    if [ ! -s "$pkl_path" ]; then
        echo "[$(date '+%T')] [step ${step_k}] no pkl yet → skip LCVR"
        return 1
    fi
    if [ ! -e "$model_path" ]; then
        echo "[$(date '+%T')] [step ${step_k}] no model.pt: $model_path"
        return 1
    fi

    cd improved-diffusion
    # Map our 2 reported variants to decode_pkl_lcvr.py's --restrict_freq:
    #   none           → vanilla LCVR (no frequency cut)        → file tag "no_freq"
    #   drop_low_zero  → LCVR + drop chars w/ freq <= f_low(20%) → file tag "drop_low_zero"
    for variant_pair in "none:no_freq" "drop_low_zero:drop_low_zero"; do
        local rf="${variant_pair%%:*}"     # restrict_freq value passed to script
        local tag="${variant_pair##*:}"    # filename tag for downstream eval
        local out_path="out_gen/control_tone_vowel_length/infill_lcvr_${step_k}_bfreeze_${tag}.json"
        if [ -s "$out_path" ]; then
            echo "[$(date '+%T')] [step ${step_k}] LCVR $tag exists → skip"
            continue
        fi
        echo "[$(date '+%T')] [step ${step_k}] running LCVR rf=$rf tag=$tag"
        TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
        python scripts/decode_pkl_lcvr.py \
            --model_path "diffusion_models/${RUN_NAME}/ema_0.9999_$(printf '%06d' $((step*1000))).pt" \
            --pkl "out_gen/control_tone_vowel_length/checkpoint_tone_vowel_length_sz12_subset10_step${step_k}_bfreeze.pkl" \
            --target_json "$TGT_JSON" \
            --corpus "$CORPUS" \
            --restrict_freq "$rf" \
            --out "$out_path" 2>&1 | tail -10
    done
    cd "$PROJ_ROOT"
    return 0
}

# poll loop
while true; do
    pending=0
    for step in "${STEPS_K[@]}"; do
        sentinel="${DONE_DIR}/step${step}k.done"
        if [ -s "$sentinel" ]; then continue; fi
        # baseline must exist (= infill finished + meta says done)
        baseline="${OUT_DIR}/infill_tone_vowel_length_sz12_subset10_step${step}k_bfreeze_subset10.json"
        if [ ! -s "$baseline" ]; then
            pending=$((pending+1))
            continue
        fi
        echo "[$(date '+%T')] step ${step}k baseline ready → run LCVR"
        run_lcvr_for_step "$step" || { pending=$((pending+1)); continue; }
        # mark done after LCVR done
        touch "$sentinel"
    done

    if [ "$pending" -eq 0 ]; then
        echo "[$(date '+%T')] all steps have LCVR ready, running eval_ci_transfer_curve.py"
        cd improved-diffusion
        python scripts/eval_ci_transfer_curve.py \
            --steps "${STEPS_K[@]}" \
            --run_name "$RUN_NAME" \
            --tag "from_scratch" \
            --target_json "$TGT_JSON" \
            --corpus "$CORPUS" \
            --out_md "../doc/crossdomain_baseline_table.md"
        cd "$PROJ_ROOT"
        echo "[$(date '+%T')] eval done. exiting."
        break
    fi
    echo "[$(date '+%T')] $pending steps still pending; sleep 300s"
    sleep 300
done
