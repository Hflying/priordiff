#!/usr/bin/env bash
# E3 watcher: as each E2E→ci finetune ckpt (10k/20k/30k/40k/50k) appears,
# automatically run infill + LCVR + metrics dump.
# Emits to doc/e3_e2e_to_ci.md on completion of each step.
#
# Usage:
#   nohup bash bash/e3_watcher.sh > logs/e3_watcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

PROJ_ROOT=${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}
cd "$PROJ_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_NAME=diff_e2e-tgt_block_2000steps_16dim_ci_sz12_initFromE2E
TGT_JSON=control_gen/target_ci_subset10.json
CORPUS=../datasets/ci8w/ci_train.txt
OUT_DIR=improved-diffusion/out_gen/control_tone_vowel_length
DONE_DIR=logs/e3_done
mkdir -p "$DONE_DIR" "$OUT_DIR"

STEPS=(10 20 30 40 50)

log()  { echo "[$(date '+%T')] $*"; }

run_infill_step() {
    local step=$1
    local step_k="${step}k"
    local sentinel="$DONE_DIR/step${step_k}.done"
    [ -f "$sentinel" ] && return 0

    local ema_file="improved-diffusion/diffusion_models/${RUN_NAME}/ema_0.9999_$(printf '%06d' $((step*1000))).pt"
    [ ! -f "$ema_file" ] && return 1

    # path relative to improved-diffusion/ after we cd below
    local ema_file_rel="diffusion_models/${RUN_NAME}/ema_0.9999_$(printf '%06d' $((step*1000))).pt"

    log "step ${step_k} EMA exists → run infill"
    local pkl_path="out_gen/control_tone_vowel_length/checkpoint_tone_vowel_length_sz12_subset10_step${step_k}_bfreeze_initFromE2E.pkl"
    local baseline_out="out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step${step_k}_bfreeze_initFromE2E_subset10.json"

    cd improved-diffusion
    if [ ! -s "$baseline_out" ]; then
        log "  [$step_k] running baseline infill"
        env CKPT_STEP=$(printf '%06d' $((step*1000))) \
            INFILL_NOTES=tone_vowel_length_sz12_subset10_step${step_k}_bfreeze_initFromE2E \
            BOUNDARY_FREEZE=true \
            RUN_NAME="$RUN_NAME" \
            bash ../bash/ci_control_tone_vowel_sz12.sh \
            > "../logs/e3_infill_${step_k}_$(date +%Y%m%d_%H%M%S).log" 2>&1
        if [ ! -s "$baseline_out" ]; then
            log "  [$step_k] baseline infill FAILED → skip LCVR"
            cd "$PROJ_ROOT"
            return 1
        fi
    fi

    # LCVR on the .pkl latents
    for variant_pair in "none:no_freq" "drop_low_zero:drop_low_zero"; do
        local rf="${variant_pair%%:*}"
        local tag="${variant_pair##*:}"
        local lcvr_out="out_gen/control_tone_vowel_length/infill_lcvr_${step_k}_bfreeze_initFromE2E_${tag}.json"
        if [ -s "$lcvr_out" ]; then
            log "  [$step_k $tag] LCVR exists → skip"
            continue
        fi
        [ ! -s "$pkl_path" ] && log "  [$step_k $tag] pkl missing → skip LCVR" && continue
        log "  [$step_k $tag] running LCVR"
        python scripts/decode_pkl_lcvr.py \
            --model_path "$ema_file_rel" \
            --pkl "$pkl_path" \
            --target_json "$TGT_JSON" \
            --corpus "$CORPUS" \
            --restrict_freq "$rf" \
            --out "$lcvr_out" 2>&1 | tail -3
    done
    cd "$PROJ_ROOT"

    touch "$sentinel"
    log "step ${step_k} done"
    return 0
}

while true; do
    n_done=0
    for s in "${STEPS[@]}"; do
        [ -f "$DONE_DIR/step${s}k.done" ] && n_done=$((n_done+1))
    done
    if [ "$n_done" -ge "${#STEPS[@]}" ]; then
        log "all E3 steps done; exiting"
        break
    fi
    for s in "${STEPS[@]}"; do
        run_infill_step "$s"
    done
    log "$((${#STEPS[@]} - n_done)) E3 steps pending; sleep 900s"
    sleep 900
done
