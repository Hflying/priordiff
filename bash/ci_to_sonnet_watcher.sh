#!/usr/bin/env bash
# ci_to_sonnet watcher: as each sonnet-from-ci finetune ckpt (10k/20k/30k/40k/50k)
# appears, run sonnet sample_lcvr + eval automatically.
#
# Usage:
#   nohup bash bash/ci_to_sonnet_watcher.sh > logs/c2s_watcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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

CKPT_DIR=improved-diffusion/diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet_initFromCi
TGT_JSON=control_gen/target_son10.json
DONE_DIR=logs/c2s_done
mkdir -p "$DONE_DIR"

STEPS=(10 20 30 40 50)

log()  { echo "[$(date '+%T')] $*"; }

run_step() {
    local step=$1
    local step_k="${step}k"
    local sentinel="$DONE_DIR/step${step_k}.done"
    [ -f "$sentinel" ] && return 0

    local step_full=$(printf '%06d' $((step*1000)))
    local ema_file="${CKPT_DIR}/ema_0.9999_${step_full}.pt"
    [ ! -f "$ema_file" ] && return 1

    log "step ${step_k} EMA exists -> run sample+eval"

    local out_json="out_gen/sonnet_son10_c2s_step${step_k}.json"
    local eval_md="control_gen/sonnet_compare_c2s_step${step_k}.md"
    local decoded_dir="control_gen/decoded_son10_c2s_step${step_k}"
    local elog="logs/c2s_eval_step${step_k}_$(date +%Y%m%d_%H%M%S).log"

    cd improved-diffusion
    # sonnet_sample_lcvr 会同时产出 baseline + LCVR 两列（ref wait_sonnet_ckpt_eval.sh）
    if [ ! -s "../$out_json" ] && [ ! -s "$out_json" ]; then
        log "  [$step_k] sample+LCVR -> $out_json"
        python -u scripts/sonnet_sample_lcvr.py \
            --ckpt "diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet_initFromCi/ema_0.9999_${step_full}.pt" \
            --target_json "$TGT_JSON" \
            --num_samples 50 --batch_size 25 \
            --out "$out_json" 2>&1 | tail -5 | tee -a "../$elog"
    fi
    if [ ! -s "$out_json" ]; then
        log "  [$step_k] sample FAILED -> skip eval"
        cd "$PROJ_ROOT"
        return 1
    fi

    log "  [$step_k] eval -> $eval_md"
    python -u scripts/sonnet_eval.py \
        --out_json "$out_json" \
        --target_json "$TGT_JSON" \
        --out_md "$eval_md" \
        --out_dir "$decoded_dir" 2>&1 | tail -10 | tee -a "../$elog"
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
        log "all ci->sonnet steps done; exiting"
        break
    fi
    for s in "${STEPS[@]}"; do
        run_step "$s"
    done
    log "$((${#STEPS[@]} - n_done)) steps pending; sleep 900s"
    sleep 900
done
