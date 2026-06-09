#!/usr/bin/env bash
# 平仄 + 韵部 + 句读 控制生成。需先：
#   bash bash/ci_classifier.sh tone
#   bash bash/ci_classifier.sh vowel
# 且目录名与下面 CLASSIFIER_TONE / CLASSIFIER_VOWEL 一致（默认与 ci_classifier.sh 输出一致）。

set -euo pipefail
PROJ_ROOT="${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}"
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_gpu}"
CKPT_STEP="${CKPT_STEP:-200000}"
CLASSIFIER_TONE="${CLASSIFIER_TONE:-ci-tone_epochs=20}"
CLASSIFIER_VOWEL="${CLASSIFIER_VOWEL:-ci-vowel_epochs=20}"
TGT_FILE="${TGT_FILE:-target_ci.json}"
# 每 N 条把 sample_dict 落盘（pickle），崩溃后可加 SAMPLE_RESUME=true 续跑
# 兼容旧变量名 CHECKPOINT_EVERY / RESUME_FROM_CHECKPOINT
SAMPLE_CKPT_EVERY="${SAMPLE_CKPT_EVERY:-${CHECKPOINT_EVERY:-10}}"
SAMPLE_RESUME="${SAMPLE_RESUME:-${RESUME_FROM_CHECKPOINT:-false}}"
SAMPLE_CKPT_PATH="${SAMPLE_CKPT_PATH:-}"

cd "$PROJ_ROOT/improved-diffusion"
mkdir -p out_gen/control_tone_vowel_length

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/infill.py \
  --model_path "diffusion_models/${RUN_NAME}/ema_0.9999_${CKPT_STEP}.pt" \
  --eval_task_ control_tone_vowel_length \
  --use_ddim True \
  --infill_notes "tone_vowel_length" \
  --eta 1. \
  --verbose pipe \
  --classifier_model_name "$CLASSIFIER_TONE" \
  --classifier_model_name_2 "$CLASSIFIER_VOWEL" \
  --num_samples 50 \
  --print_middle_sent False \
  --change_num_steps 200 \
  --tgt_file "$TGT_FILE" \
  --sample_ckpt_every "$SAMPLE_CKPT_EVERY" \
  --sample_ckpt_path "$SAMPLE_CKPT_PATH" \
  --sample_resume "$SAMPLE_RESUME"
