#!/usr/bin/env bash
# 平仄 + 韵部 + 句读 控制生成（sz12 版本）。
#
# 使用：
#   - 扩散模型：diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu (image_size=12, seqlen=144)
#   - 分类器：classifier_models/ci-{tone,vowel}_sz12_epochs=20
#   - 目标：默认随机抽 10 个词牌 (control_gen/target_ci_subset10.json)
#
# 关键点：
#   1) infill.py 启动时会从 model_path 同目录的 training_args.json 自动读
#      image_size=12，所以这里不需要显式传 --image_size。
#   2) eval_task_ 名带 length，避免 infill 内部 246/256 行硬编码 image_size=8/14。
#   3) 输出目录用 _sz12 后缀，避免覆盖旧 _ci_gpu 的输出。
#
# 启动：
#   cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}
#   bash bash/ci_control_tone_vowel_sz12.sh 2>&1 | tee logs/infill_sz12_$(date +%Y%m%d_%H%M).log
# 后台：
#   nohup bash bash/ci_control_tone_vowel_sz12.sh \
#     > logs/infill_sz12_$(date +%Y%m%d_%H%M).log 2>&1 &

set -euo pipefail
PROJ_ROOT="${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}"

# ---- conda env ----
CONDA_ENV="${CONDA_ENV:-priordiff}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
echo "[env] python = $(which python)"

# ---- 模型 / 分类器 ----
RUN_NAME="${RUN_NAME:-diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu}"
CKPT_STEP="${CKPT_STEP:-200000}"
CLASSIFIER_TONE="${CLASSIFIER_TONE:-ci-tone_sz12_epochs=20}"
CLASSIFIER_VOWEL="${CLASSIFIER_VOWEL:-ci-vowel_sz12_epochs=20}"

# ---- 生成目标 ----
TGT_FILE="${TGT_FILE:-target_ci_subset10.json}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"   # 每个词牌生成 N 个候选

# ---- 输出 / 检查点 ----
INFILL_NOTES="${INFILL_NOTES:-tone_vowel_length_sz12_subset10}"
SAMPLE_CKPT_EVERY="${SAMPLE_CKPT_EVERY:-${CHECKPOINT_EVERY:-2}}"
SAMPLE_RESUME="${SAMPLE_RESUME:-${RESUME_FROM_CHECKPOINT:-false}}"
SAMPLE_CKPT_PATH="${SAMPLE_CKPT_PATH:-out_gen/control_tone_vowel_length/checkpoint_${INFILL_NOTES}.pkl}"

# ---- 边界冻结开关（用于 PAD-loss-masking 替代实验 / boundary-freeze 验证）----
# true: target 长度 L 之后所有位置 freeze 为 END，避免模型生成「下一首词」的噪声
# false: 保持原行为（[L, SEQ_LEN) 自由生成，会出现 END START + 下一首词的内容）
BOUNDARY_FREEZE="${BOUNDARY_FREEZE:-false}"

cd "$PROJ_ROOT/improved-diffusion"
mkdir -p out_gen/control_tone_vowel_length

# 校验关键文件是否存在
MODEL_PT="diffusion_models/${RUN_NAME}/ema_0.9999_${CKPT_STEP}.pt"
TARGS_JSON="diffusion_models/${RUN_NAME}/training_args.json"
TONE_DIR="../classifier_models/${CLASSIFIER_TONE}"
VOWEL_DIR="../classifier_models/${CLASSIFIER_VOWEL}"
TGT_PATH="control_gen/${TGT_FILE}"

for p in "$MODEL_PT" "$TARGS_JSON" "$TONE_DIR" "$VOWEL_DIR" "$TGT_PATH"; do
  if [[ ! -e "$p" ]]; then
    echo "[err] missing: $p" >&2
    exit 1
  fi
done

# 从 training_args.json 提一下 image_size 印到日志，便于核对
IMG_SIZE=$(python -c "import json,sys; print(json.load(open('${TARGS_JSON}'))['image_size'])")
SEQ_LEN=$((IMG_SIZE * IMG_SIZE))

echo "============================================================"
echo " CI controlled generation (sz12 subset)"
echo "   model         : $MODEL_PT  (image_size=$IMG_SIZE, seqlen=$SEQ_LEN)"
echo "   tone clf      : $TONE_DIR"
echo "   vowel clf     : $VOWEL_DIR"
echo "   tgt file      : $TGT_PATH  ($(wc -l < $TGT_PATH) tasks)"
echo "   num_samples   : $NUM_SAMPLES per task"
echo "   infill_notes  : $INFILL_NOTES"
echo "   ckpt every    : $SAMPLE_CKPT_EVERY tasks  -> $SAMPLE_CKPT_PATH"
echo "   resume        : $SAMPLE_RESUME"
echo "   boundary_frz  : $BOUNDARY_FREEZE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/infill.py \
  --model_path "$MODEL_PT" \
  --eval_task_ control_tone_vowel_length \
  --use_ddim True \
  --infill_notes "$INFILL_NOTES" \
  --eta 1. \
  --verbose pipe \
  --classifier_model_name "$CLASSIFIER_TONE" \
  --classifier_model_name_2 "$CLASSIFIER_VOWEL" \
  --num_samples "$NUM_SAMPLES" \
  --print_middle_sent False \
  --change_num_steps 200 \
  --tgt_file "$TGT_FILE" \
  --sample_ckpt_every "$SAMPLE_CKPT_EVERY" \
  --sample_ckpt_path "$SAMPLE_CKPT_PATH" \
  --sample_resume "$SAMPLE_RESUME" \
  --boundary_freeze "$BOUNDARY_FREEZE"
