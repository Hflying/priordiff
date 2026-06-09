#!/usr/bin/env bash
# M1 BMT (Boundary-aware Mixed Training) 训练 — 最简版本：
#   把 ci 训练数据的 padding_mode 从 'block' 改为 'pad'，
#   这等价于把"句子拼接连续流"改成"每首词独立成行 + END/PAD 边界"。
#
# 预期效果（依据 sonnet pad-mode 已观测）：
#   - tail_outsider 显著下降（ci block 74% → 类似 sonnet 8%）
#   - head→tail attention 漂移减弱
#   - head_uniq / common 略降（信息密度变低，是预期 trade-off）
#
# 与 ci_diffusion_sz12.sh 的唯一差别：
#   --padding_mode pad  (而非 block)
#   --notes ci_bmt_sz12 (避免与现有 ckpt 冲突)
#
# 用法：
#   nohup bash bash/ci_diffusion_bmt.sh \
#       > logs/ci_bmt_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# 也支持从已有 ci 200k EMA fork-train 来节省时间：
#   FORK_FROM=diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/ema_0.9999_200000.pt \
#       bash bash/ci_diffusion_bmt.sh

set -euo pipefail
cd "$(dirname "$0")/../improved-diffusion"

CONDA_ENV="${CONDA_ENV:-priordiff}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
echo "[env] python = $(which python)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
mkdir -p "$WANDB_DIR"

IMAGE_SIZE="${IMAGE_SIZE:-12}"
DIFF_STEPS="${DIFF_STEPS:-2000}"
IN_CHANNEL="${IN_CHANNEL:-16}"
LR="${LR:-0.0001}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-100000}"   # BMT 训练 100k 即可（已有 backbone 起步）
SEED="${SEED:-103}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-sqrt}"
NOTES="${NOTES:-ci_bmt_sz12}"
DATASET_DIR="${DATASET_DIR:-../datasets/ci8w}"
VOCAB_SIZE="${VOCAB_SIZE:-5049}"
PADDING_MODE="${PADDING_MODE:-pad}"

# 可选 fork-from
FORK_FROM="${FORK_FROM:-}"
RESUME_ARG=""
if [[ -n "$FORK_FROM" ]]; then
    if [[ ! -s "$FORK_FROM" ]]; then
        echo "[error] FORK_FROM not found: $FORK_FROM"; exit 1
    fi
    RESUME_ARG="--resume_checkpoint $FORK_FROM"
    echo "[info] fork-from: $FORK_FROM"
fi

OUT_DIR="diffusion_models/diff_e2e-tgt_${PADDING_MODE}_${DIFF_STEPS}steps_${IN_CHANNEL}dim_${NOTES}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " M1 BMT training"
echo "   image_size      : $IMAGE_SIZE  (seqlen = $((IMAGE_SIZE * IMAGE_SIZE)))"
echo "   padding_mode    : $PADDING_MODE  (BMT key change)"
echo "   lr_anneal_steps : $LR_ANNEAL_STEPS"
echo "   notes           : $NOTES"
echo "   output dir      : $OUT_DIR"
echo "   fork-from       : ${FORK_FROM:-(scratch)}"
echo "============================================================"

python scripts/run_train.py \
  --diff_steps "$DIFF_STEPS" \
  --model_arch transformer \
  --lr "$LR" \
  --lr_anneal_steps "$LR_ANNEAL_STEPS" \
  --seed "$SEED" \
  --noise_schedule "$NOISE_SCHEDULE" \
  --image_size "$IMAGE_SIZE" \
  --in_channel "$IN_CHANNEL" \
  --modality e2e-tgt \
  --submit no \
  --padding_mode "$PADDING_MODE" \
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train $DATASET_DIR --notes $NOTES $RESUME_ARG --save_interval 25000 " \
  --notes "$NOTES"
