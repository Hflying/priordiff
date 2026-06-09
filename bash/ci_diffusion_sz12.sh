#!/usr/bin/env bash
# 训练扩散模型（image_size=12，seqlen=144），覆盖更长的词牌（≈99.5%）。
# 与原 ci_diffusion.sh 区别：
#   - image_size: 8  -> 12   （seqlen: 64 -> 144）
#   - notes:      ci -> ci_sz12_gpu
#   - 输出目录：  diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/
#                （独立目录，不会覆盖旧的 _ci_gpu 模型）
#
# 用法（在仓库根目录执行）：
#   bash bash/ci_diffusion_sz12.sh
# 后台 + 落日志（推荐）：
#   mkdir -p logs
#   nohup bash bash/ci_diffusion_sz12.sh \
#     > logs/diffusion_sz12_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# 续训（修改 CKPT_STEP 即可，从该步开始）：
#   CKPT_STEP=050000 bash bash/ci_diffusion_sz12.sh

set -euo pipefail
cd "$(dirname "$0")/../improved-diffusion"

# 激活 conda 环境（兼容 nohup / 非交互 shell，不依赖 ~/.bashrc）
CONDA_ENV="${CONDA_ENV:-priordiff}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
echo "[env] python = $(which python)"

# 不向 W&B 联网（train.py 硬编码 wandb.init；offline 离线写文件，disabled 直接 no-op）
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
mkdir -p "$WANDB_DIR"

# ---- 训练超参（沿用原 _ci_gpu 设置，仅改 image_size + notes）----
IMAGE_SIZE="${IMAGE_SIZE:-12}"
DIFF_STEPS="${DIFF_STEPS:-2000}"
IN_CHANNEL="${IN_CHANNEL:-16}"
LR="${LR:-0.0001}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-200000}"
SEED="${SEED:-102}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-sqrt}"
NOTES="${NOTES:-ci_sz12_gpu}"
DATASET_DIR="${DATASET_DIR:-../datasets/ci8w}"
VOCAB_SIZE="${VOCAB_SIZE:-5049}"

# ---- 续训选项 ----
# 留空 = 从头训练；否则填 6 位补零的步数（如 050000）
CKPT_STEP="${CKPT_STEP:-}"

CKPT_ARG=""
if [[ -n "$CKPT_STEP" ]]; then
  CKPT_ARG="--checkpoint $CKPT_STEP"
fi

# 输出目录（与 run_train.py 内部命名规则一致），train.py 不会自动 mkdir，预先建好
OUT_DIR="diffusion_models/diff_e2e-tgt_block_${DIFF_STEPS}steps_${IN_CHANNEL}dim_${NOTES}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " CI Diffusion training (extended seqlen)"
echo "   image_size      : $IMAGE_SIZE  (seqlen = $((IMAGE_SIZE * IMAGE_SIZE)))"
echo "   diffusion_steps : $DIFF_STEPS"
echo "   in_channel      : $IN_CHANNEL"
echo "   lr_anneal_steps : $LR_ANNEAL_STEPS"
echo "   notes           : $NOTES"
echo "   output dir      : diffusion_models/diff_e2e-tgt_block_${DIFF_STEPS}steps_${IN_CHANNEL}dim_${NOTES}/"
echo "   resume from     : ${CKPT_STEP:-(scratch)}"
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
  --padding_mode block \
  --app "--predict_xstart True --training_mode e2e --vocab_size $VOCAB_SIZE --e2e_train $DATASET_DIR --notes $NOTES " \
  --notes "$NOTES" \
  $CKPT_ARG
