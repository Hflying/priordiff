#!/usr/bin/env bash
# 等 sonnet 100k ckpt 出现 → fork backbone 给 ci → finetune ci 50k step。
# 这是路径 A: cross-domain backbone transfer 实验。
#
# 行为：
#   1. 等 ema_0.9999_100000.pt 出现在 sonnet 目录
#   2. 用 init_ci_from_sonnet.py 把 sonnet backbone 迁移到 ci-shaped model
#   3. 启动 ci finetune 50k step (resume from init ckpt)
#
# 后台运行示例：
#   nohup bash bash/ci_finetune_from_sonnet.sh > logs/ci_ft_from_sonnet_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

mkdir -p logs improved-diffusion/diffusion_models

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# ===== 输入 =====
SONNET_CKPT_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355"
SONNET_TARGET_STEP="${SONNET_TARGET_STEP:-100000}"
SONNET_CKPT="${SONNET_CKPT_DIR}/ema_0.9999_$(printf '%06d' "$SONNET_TARGET_STEP").pt"

# 用来获取 ci shape 的 template
CI_TEMPLATE_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu"

# ===== 输出 =====
INIT_DIR="improved-diffusion/diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_initFromSonnet"
FT_LR_STEPS="${FT_LR_STEPS:-50000}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
SLEEP_SEC="${SLEEP_SEC:-300}"

echo "[$(date '+%F %T')] watcher start"
echo "  sonnet ckpt   = $SONNET_CKPT"
echo "  ci template   = $CI_TEMPLATE_DIR"
echo "  init dir      = $INIT_DIR"
echo "  ft lr_steps   = $FT_LR_STEPS"

# ===== 等 sonnet 100k =====
while [ ! -s "$SONNET_CKPT" ]; do
    sleep "$SLEEP_SEC"
    LATEST=$(ls -t ${SONNET_CKPT_DIR}/ema_0.9999_*.pt 2>/dev/null | head -1 || echo "(none)")
    echo "[$(date '+%F %T')] waiting; latest sonnet ckpt = $LATEST"
done

echo "[$(date '+%F %T')] sonnet ckpt detected, sleep 30s for safe write"
sleep 30

# ===== Step 1: backbone transfer =====
echo "[$(date '+%F %T')] forking sonnet backbone -> ci-shaped model"
cd improved-diffusion
python -u scripts/init_ci_from_sonnet.py \
    --sonnet_ckpt "${SONNET_CKPT#improved-diffusion/}" \
    --ci_template_dir "${CI_TEMPLATE_DIR#improved-diffusion/}" \
    --out_dir "${INIT_DIR#improved-diffusion/}" \
    --out_step 0

if [ ! -s "${INIT_DIR#improved-diffusion/}/ema_0.9999_000000.pt" ]; then
    echo "[error] init ckpt missing, abort"
    exit 1
fi

# ===== Step 2: finetune =====
LOG="../logs/ci_ft_from_sonnet_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%F %T')] start ci finetune; log -> $LOG"

# resume_checkpoint 让 train.py 从 init 的 model000000.pt 恢复
python -u scripts/run_train.py \
    --diff_steps 2000 \
    --model_arch transformer \
    --lr 0.0001 \
    --lr_anneal_steps "$FT_LR_STEPS" \
    --seed 102 \
    --noise_schedule sqrt \
    --image_size 12 \
    --in_channel 16 \
    --modality e2e-tgt \
    --submit no \
    --padding_mode block \
    --bsz "$FT_BATCH_SIZE" \
    --app "--predict_xstart True --training_mode e2e --vocab_size 5049 --e2e_train ../datasets/ci8w --notes ci_sz12_initFromSonnet --resume_checkpoint diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_initFromSonnet/model000000.pt --save_interval $SAVE_INTERVAL " \
    --notes ci_sz12_initFromSonnet \
    2>&1 | tee "$LOG"

echo "[$(date '+%F %T')] finetune done"
