#!/usr/bin/env bash
# 训练宋词格律分类器（平仄 / 韵部），供 infill.py 控制生成使用。
# 须在仓库根目录执行：bash bash/ci_classifier.sh tone   或   bash bash/ci_classifier.sh vowel
#
# 注意：--experiment 必须是 ci-tone 或 ci-vowel（带连字符），不能写成 ci_tone_length。

set -euo pipefail
cd "$(dirname "$0")/.."
# 不向 Weights & Biases 联网登录（用 --report_to none，避免 WANDB_DISABLED 弃用告警）
export WANDB_MODE=offline
MODE="${1:-tone}"

PROJ="$(pwd)"
DIFF_DIR="${DIFF_DIR:-$PROJ/improved-diffusion/diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_gpu}"
CI_DATA="${CI_DATA:-$PROJ/datasets/ci8w}"

if [[ "$MODE" == "tone" ]]; then
  EXP="ci-tone"
  OUT="${CLASSIFIER_OUT:-$PROJ/classifier_models/ci-tone_epochs=20}"
elif [[ "$MODE" == "vowel" ]]; then
  EXP="ci-vowel"
  OUT="${CLASSIFIER_OUT:-$PROJ/classifier_models/ci-vowel_epochs=20}"
else
  echo "用法: bash bash/ci_classifier.sh [tone|vowel]"
  exit 1
fi

mkdir -p "$OUT"

python transformers/examples/pytorch/language-modeling/run_clm.py \
  --report_to none \
  --output_dir="$OUT" \
  --model_name_or_path=bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 40 \
  --save_steps 10000 \
  --num_train_epochs 6 \
  --do_train \
  --eval_steps 2000 \
  --evaluation_strategy steps \
  --do_eval \
  --dataloader_num_workers 1 \
  --save_total_limit 1 \
  --overwrite_output_dir \
  --logging_dir "/tmp/classifier_logs_${MODE}" \
  --block_size 100 \
  --disable_tqdm True \
  --model_type gpt2 \
  --gradient_accumulation_steps 1 \
  --seed 101 \
  --experiment "$EXP" \
  --e2e_train "$CI_DATA" \
  --dataset_name=wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --task wp \
  --init_emb "$DIFF_DIR" \
  --e_step 200000 \
  --n_embd 16 \
  --learned_emb yes \
  --diffusion_steps 200
