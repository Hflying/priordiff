#!/usr/bin/env bash
# Finetune sonnet from ci-400k backbone.
# Reverse-direction counterpart of ci_finetune_from_sonnet.sh.
#
# Usage:
#   nohup bash bash/sonnet_finetune_from_ci.sh > logs/sonnet_ft_from_ci_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate priordiff
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

cd improved-diffusion
python scripts/train.py \
    --checkpoint_path diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet_initFromCi \
    --model_arch transformer \
    --modality e2e-tgt \
    --save_interval 10000 \
    --lr 0.0001 \
    --batch_size 64 \
    --diffusion_steps 2000 \
    --noise_schedule sqrt \
    --use_kl False \
    --learn_sigma False \
    --image_size 14 \
    --num_channels 128 \
    --seed 102 \
    --dropout 0.1 \
    --in_channel 16 \
    --out_channel 16 \
    --padding_mode pad \
    --experiment random \
    --lr_anneal_steps 50000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode e2e \
    --vocab_size 22970 \
    --e2e_train ../datasets/sonnet3355 \
    --notes sonnet3355_initFromCi \
    --resume_checkpoint diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet_initFromCi/model000000.pt
