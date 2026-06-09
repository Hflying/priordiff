#!/usr/bin/env bash
# 串行跑 boundary_freeze 推理：300k → 200k
# (400k 已存在，subset10 共 10 条 × 50 候选，每个 ckpt 约 75-90 min)

set -euo pipefail
cd ${PROJ_ROOT:?Please set PROJ_ROOT to the repo root}

mkdir -p logs

echo "=========================================="
echo " [$(date '+%F %T')] start: 300k bfreeze"
echo "=========================================="
CKPT_STEP=300000 \
INFILL_NOTES=tone_vowel_length_sz12_subset10_step300k_bfreeze \
BOUNDARY_FREEZE=true \
SAMPLE_RESUME=false \
bash bash/ci_control_tone_vowel_sz12.sh

echo ""
echo "=========================================="
echo " [$(date '+%F %T')] start: 200k bfreeze"
echo "=========================================="
CKPT_STEP=200000 \
INFILL_NOTES=tone_vowel_length_sz12_subset10_step200k_bfreeze \
BOUNDARY_FREEZE=true \
SAMPLE_RESUME=false \
bash bash/ci_control_tone_vowel_sz12.sh

echo ""
echo "=========================================="
echo " [$(date '+%F %T')] all done"
echo "=========================================="
