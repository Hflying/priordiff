#!/usr/bin/env bash
# 典故贴合度批量评测（Phase 1/2 自动指标）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KB="${KB:-datasets/allusion_kb/allusion_v1.json}"
OUT_DIR="${OUT_DIR:-results}"
mkdir -p "$OUT_DIR"

echo "=== 典故评测集 (labeled) ==="
python eval/run_allusion_batch.py \
  --kb "$KB" \
  --eval_set eval/allusion_eval_set.json \
  --out "$OUT_DIR/allusion_eval_set_metrics.json"

echo ""
echo "=== v0 KB 基线（同评测集） ==="
python eval/run_allusion_batch.py \
  --kb datasets/allusion_kb/allusion_v0.json \
  --eval_set eval/allusion_eval_set.json \
  --out "$OUT_DIR/allusion_eval_set_v0_metrics.json"

echo ""
echo "[done] 结果: $OUT_DIR/allusion_eval_set_*_metrics.json"
