#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"

MANIFEST="${1:-data/manifests/train_debug.jsonl}"
OUT_ROOT="${2:-outputs/no_train_diagnostics}"

ENCODERS=(
  clip-vit-l-14-336
  siglip-so400m
  siglip2-so400m
  internvit-300m
  dinov2-vitl14
  videomaev2-base
  vjepa2-vith-256
  internvideo2-clip-1b
)

for ENCODER in "${ENCODERS[@]}"; do
  echo "==> No-train diagnostics for ${ENCODER}"
  python scripts/analyze_encoder_no_train.py \
    --manifest "${MANIFEST}" \
    --encoder "${ENCODER}" \
    --out-jsonl "${OUT_ROOT}/${ENCODER}/per_example.jsonl" \
    --out-csv "${OUT_ROOT}/${ENCODER}/summary.csv"
done
