#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-data/manifests/train_230k.jsonl}"
FEATURE_ROOT="${2:-features/train_230k}"
OUT_ROOT="${3:-checkpoints/projectors}"
LLM_ID="${LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"

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
  echo "==> Training projector for ${ENCODER}"
  accelerate launch scripts/train_projector.py \
    --manifest "${MANIFEST}" \
    --feature-index "${FEATURE_ROOT}/${ENCODER}/index.jsonl" \
    --out-dir "${OUT_ROOT}/${ENCODER}" \
    --encoder-name "${ENCODER}" \
    --llm-id "${LLM_ID}" \
    --batch-size 1 \
    --grad-accum 16 \
    --epochs 1 \
    --lr 1e-3 \
    --dtype bf16
done
