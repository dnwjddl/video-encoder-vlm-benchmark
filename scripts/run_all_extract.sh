#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"
export VLMEB_LOCAL_FILES_ONLY="${VLMEB_LOCAL_FILES_ONLY:-0}"

MANIFEST="${1:-data/manifests/train_230k.jsonl}"
OUT_ROOT="${2:-features/train_230k}"

ENCODERS=(
  clip-vit-l-14-336
  siglip-so400m
  siglip2-so400m
  internvit-300m
  dinov2-vitl14
  videomaev2-base
  vjepa2-vith-256
  internvideo2-clip-s
)

for ENCODER in "${ENCODERS[@]}"; do
  echo "==> Extracting ${ENCODER}"
  echo "HF_HOME=${HF_HOME}"
  echo "VLMEB_LOCAL_FILES_ONLY=${VLMEB_LOCAL_FILES_ONLY}"
  python scripts/extract_features.py \
    --manifest "${MANIFEST}" \
    --encoder "${ENCODER}" \
    --out-dir "${OUT_ROOT}/${ENCODER}" \
    --skip-existing
done
