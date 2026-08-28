#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"
export VLMEB_LOCAL_FILES_ONLY="${VLMEB_LOCAL_FILES_ONLY:-0}"

MANIFEST="${1:-data/manifests/train_230k.jsonl}"
FEATURE_ROOT="${2:-features/train_230k}"
OUT_ROOT="${3:-checkpoints/projectors}"
LLM_ID="${LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"

if [ "${VLMEB_LOCAL_FILES_ONLY}" = "1" ]; then
  if ! python scripts/check_hf_model_cache.py --model-id "${LLM_ID}" --trust-remote-code; then
    echo
    echo "The frozen LLM is not fully available in HF_HOME=${HF_HOME}."
    echo "Download it once with:"
    echo "  HF_HOME=${HF_HOME} VLMEB_LOCAL_FILES_ONLY=0 python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(\"${LLM_ID}\", repo_type=\"model\"))'"
    exit 1
  fi
fi

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
  echo "==> Training projector for ${ENCODER}"
  echo "HF_HOME=${HF_HOME}"
  echo "VLMEB_LOCAL_FILES_ONLY=${VLMEB_LOCAL_FILES_ONLY}"
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
