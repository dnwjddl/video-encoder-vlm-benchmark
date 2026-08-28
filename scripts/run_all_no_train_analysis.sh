#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"
export VLMEB_LOCAL_FILES_ONLY="${VLMEB_LOCAL_FILES_ONLY:-0}"

MANIFEST="${1:-data/manifests/train_debug.jsonl}"
OUT_ROOT="${2:-outputs/no_train_diagnostics}"
ENCODERS_CSV="${ENCODERS:-clip-vit-l-14-336,siglip-so400m,siglip2-so400m,internvit-300m,dinov2-vitl14,videomaev2-base,vjepa2-vith-256,internvideo2-clip-s}"
DTYPE="${DTYPE:-bf16}"
LIMIT="${LIMIT:-}"
STRICT_MEDIA="${STRICT_MEDIA:-0}"
FORCE="${FORCE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

IFS=',' read -r -a ENCODERS <<< "${ENCODERS_CSV}"

failed=0

for ENCODER in "${ENCODERS[@]}"; do
  if [ "${FORCE}" != "1" ] \
    && [ -s "${OUT_ROOT}/${ENCODER}/per_example.jsonl" ] \
    && [ -s "${OUT_ROOT}/${ENCODER}/summary.csv" ]; then
    echo "==> Skipping ${ENCODER}; existing outputs found. Set FORCE=1 to rerun."
    continue
  fi

  echo "==> No-train diagnostics for ${ENCODER}"
  echo "HF_HOME=${HF_HOME}"
  echo "VLMEB_LOCAL_FILES_ONLY=${VLMEB_LOCAL_FILES_ONLY}"
  args=(
    scripts/analyze_encoder_no_train.py
    --manifest "${MANIFEST}"
    --encoder "${ENCODER}"
    --out-jsonl "${OUT_ROOT}/${ENCODER}/per_example.jsonl"
    --out-csv "${OUT_ROOT}/${ENCODER}/summary.csv"
    --dtype "${DTYPE}"
  )
  if [ -n "${LIMIT}" ]; then
    args+=(--limit "${LIMIT}")
  fi
  if [ "${STRICT_MEDIA}" = "1" ]; then
    args+=(--strict-media)
  fi

  if ! python "${args[@]}"; then
    failed=1
    echo "Warning: diagnostics failed for ${ENCODER}."
    if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
      exit 1
    fi
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "One or more encoders failed. Completed encoders can still be aggregated."
  echo "Install missing dependencies or rerun with a smaller ENCODERS list."
fi
