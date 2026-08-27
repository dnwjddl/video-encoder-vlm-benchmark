#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"

BENCH_MANIFEST="${1:-data/benchmarks/mcq_all.jsonl}"
FEATURE_ROOT="${2:-features/benchmarks}"
CKPT_ROOT="${3:-checkpoints/projectors}"
OUT_ROOT="${4:-outputs/eval}"

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
  STEP_DIR="$(find "${CKPT_ROOT}/${ENCODER}" -maxdepth 1 -type d -name 'step_*' | sort | tail -n 1)"
  echo "==> Evaluating ${ENCODER} with ${STEP_DIR}"
  python scripts/evaluate_mcq.py \
    --bench-manifest "${BENCH_MANIFEST}" \
    --feature-index "${FEATURE_ROOT}/${ENCODER}/index.jsonl" \
    --projector-ckpt "${STEP_DIR}/projector.pt" \
    --projector-metadata "${STEP_DIR}/metadata.json" \
    --out-jsonl "${OUT_ROOT}/${ENCODER}/predictions.jsonl" \
    --out-csv "${OUT_ROOT}/${ENCODER}/summary.csv"
done
