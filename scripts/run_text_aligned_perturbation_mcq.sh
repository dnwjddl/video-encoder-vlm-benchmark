#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"

MANIFEST="${1:-data/benchmarks/mcq_all.jsonl}"
OUT_ROOT="${2:-outputs/zeroshot_perturbation_mcq}"

ENCODERS=(
  clip-vit-l-14-336
  siglip-so400m
  siglip2-so400m
)

for ENCODER in "${ENCODERS[@]}"; do
  echo "==> Zero-shot perturbation MCQ for ${ENCODER}"
  python scripts/evaluate_zeroshot_perturbation_mcq.py \
    --manifest "${MANIFEST}" \
    --encoder "${ENCODER}" \
    --out-jsonl "${OUT_ROOT}/${ENCODER}/predictions.jsonl" \
    --out-csv "${OUT_ROOT}/${ENCODER}/summary.csv"
done
