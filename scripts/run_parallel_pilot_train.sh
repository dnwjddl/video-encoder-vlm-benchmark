#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"

MANIFEST="${1:-data/benchmarks/mcq_all.jsonl}"
FEATURE_ROOT="${2:-features/pilot_train}"
OUT_ROOT="${3:-checkpoints/pilot_projectors}"
RUN_ROOT="${4:-runs/pilot_train}"

LLM_ID="${LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"
ENCODERS_CSV="${ENCODERS:-clip-vit-l-14-336,siglip2-so400m,vjepa2-vith-256}"
GPUS_CSV="${GPUS:-1,2,3}"
DTYPE="${DTYPE:-bf16}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"

IFS=',' read -r -a ENCODERS <<< "${ENCODERS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if [ ! -s "${MANIFEST}" ]; then
  echo "Missing or empty manifest: ${MANIFEST}" >&2
  exit 1
fi

if [ "${#ENCODERS[@]}" -gt "${#GPUS[@]}" ]; then
  echo "Need at least as many GPUS as ENCODERS." >&2
  echo "ENCODERS=${ENCODERS_CSV}" >&2
  echo "GPUS=${GPUS_CSV}" >&2
  exit 1
fi

mkdir -p "${FEATURE_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

run_one() {
  local encoder="$1"
  local gpu="$2"
  local log_file="${RUN_ROOT}/${encoder}.gpu${gpu}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "encoder=${encoder}"
    echo "gpu=${gpu}"
    echo "manifest=${MANIFEST}"
    echo "llm_id=${LLM_ID}"
    echo "feature_root=${FEATURE_ROOT}"
    echo "out_root=${OUT_ROOT}"
    echo "HF_HOME=${HF_HOME}"

    python scripts/extract_features.py \
      --manifest "${MANIFEST}" \
      --encoder "${encoder}" \
      --out-dir "${FEATURE_ROOT}/${encoder}" \
      --max-tokens "${MAX_TOKENS}" \
      --dtype "${DTYPE}" \
      --skip-existing

    python scripts/train_projector.py \
      --manifest "${MANIFEST}" \
      --feature-index "${FEATURE_ROOT}/${encoder}/index.jsonl" \
      --out-dir "${OUT_ROOT}/${encoder}" \
      --encoder-name "${encoder}" \
      --llm-id "${LLM_ID}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --dtype "${DTYPE}" \
      --max-length "${MAX_LENGTH}" \
      --num-workers "${NUM_WORKERS}" \
      --save-every-steps "${SAVE_EVERY_STEPS}"
  ) > "${log_file}" 2>&1 &

  echo "$!"
}

pids=()
for idx in "${!ENCODERS[@]}"; do
  encoder="${ENCODERS[$idx]}"
  gpu="${GPUS[$idx]}"
  pid="$(run_one "${encoder}" "${gpu}")"
  pids+=("${pid}")
  echo "Started ${encoder} on GPU ${gpu}; pid=${pid}; log=${RUN_ROOT}/${encoder}.gpu${gpu}.log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "One or more pilot training jobs failed. Check logs under ${RUN_ROOT}." >&2
  exit 1
fi

echo "All pilot training jobs completed."
echo "Checkpoints: ${OUT_ROOT}"
echo "Logs:        ${RUN_ROOT}"
