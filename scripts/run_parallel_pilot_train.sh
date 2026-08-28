#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"
export VLMEB_LOCAL_FILES_ONLY="${VLMEB_LOCAL_FILES_ONLY:-0}"

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
ALLOW_MISSING_MEDIA="${ALLOW_MISSING_MEDIA:-1}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-80}"
SCHEDULER="${SCHEDULER:-dynamic}"

IFS=',' read -r -a ENCODERS <<< "${ENCODERS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if [ ! -s "${MANIFEST}" ]; then
  echo "Missing or empty manifest: ${MANIFEST}" >&2
  exit 1
fi

if [ "${VLMEB_LOCAL_FILES_ONLY}" = "1" ]; then
  if ! python scripts/check_hf_model_cache.py --model-id "${LLM_ID}" --trust-remote-code; then
    echo
    echo "The frozen LLM is not fully available in HF_HOME=${HF_HOME}."
    echo "Download it once with:"
    echo "  HF_HOME=${HF_HOME} VLMEB_LOCAL_FILES_ONLY=0 python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(\"${LLM_ID}\", repo_type=\"model\"))'"
    exit 1
  fi
fi

mkdir -p "${FEATURE_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

QUEUE_FILE="${RUN_ROOT}/encoder_queue_$$.txt"
QUEUE_LOCK="${RUN_ROOT}/encoder_queue_$$.lock"
current_log_files=()
current_status_files=()

if [ "${SCHEDULER}" = "dynamic" ]; then
  printf "%s\n" "${ENCODERS[@]}" > "${QUEUE_FILE}"
fi

cleanup_queue() {
  rm -f "${QUEUE_FILE}" "${QUEUE_FILE}.tmp"
  rmdir "${QUEUE_LOCK}" 2>/dev/null || true
}

trap cleanup_queue EXIT

print_failure_logs() {
  echo
  echo "Recent worker log output:"
  for log_file in "${current_log_files[@]}"; do
    if [ ! -f "${log_file}" ]; then
      continue
    fi
    echo
    echo "==> ${log_file}"
    tail -n "${LOG_TAIL_LINES}" "${log_file}" || true
  done
}

next_encoder() {
  local encoder=""
  local tmp_file="${QUEUE_FILE}.tmp"

  while ! mkdir "${QUEUE_LOCK}" 2>/dev/null; do
    sleep 0.2
  done

  if [ -s "${QUEUE_FILE}" ]; then
    encoder="$(head -n 1 "${QUEUE_FILE}")"
    tail -n +2 "${QUEUE_FILE}" > "${tmp_file}" || true
    mv "${tmp_file}" "${QUEUE_FILE}"
  fi

  rmdir "${QUEUE_LOCK}"

  if [ -z "${encoder}" ]; then
    return 1
  fi
  printf "%s\n" "${encoder}"
}

run_encoder_job() {
  local encoder="$1"
  local gpu="$2"
  local status_file="$3"

  echo "START ${encoder}" >> "${status_file}"
  echo "==> Extracting ${encoder} on GPU ${gpu}"
  extract_args=(
    scripts/extract_features.py
    --manifest "${MANIFEST}"
    --encoder "${encoder}"
    --out-dir "${FEATURE_ROOT}/${encoder}"
    --max-tokens "${MAX_TOKENS}"
    --dtype "${DTYPE}"
    --skip-existing
  )
  if [ "${ALLOW_MISSING_MEDIA}" = "1" ]; then
    extract_args+=(--allow-missing-media)
  fi

  if ! python "${extract_args[@]}"; then
    echo "FAIL extract ${encoder}" >> "${status_file}"
    return 1
  fi

  echo "==> Training projector for ${encoder} on GPU ${gpu}"
  if ! python scripts/train_projector.py \
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
    --save-every-steps "${SAVE_EVERY_STEPS}"; then
    echo "FAIL train ${encoder}" >> "${status_file}"
    return 1
  fi
  echo "OK ${encoder}" >> "${status_file}"
}

run_worker() {
  local gpu_idx="$1"
  local gpu="${GPUS[$gpu_idx]}"
  local log_file="${RUN_ROOT}/worker_gpu${gpu}.log"
  local status_file="${RUN_ROOT}/worker_gpu${gpu}.status"

  current_log_files+=("${log_file}")
  current_status_files+=("${status_file}")

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    : > "${status_file}"
    echo "worker_gpu=${gpu}"
    echo "manifest=${MANIFEST}"
    echo "llm_id=${LLM_ID}"
    echo "feature_root=${FEATURE_ROOT}"
    echo "out_root=${OUT_ROOT}"
    echo "HF_HOME=${HF_HOME}"
    echo "VLMEB_LOCAL_FILES_ONLY=${VLMEB_LOCAL_FILES_ONLY}"
    echo "ALLOW_MISSING_MEDIA=${ALLOW_MISSING_MEDIA}"
    echo "SCHEDULER=${SCHEDULER}"

    if [ "${SCHEDULER}" = "dynamic" ]; then
      while encoder="$(next_encoder)"; do
        run_encoder_job "${encoder}" "${gpu}" "${status_file}" || exit 1
      done
    else
      for idx in "${!ENCODERS[@]}"; do
        if [ $((idx % ${#GPUS[@]})) -ne "${gpu_idx}" ]; then
          continue
        fi
        run_encoder_job "${ENCODERS[$idx]}" "${gpu}" "${status_file}" || exit 1
      done
    fi
  ) > "${log_file}" 2>&1 &

  WORKER_PID="$!"
}

pids=()
for gpu_idx in "${!GPUS[@]}"; do
  run_worker "${gpu_idx}"
  pids+=("${WORKER_PID}")
  echo "Started pilot training worker on GPU ${GPUS[$gpu_idx]}; pid=${WORKER_PID}; log=${RUN_ROOT}/worker_gpu${GPUS[$gpu_idx]}.log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "Worker status:"
  cat "${current_status_files[@]}" 2>/dev/null || true
  print_failure_logs
  echo "One or more pilot training jobs failed. Check logs under ${RUN_ROOT}." >&2
  exit 1
fi

echo "Worker status:"
cat "${current_status_files[@]}" 2>/dev/null || true

echo "All pilot training jobs completed."
echo "Checkpoints: ${OUT_ROOT}"
echo "Logs:        ${RUN_ROOT}"
