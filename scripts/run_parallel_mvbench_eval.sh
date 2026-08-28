#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"
export VLMEB_LOCAL_FILES_ONLY="${VLMEB_LOCAL_FILES_ONLY:-0}"

MANIFEST="${1:-data/benchmarks/mvbench_all.jsonl}"
FEATURE_ROOT="${2:-features/mvbench}"
CKPT_ROOT="${3:-checkpoints/projectors_20k}"
OUT_ROOT="${4:-outputs/mvbench}"
RUN_ROOT="${5:-runs/mvbench_eval}"

LLM_ID="${LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"
ENCODERS_CSV="${ENCODERS:-clip-vit-l-14-336,siglip-so400m,siglip2-so400m,dinov2-vitl14,internvit-300m,videomaev2-base,vjepa2-vith-256,internvideo2-clip-s}"
GPUS_CSV="${GPUS:-0,1,2,3}"
MODES_CSV="${MODES:-original,single,reverse,shuffle}"
DTYPE="${DTYPE:-bf16}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
ALLOW_MISSING_MEDIA="${ALLOW_MISSING_MEDIA:-1}"
FORCE="${FORCE:-0}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-80}"

IFS=',' read -r -a ENCODERS <<< "${ENCODERS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"

if [ ! -s "${MANIFEST}" ]; then
  echo "Missing or empty MVBench manifest: ${MANIFEST}" >&2
  exit 1
fi

MANIFEST_ROWS="$(wc -l < "${MANIFEST}" | tr -d ' ')"

mkdir -p "${FEATURE_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

TEXT_DIR="${OUT_ROOT}/text_only"
EVAL_ROOT="${OUT_ROOT}/projector_eval"
ANALYSIS_DIR="${OUT_ROOT}/analysis"
QUEUE_FILE="${RUN_ROOT}/encoder_queue_$$.txt"
QUEUE_LOCK="${RUN_ROOT}/encoder_queue_$$.lock"
current_log_files=()
current_status_files=()

printf "%s\n" "${ENCODERS[@]}" > "${QUEUE_FILE}"

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

prediction_complete() {
  local path="$1"
  if [ ! -s "${path}" ]; then
    return 1
  fi
  local rows
  rows="$(wc -l < "${path}" | tr -d ' ')"
  [ "${rows}" = "${MANIFEST_ROWS}" ]
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

latest_step_dir() {
  local encoder="$1"
  if [ ! -d "${CKPT_ROOT}/${encoder}" ]; then
    return 0
  fi
  find "${CKPT_ROOT}/${encoder}" -maxdepth 1 -type d -name 'step_*' | sort | tail -n 1
}

run_text_only() {
  mkdir -p "${TEXT_DIR}"
  if [ "${FORCE}" != "1" ] && prediction_complete "${TEXT_DIR}/predictions.jsonl"; then
    echo "SKIP text-only; found ${TEXT_DIR}/predictions.jsonl"
    return
  fi
  echo "==> Running text-only MVBench filter on GPU ${GPUS[0]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python scripts/evaluate_text_only_mcq.py \
    --manifest "${MANIFEST}" \
    --out-jsonl "${TEXT_DIR}/predictions.jsonl" \
    --out-csv "${TEXT_DIR}/summary.csv" \
    --llm-id "${LLM_ID}" \
    --dtype "${DTYPE}" \
    --max-length "${MAX_LENGTH}" \
    --require-media
}

run_encoder_mode() {
  local encoder="$1"
  local mode="$2"
  local gpu="$3"
  local status_file="$4"
  local step_dir
  step_dir="$(latest_step_dir "${encoder}")"
  if [ -z "${step_dir}" ]; then
    echo "FAIL missing_checkpoint ${encoder}" >> "${status_file}"
    echo "Missing checkpoint for ${encoder} under ${CKPT_ROOT}/${encoder}" >&2
    return 1
  fi

  local feature_dir="${FEATURE_ROOT}/${encoder}/${mode}"
  local pred_dir="${EVAL_ROOT}/${encoder}/${mode}"
  mkdir -p "${feature_dir}" "${pred_dir}"

  if [ "${FORCE}" != "1" ] && prediction_complete "${pred_dir}/predictions.jsonl"; then
    echo "SKIP ${encoder} ${mode}" >> "${status_file}"
    return 0
  fi

  echo "==> Extracting ${encoder} mode=${mode} on GPU ${gpu}"
  local -a extract_args=(
    scripts/extract_features.py
    --manifest "${MANIFEST}"
    --encoder "${encoder}"
    --out-dir "${feature_dir}"
    --max-tokens "${MAX_TOKENS}"
    --dtype "${DTYPE}"
    --skip-existing
    --frame-mode "${mode}"
  )
  if [ "${ALLOW_MISSING_MEDIA}" = "1" ]; then
    extract_args+=(--allow-missing-media)
  fi
  python "${extract_args[@]}"

  echo "==> Evaluating ${encoder} mode=${mode} on GPU ${gpu}"
  python scripts/evaluate_mcq.py \
    --bench-manifest "${MANIFEST}" \
    --feature-index "${feature_dir}/index.jsonl" \
    --projector-ckpt "${step_dir}/projector.pt" \
    --projector-metadata "${step_dir}/metadata.json" \
    --out-jsonl "${pred_dir}/predictions.jsonl" \
    --out-csv "${pred_dir}/summary.csv" \
    --llm-id "${LLM_ID}" \
    --dtype "${DTYPE}" \
    --max-length "${MAX_LENGTH}"
  echo "OK ${encoder} ${mode}" >> "${status_file}"
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
    echo "manifest_rows=${MANIFEST_ROWS}"
    echo "feature_root=${FEATURE_ROOT}"
    echo "ckpt_root=${CKPT_ROOT}"
    echo "out_root=${OUT_ROOT}"
    echo "HF_HOME=${HF_HOME}"
    echo "VLMEB_LOCAL_FILES_ONLY=${VLMEB_LOCAL_FILES_ONLY}"
    echo "ALLOW_MISSING_MEDIA=${ALLOW_MISSING_MEDIA}"

    while encoder="$(next_encoder)"; do
      echo "START ${encoder}" >> "${status_file}"
      for mode in "${MODES[@]}"; do
        run_encoder_mode "${encoder}" "${mode}" "${gpu}" "${status_file}" || exit 1
      done
      echo "DONE ${encoder}" >> "${status_file}"
    done
  ) > "${log_file}" 2>&1 &

  WORKER_PID="$!"
}

run_text_only

pids=()
for gpu_idx in "${!GPUS[@]}"; do
  run_worker "${gpu_idx}"
  pids+=("${WORKER_PID}")
  echo "Started MVBench eval worker on GPU ${GPUS[$gpu_idx]}; pid=${WORKER_PID}; log=${RUN_ROOT}/worker_gpu${GPUS[$gpu_idx]}.log"
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
  echo "One or more MVBench eval workers failed. Check logs under ${RUN_ROOT}." >&2
  exit 1
fi

echo "Worker status:"
cat "${current_status_files[@]}" 2>/dev/null || true

python scripts/aggregate_mvbench_filters.py \
  --manifest "${MANIFEST}" \
  --text-predictions "${TEXT_DIR}/predictions.jsonl" \
  --eval-root "${EVAL_ROOT}" \
  --out-dir "${ANALYSIS_DIR}" \
  --encoders "${ENCODERS_CSV}"

echo "All MVBench evaluation jobs completed."
echo "Analysis: ${ANALYSIS_DIR}"
echo "Logs:     ${RUN_ROOT}"
