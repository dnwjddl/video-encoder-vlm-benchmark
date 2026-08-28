#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/disks/data/hf_cache}"

MANIFEST="${1:-data/manifests/hf_video_debug.jsonl}"
OUT_ROOT="${2:-outputs/no_train_diagnostics}"
RUN_ROOT="${3:-runs/no_train_diagnostics}"

ENCODERS_CSV="${ENCODERS:-clip-vit-l-14-336,siglip-so400m,siglip2-so400m,internvit-300m,dinov2-vitl14,videomaev2-base,vjepa2-vith-256,internvideo2-clip-1b}"
GPUS_CSV="${GPUS:-0,1,2,3}"
DTYPE="${DTYPE:-bf16}"
LIMIT="${LIMIT:-}"
STRICT_MEDIA="${STRICT_MEDIA:-0}"
FORCE="${FORCE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

IFS=',' read -r -a ENCODERS <<< "${ENCODERS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if [ ! -s "${MANIFEST}" ]; then
  echo "Missing or empty manifest: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${RUN_ROOT}"

run_worker() {
  local gpu_idx="$1"
  local gpu="${GPUS[$gpu_idx]}"
  local log_file="${RUN_ROOT}/worker_gpu${gpu}.log"
  local status_file="${RUN_ROOT}/worker_gpu${gpu}.status"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    : > "${status_file}"
    echo "worker_gpu=${gpu}"
    echo "manifest=${MANIFEST}"
    echo "out_root=${OUT_ROOT}"
    echo "HF_HOME=${HF_HOME}"
    worker_failed=0

    for idx in "${!ENCODERS[@]}"; do
      if [ $((idx % ${#GPUS[@]})) -ne "${gpu_idx}" ]; then
        continue
      fi

      encoder="${ENCODERS[$idx]}"
      if [ "${FORCE}" != "1" ] \
        && [ -s "${OUT_ROOT}/${encoder}/per_example.jsonl" ] \
        && [ -s "${OUT_ROOT}/${encoder}/summary.csv" ]; then
        echo "==> Skipping ${encoder}; existing outputs found. Set FORCE=1 to rerun."
        echo "SKIP ${encoder}" >> "${status_file}"
        continue
      fi

      echo "==> No-train diagnostics for ${encoder} on GPU ${gpu}"

      args=(
        scripts/analyze_encoder_no_train.py
        --manifest "${MANIFEST}"
        --encoder "${encoder}"
        --out-jsonl "${OUT_ROOT}/${encoder}/per_example.jsonl"
        --out-csv "${OUT_ROOT}/${encoder}/summary.csv"
        --dtype "${DTYPE}"
      )
      if [ -n "${LIMIT}" ]; then
        args+=(--limit "${LIMIT}")
      fi
      if [ "${STRICT_MEDIA}" = "1" ]; then
        args+=(--strict-media)
      fi

      if ! python "${args[@]}"; then
        echo "Warning: diagnostics failed for ${encoder} on GPU ${gpu}."
        echo "FAIL ${encoder}" >> "${status_file}"
        worker_failed=1
        if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
          exit 1
        fi
      else
        echo "OK ${encoder}" >> "${status_file}"
      fi
    done

    if [ "${worker_failed}" -ne 0 ]; then
      exit 1
    fi
  ) > "${log_file}" 2>&1 &

  WORKER_PID="$!"
}

pids=()
for gpu_idx in "${!GPUS[@]}"; do
  run_worker "${gpu_idx}"
  pids+=("${WORKER_PID}")
  echo "Started no-train worker on GPU ${GPUS[$gpu_idx]}; pid=${WORKER_PID}; log=${RUN_ROOT}/worker_gpu${GPUS[$gpu_idx]}.log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "Worker status:"
  cat "${RUN_ROOT}"/worker_gpu*.status 2>/dev/null || true
  echo "One or more no-train diagnostic workers failed. Check logs under ${RUN_ROOT}." >&2
  exit 1
fi

echo "Worker status:"
cat "${RUN_ROOT}"/worker_gpu*.status 2>/dev/null || true

python scripts/aggregate_diagnostics.py \
  --diagnostics-root "${OUT_ROOT}" \
  --out "${OUT_ROOT}_table.csv"

echo "All no-train diagnostic jobs completed."
echo "Summary: ${OUT_ROOT}_table.csv"
echo "Logs:    ${RUN_ROOT}"
