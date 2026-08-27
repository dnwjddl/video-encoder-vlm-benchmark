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

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "worker_gpu=${gpu}"
    echo "manifest=${MANIFEST}"
    echo "out_root=${OUT_ROOT}"
    echo "HF_HOME=${HF_HOME}"

    for idx in "${!ENCODERS[@]}"; do
      if [ $((idx % ${#GPUS[@]})) -ne "${gpu_idx}" ]; then
        continue
      fi

      encoder="${ENCODERS[$idx]}"
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

      python "${args[@]}"
    done
  ) > "${log_file}" 2>&1 &

  echo "$!"
}

pids=()
for gpu_idx in "${!GPUS[@]}"; do
  pid="$(run_worker "${gpu_idx}")"
  pids+=("${pid}")
  echo "Started no-train worker on GPU ${GPUS[$gpu_idx]}; pid=${pid}; log=${RUN_ROOT}/worker_gpu${GPUS[$gpu_idx]}.log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "One or more no-train diagnostic workers failed. Check logs under ${RUN_ROOT}." >&2
  exit 1
fi

python scripts/aggregate_diagnostics.py \
  --diagnostics-root "${OUT_ROOT}" \
  --out "${OUT_ROOT}_table.csv"

echo "All no-train diagnostic jobs completed."
echo "Summary: ${OUT_ROOT}_table.csv"
echo "Logs:    ${RUN_ROOT}"
