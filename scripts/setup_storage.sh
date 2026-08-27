#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${1:-/mnt/disks/data/vlm_encoder_benchmark}"
HF_CACHE_ROOT="${HF_HOME:-/mnt/disks/data/hf_cache}"

mkdir -p "${STORAGE_ROOT}"/{data,features,outputs,checkpoints,runs,videos}
mkdir -p "${STORAGE_ROOT}/data"/{manifests,benchmarks}
mkdir -p "${HF_CACHE_ROOT}"

link_dir() {
  local name="$1"
  local target="${STORAGE_ROOT}/${name}"

  if [ -L "${name}" ]; then
    rm "${name}"
  fi

  if [ -e "${name}" ]; then
    echo "Keeping existing ${name}; not replacing it with a symlink."
    echo "Move it manually if you want everything under ${STORAGE_ROOT}."
    return
  fi

  ln -s "${target}" "${name}"
  echo "Linked ${name} -> ${target}"
}

link_dir data
link_dir features
link_dir outputs
link_dir checkpoints
link_dir runs

cat > .env.storage <<EOF
export VLMEB_STORAGE_ROOT="${STORAGE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT}"
EOF

echo
echo "Storage root: ${STORAGE_ROOT}"
echo "HF cache:     ${HF_CACHE_ROOT}"
echo "Wrote .env.storage. Load it with: source .env.storage"
