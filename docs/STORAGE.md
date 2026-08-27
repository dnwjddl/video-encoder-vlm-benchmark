# Storage layout

Recommended server layout:

```text
/mnt/disks/data/
  hf_cache/
  vlm_encoder_benchmark/
    data/
      manifests/
      benchmarks/
    videos/
      activitynet/
      activitynet_1k/
      activitynet_5k/
    features/
    outputs/
    checkpoints/
    runs/
```

Inside the git repository, run:

```bash
bash scripts/setup_storage.sh /mnt/disks/data/vlm_encoder_benchmark
source .env.storage
```

This creates local symlinks:

```text
data        -> /mnt/disks/data/vlm_encoder_benchmark/data
features    -> /mnt/disks/data/vlm_encoder_benchmark/features
outputs     -> /mnt/disks/data/vlm_encoder_benchmark/outputs
checkpoints -> /mnt/disks/data/vlm_encoder_benchmark/checkpoints
runs        -> /mnt/disks/data/vlm_encoder_benchmark/runs
```

The model cache uses:

```bash
export HF_HOME=/mnt/disks/data/hf_cache
```

## If you already created files elsewhere

Current earlier commands may have created files in:

```text
/data/hf_cache
/data/videos
~/video-encoder-vlm-benchmark/data
~/video-encoder-vlm-benchmark/outputs
~/video-encoder-vlm-benchmark/features
~/video-encoder-vlm-benchmark/checkpoints
```

Move them before running `setup_storage.sh` if you want everything in one place:

```bash
mkdir -p /mnt/disks/data/vlm_encoder_benchmark

rsync -a ~/video-encoder-vlm-benchmark/data/ /mnt/disks/data/vlm_encoder_benchmark/data/ 2>/dev/null || true
rsync -a ~/video-encoder-vlm-benchmark/features/ /mnt/disks/data/vlm_encoder_benchmark/features/ 2>/dev/null || true
rsync -a ~/video-encoder-vlm-benchmark/outputs/ /mnt/disks/data/vlm_encoder_benchmark/outputs/ 2>/dev/null || true
rsync -a ~/video-encoder-vlm-benchmark/checkpoints/ /mnt/disks/data/vlm_encoder_benchmark/checkpoints/ 2>/dev/null || true
rsync -a /data/videos/ /mnt/disks/data/vlm_encoder_benchmark/videos/ 2>/dev/null || true
rsync -a /data/hf_cache/ /mnt/disks/data/hf_cache/ 2>/dev/null || true
```

Then, if you have confirmed the copies are good, you can remove the old copies manually.
