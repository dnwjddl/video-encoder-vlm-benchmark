# Video Encoder VLM Benchmark

Controlled benchmark harness for comparing frozen visual encoders under the same LLM and projector-only alignment setup.

The main research question is:

> If we freeze the visual encoder and freeze `Qwen/Qwen2.5-7B-Instruct`, which image/video encoder transfers best to temporal video understanding benchmarks under the same alignment budget?

This repo intentionally separates three things:

1. **Encoder feature extraction**
2. **No-training encoder diagnostics**
3. **Projector-only training**
4. **Benchmark evaluation**

That separation keeps the comparison cleaner. The LLM, projector architecture, training data, prompts, and decoding/scoring code stay fixed; only the visual encoder changes.

## Encoders included

Image encoder track:

- `clip-vit-l-14-336`: `openai/clip-vit-large-patch14-336`
- `siglip-so400m`: `google/siglip-so400m-patch14-384`
- `siglip2-so400m`: `google/siglip2-so400m-patch14-384`
- `internvit-300m`: `OpenGVLab/InternViT-300M-448px`
- `dinov2-vitl14`: `facebook/dinov2-large`

Video encoder track:

- `videomaev2-base`: `OpenGVLab/VideoMAEv2-Base`
- `vjepa2-vith-256`: `facebook/vjepa2-vith-fpc64-256`
- `internvideo2-clip-s`: `OpenGVLab/InternVideo2_CLIP_S`
- `internvideo2-clip-1b`: `OpenGVLab/InternVideo2-CLIP-1B-224p-f8` external/manual path

Note: `internvideo2-clip-s` is the AutoModel-compatible InternVideo2-CLIP
checkpoint used by this harness. `internvideo2-clip-1b` is a gated add-on
checkpoint containing `1B_clip.pth` but no Transformers `config.json`, so it
requires the official OpenGVLab InternVideo code path and additional checkpoints.
`internvit-300m` and `internvideo2-clip-s` are configured with
`disable_flash_attn: true` so they can run on systems where `flash-attn` is not
installed or the prebuilt wheel fails with a `GLIBC_2.32 not found` error. They
use the same weights with non-flash attention paths.

## Recommended experiment size

Start with about **230K training examples**:

- 100K caption/alignment examples
- 100K open-ended QA/instruction examples
- 30K multiple-choice QA examples

For a quick smoke test, use 1K-5K examples first. The full 230K run is for the main comparison.

## 1. Clone and install on the server

```bash
git clone https://github.com/<YOUR_USER>/video-encoder-vlm-benchmark.git
cd video-encoder-vlm-benchmark

conda create -n vlmenc python=3.10 -y
conda activate vlmenc

pip install -e .
pip install -r requirements.txt
```

If you use gated Hugging Face models or datasets:

```bash
huggingface-cli login
```

If gated models were already downloaded into the configured cache, you can force
all model loaders to use only local files:

```bash
export VLMEB_LOCAL_FILES_ONLY=1
```

This is useful for gated models: without local-only mode,
Transformers may still make a Hugging Face metadata request and fail with a 401
if the current shell is not authenticated.

For `internvit-300m` and `internvideo2-clip-s`, this harness disables
flash/fused attention in the model config and provides a small import stub for
the `flash_attn` package. That means the recommended path is to avoid installing
`flash-attn` unless you specifically want to benchmark the vendor implementation.

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
make download-internvideo2-clip-s
VLMEB_LOCAL_FILES_ONLY=1 make check-flash-attn
```

To check the encoders that most often hit remote-code runtime issues:

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
export VLMEB_LOCAL_FILES_ONLY=1
make check-problem-encoders GPU=0
make smoke-encoder ENCODER=videomaev2-base GPU=0
make smoke-encoder ENCODER=internvit-300m GPU=0
make smoke-encoder ENCODER=internvideo2-clip-s GPU=0
```

## Storage setup

Put large files under `/mnt/disks/data`:

```bash
bash scripts/setup_storage.sh /mnt/disks/data/vlm_encoder_benchmark
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
```

This stores repo artifacts under:

```text
/mnt/disks/data/vlm_encoder_benchmark/
  data/
  videos/
  features/
  outputs/
  checkpoints/
  runs/
```

and Hugging Face models under:

```text
/mnt/disks/data/hf_cache
```

More details are in [docs/STORAGE.md](docs/STORAGE.md).

## 2. Prepare training manifest

The script below creates a unified JSONL manifest from public annotation datasets.

```bash
python scripts/download_data.py \
  --out data/manifests/train_230k.jsonl \
  --streaming \
  --caption-count 100000 \
  --qa-count 100000 \
  --mcq-count 30000 \
  --video-root /mnt/disks/data/vlm_encoder_benchmark/videos
```

Important: this downloads/streams **annotations**. Many video datasets do not redistribute all raw videos in one simple archive. Put videos on the server, then set `--video-root` so each row's `media_path` points to the correct local file.

If you already have local annotation JSON/JSONL files:

```bash
python scripts/download_data.py \
  --out data/manifests/train_230k.jsonl \
  --local-json /path/to/local_annotations.jsonl \
  --video-root /mnt/disks/data/vlm_encoder_benchmark/videos \
  --caption-count 100000 \
  --qa-count 100000 \
  --mcq-count 30000
```

The unified schema is documented in [docs/MANIFEST_SCHEMA.md](docs/MANIFEST_SCHEMA.md).

### 5K video training manifest

For the next step after the 358-row smoke test, build a 5K caption-alignment
manifest from `VLM2Vec/MSR-VTT`. This downloads real MP4 files from the dataset
repo and writes everything under `/mnt/disks/data` through the `data` symlink.

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
make train-manifest-5k
wc -l data/manifests/train_5k_msrvtt.jsonl
```

Then train the same projector setup on this 5K manifest:

```bash
VLMEB_LOCAL_FILES_ONLY=1 \
GPUS=1,2,3 \
ENCODERS=siglip2-so400m,dinov2-vitl14,vjepa2-vith-256 \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
GRAD_ACCUM=8 \
make train-5k
```

### 20K analysis training manifest

For a cleaner encoder comparison, prefer more data over many repeated epochs.
The 20K target uses the Hugging Face dataset `VLM2Vec/MSR-VTT`, config
`train_9k`. MSR-VTT has roughly 10K video clips and 200K captions; this target
samples 20K caption-alignment examples from the 9K-video training split,
expanding up to 3 captions per video. It writes a separate
manifest/checkpoint/run tree so the 5K smoke run does not mix with the analysis
run.

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
make train-manifest-20k
wc -l data/manifests/train_20k_msrvtt.jsonl
```

Recommended analysis run:

```bash
VLMEB_LOCAL_FILES_ONLY=1 \
EPOCHS=2 \
GPUS=0,1,2,3 \
ENCODERS=clip-vit-l-14-336,siglip-so400m,siglip2-so400m,dinov2-vitl14 \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
GRAD_ACCUM=8 \
ALLOW_MISSING_MEDIA=1 \
make train-20k
```

Then run the remaining encoders with the same settings:

```bash
VLMEB_LOCAL_FILES_ONLY=1 \
EPOCHS=2 \
GPUS=0,1,2,3 \
ENCODERS=internvit-300m,videomaev2-base,vjepa2-vith-256,internvideo2-clip-s \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
GRAD_ACCUM=8 \
ALLOW_MISSING_MEDIA=1 \
make train-20k
```

Or run all configured encoders as one dynamic GPU queue. This starts one worker
per GPU; when a worker finishes an encoder, it automatically grabs the next
remaining encoder.

```bash
VLMEB_LOCAL_FILES_ONLY=1 \
GPUS=0,1,2,3 \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
GRAD_ACCUM=8 \
ALLOW_MISSING_MEDIA=1 \
make train-20k-all
```

By default, `train-20k-all` uses `EPOCHS=2`. Override with `EPOCHS=1` or
`EPOCHS=3` only when you intentionally want a different analysis budget.

### If you need actual video files for no-training diagnostics

The instruction datasets above often provide annotations and relative video ids, not the raw MP4 files. If your manifest paths look like `/data/videos/...` but `os.path.exists(...)` returns `False`, use a small directly downloadable video subset first.

```bash
HF_HOME=/mnt/disks/data/hf_cache python scripts/download_hf_video_dataset.py \
  --dataset-id VLM2Vec/mvbench-FunQA_test \
  --split test \
  --source-mode video-column \
  --video-column video \
  --label-column label \
  --video-dir /mnt/disks/data/vlm_encoder_benchmark/videos/mvbench_funqa_debug \
  --out data/manifests/hf_video_debug.jsonl \
  --max-samples 358 \
  --validate
```

This exports actual videos from a Hugging Face `Video` column, not YouTube pages. `VLM2Vec/Kinetics-700` is metadata-only in this repo: its `video_path` rows point to Kinetics files, but those MP4 files are not hosted there.

Check that the files exist:

```bash
python -c 'import json,os; rows=[json.loads(l) for _,l in zip(range(5),open("data/manifests/hf_video_debug.jsonl"))]; [print(r["media_path"], os.path.exists(r["media_path"])) for r in rows]'
```

This manifest is enough for no-training encoder diagnostics because those metrics only need real videos, not QA labels.

ActivityNet can still be useful later, but its public URLs are YouTube-based and often trigger HTTP 429. If a run prints many `moov atom not found` errors, those files are not valid videos and should not be used for analysis.

If some MP4s are corrupt because a download was interrupted or rate-limited, create a clean manifest:

```bash
python scripts/filter_valid_media.py \
  --input data/manifests/activitynet_1k.jsonl \
  --out data/manifests/activitynet_1k.valid.jsonl
```

Then use `activitynet_1k.valid.jsonl` for diagnostics and MCQ manifest generation.

If you still want to try the ActivityNet helper, use:

```bash
HF_HOME=/mnt/disks/data/hf_cache python scripts/download_activitynet_subset.py \
  --video-dir /mnt/disks/data/vlm_encoder_benchmark/videos/activitynet_1k \
  --out data/manifests/activitynet_1k.jsonl \
  --max-samples 1000 \
  --max-duration 180 \
  --sleep-between-downloads 1 \
  --skip-existing
```

If YouTube/Google returns HTTP 429, stop and resume later with the same command. The downloader writes partial manifests as it goes, so successful downloads are kept.

For a larger no-training representation analysis, use a larger HF rawvideo dataset such as `VLM2Vec/nextqa-rawvideo`. Do not scale a broken YouTube scrape just to increase the count.

```bash
HF_HOME=/mnt/disks/data/hf_cache python scripts/download_hf_video_dataset.py \
  --dataset-id VLM2Vec/nextqa-rawvideo \
  --split train \
  --source-mode video-column \
  --video-column video \
  --label-column "" \
  --video-dir /mnt/disks/data/vlm_encoder_benchmark/videos/nextqa_rawvideo_1k \
  --out data/manifests/nextqa_rawvideo_1k.jsonl \
  --max-samples 1000 \
  --validate
```

That is still not a reasoning benchmark, but it is much better for stable representation statistics than a broken or rate-limited YouTube scrape.

## 3. Run no-training encoder diagnostics

Before projector training, you can inspect whether an encoder is sensitive to temporal structure at all.

Run one encoder:

```bash
python scripts/analyze_encoder_no_train.py \
  --manifest data/manifests/hf_video_debug.jsonl \
  --encoder vjepa2-vith-256 \
  --out-jsonl outputs/no_train_diagnostics/vjepa2-vith-256/per_example.jsonl \
  --out-csv outputs/no_train_diagnostics/vjepa2-vith-256/summary.csv
```

Run all encoders and aggregate:

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
VLMEB_LOCAL_FILES_ONLY=1 \
bash scripts/run_all_no_train_analysis.sh \
  data/manifests/hf_video_debug.jsonl \
  outputs/no_train_diagnostics

python scripts/aggregate_diagnostics.py \
  --diagnostics-root outputs/no_train_diagnostics \
  --out outputs/no_train_diagnostics_table.csv
```

The main metrics are:

- `order_distance`: original vs reversed-frame representation distance
- `shuffle_distance`: original vs shuffled-frame representation distance
- `cycle_shift_distance`: original vs temporally shifted-frame representation distance
- `half_swap_distance`: original vs swapped first/second half representation distance
- `stride_distance`: original vs sparse temporal sampling representation distance
- `segment_diversity`: how differently segments from the same video are represented
- `segment_temporal_margin`: whether far-apart segments are more different than neighboring segments
- `segment_distance_correlation`: whether representation distance grows with temporal gap
- `token_effective_rank`: how many independent directions remain after token compression
- `token_top1_energy_ratio`: whether one dominant direction collapses the representation
- `knn_top1` / `knn_topK`: label consistency without training, when labels exist

For image encoders, `order_distance` should be near zero because frames are encoded independently and averaged. That is not a bug; it is exactly the limitation this diagnostic exposes. More details are in [docs/NO_TRAIN_DIAGNOSTICS.md](docs/NO_TRAIN_DIAGNOSTICS.md).

## 4. Run no-training perturbation MCQ for text-aligned encoders

If you have a multiple-choice benchmark manifest with real videos, you can also compare original/reverse/shuffle correctness without projector training for text-aligned encoders.

If `data/benchmarks/mcq_all.jsonl` does not exist yet, create a smoke-test MCQ file from the downloaded HF video subset:

```bash
python scripts/make_label_mcq_manifest.py \
  --input data/manifests/hf_video_debug.jsonl \
  --out data/benchmarks/mcq_all.jsonl \
  --num-choices 3 \
  --benchmark-name hf_video_label_mcq
```

This generated file is useful for checking the perturbation pipeline and frozen label consistency. It is not a replacement for temporal reasoning benchmarks such as MVBench, TempCompass, TemporalBench, or VideoMME.

```bash
HF_HOME=/mnt/disks/data/hf_cache python scripts/evaluate_zeroshot_perturbation_mcq.py \
  --manifest data/benchmarks/mcq_all.jsonl \
  --encoder siglip2-so400m \
  --out-jsonl outputs/zeroshot_perturbation_mcq/siglip2-so400m/predictions.jsonl \
  --out-csv outputs/zeroshot_perturbation_mcq/siglip2-so400m/summary.csv
```

Run the default text-aligned image encoders:

```bash
HF_HOME=/mnt/disks/data/hf_cache bash scripts/run_text_aligned_perturbation_mcq.sh \
  data/benchmarks/mcq_all.jsonl \
  outputs/zeroshot_perturbation_mcq

python scripts/aggregate_perturbation_mcq.py \
  --root outputs/zeroshot_perturbation_mcq \
  --out outputs/zeroshot_perturbation_mcq_table.csv
```

The key count is:

```text
temporal_sensitive_any = original correct and reverse/shuffle wrong
```

The robust count is:

```text
robust_correct_all = original, reverse, and shuffle all correct
```

More details are in [docs/PERTURBATION_MCQ.md](docs/PERTURBATION_MCQ.md).

## 5. Extract features for each encoder

Run all encoders:

```bash
bash scripts/run_all_extract.sh \
  data/manifests/train_230k.jsonl \
  features/train_230k
```

Run one encoder:

```bash
python scripts/extract_features.py \
  --manifest data/manifests/train_230k.jsonl \
  --encoder siglip2-so400m \
  --out-dir features/train_230k/siglip2-so400m \
  --skip-existing
```

For a smoke test:

```bash
python scripts/extract_features.py \
  --manifest data/manifests/train_230k.jsonl \
  --encoder clip-vit-l-14-336 \
  --out-dir features/debug/clip-vit-l-14-336 \
  --limit 100 \
  --skip-existing
```

## 6. Train projector only

Default setup:

- visual encoder: frozen
- `Qwen/Qwen2.5-7B-Instruct`: frozen
- projector: trainable 2-layer MLP

Before training in local-only mode, cache the frozen LLM once:

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
make download-llm
VLMEB_LOCAL_FILES_ONLY=1 make check-llm
```

Run all encoders:

```bash
LLM_ID=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_all_train.sh \
  data/manifests/train_230k.jsonl \
  features/train_230k \
  checkpoints/projectors
```

Run one encoder:

```bash
accelerate launch scripts/train_projector.py \
  --manifest data/manifests/train_230k.jsonl \
  --feature-index features/train_230k/siglip2-so400m/index.jsonl \
  --out-dir checkpoints/projectors/siglip2-so400m \
  --encoder-name siglip2-so400m \
  --llm-id Qwen/Qwen2.5-7B-Instruct \
  --batch-size 1 \
  --grad-accum 16 \
  --epochs 1 \
  --lr 1e-3 \
  --dtype bf16
```

The checkpoint layout is:

```text
checkpoints/projectors/siglip2-so400m/
  metadata.json
  step_XXXXXX/
    projector.pt
    metadata.json
```

### Multi-GPU training-free diagnostics and pilot training

First confirm CUDA works:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Run all training-free diagnostics across GPUs 0, 1, 2, and 3:

```bash
GPUS=0,1,2,3 make diagnose-parallel
```

Completed encoder outputs are skipped automatically. To rerun from scratch, use:

```bash
FORCE=1 GPUS=0,1,2,3 make diagnose-parallel
```

Plot the training-free diagnostics:

```bash
make figure-diagnostics
ls -lh outputs/figures/no_train_diagnostics_overview.*
```

This writes a PNG, a PDF, and a small ranked CSV. The overview figure includes
temporal perturbation sensitivity, segment-level structure, token
compression/diversity, KNN sanity checks, and a raw-value annotated heatmap for
all available metrics.

If one encoder is blocked by a missing dependency, run the rest first:

```bash
ENCODERS=clip-vit-l-14-336,siglip-so400m,siglip2-so400m,dinov2-vitl14,videomaev2-base,vjepa2-vith-256 \
GPUS=0,1,2,3 \
make diagnose-parallel
```

Run a small projector pilot on representative encoders:

`make train-pilot` rebuilds `data/benchmarks/hf_video_debug_mcq.jsonl` from
`data/manifests/hf_video_debug.jsonl`, so stale ActivityNet MCQ files are not reused.

```bash
GPUS=1,2,3 \
ENCODERS=clip-vit-l-14-336,siglip2-so400m,vjepa2-vith-256 \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
GRAD_ACCUM=8 \
make train-pilot
```

If each job fits in memory, run the pilot over all configured encoders:

```bash
GPUS=0,1,2,3 \
ENCODERS=clip-vit-l-14-336,siglip-so400m,siglip2-so400m,internvit-300m,dinov2-vitl14,videomaev2-base,vjepa2-vith-256,internvideo2-clip-s \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
GRAD_ACCUM=8 \
make train-pilot
```

This pilot is for checking whether the controlled training path works and how
different encoder families behave under the same frozen LLM and projector setup.
It is not the final benchmark result, because the debug manifest is small.

## 7. Prepare benchmark manifests

Evaluation expects the same JSONL schema. For multiple-choice benchmarks, use:

```json
{
  "id": "mvbench_000001",
  "benchmark": "MVBench",
  "source": "MVBench",
  "task": "mcq",
  "media_type": "video",
  "media_path": "/data/benchmarks/MVBench/videos/example.mp4",
  "question": "What happens after the person opens the box?",
  "choices": ["He sits down.", "He takes out an object.", "He leaves.", "The video ends."],
  "answer": "B"
}
```

Keep benchmark data separate from training data. If your training mix includes source datasets such as NExT-QA, ActivityNet-QA, or PerceptionTest, remove the overlapping split before evaluation.

## 8. Extract benchmark features

Use the same encoder names, but a benchmark manifest:

```bash
bash scripts/run_all_extract.sh \
  data/benchmarks/mcq_all.jsonl \
  features/benchmarks
```

## 9. Evaluate and aggregate

```bash
bash scripts/run_all_eval.sh \
  data/benchmarks/mcq_all.jsonl \
  features/benchmarks \
  checkpoints/projectors \
  outputs/eval

python scripts/aggregate_results.py \
  --eval-root outputs/eval \
  --out outputs/benchmark_table.csv
```

The final comparison table will be:

```text
outputs/benchmark_table.csv
```

Rows are encoders. Columns are benchmarks. Values are accuracies.

## 10. MVBench shortcut-filter evaluation

For MVBench, use the stricter analysis protocol:

1. evaluate text-only Qwen with no visual input,
2. evaluate each trained encoder/projector with `single`, `reverse`, and `shuffle` frames,
3. exclude questions solved by those shortcuts,
4. compare encoder/projector accuracy on the remaining hard subset.

If MVBench is stored at `/home/woojunghan_google_com/hf_cache/mvbench_video`,
run:

```bash
source /mnt/disks/data/vlm_encoder_benchmark/env.storage
export VLMEB_LOCAL_FILES_ONLY=1

GPUS=0,1,2,3 \
MAX_TOKENS=64 \
MAX_LENGTH=1024 \
ALLOW_MISSING_MEDIA=1 \
make mvbench-eval
```

This writes:

```text
data/benchmarks/mvbench_all.jsonl
features/mvbench/
outputs/mvbench/text_only/
outputs/mvbench/projector_eval/
outputs/mvbench/analysis/filter_summary.csv
outputs/mvbench/analysis/per_item_filters.csv
outputs/mvbench/analysis/shared_hard_ids.jsonl
outputs/mvbench/analysis/mvbench_filter_overview.png
outputs/mvbench/analysis/mvbench_filter_distribution.png
outputs/mvbench/analysis/mvbench_filter_rates.png
outputs/mvbench/analysis/mvbench_accuracy_after_filtering.png
outputs/mvbench/analysis/mvbench_remaining_question_counts.png
outputs/mvbench/analysis/mvbench_task_hardness_heatmap.png
```

`filter_summary.csv` contains the main numbers: text-only count, single-frame
shortcut count, reverse/shuffle shortcut count, remaining hard-question count,
all-question accuracy, per-encoder hard accuracy, and shared-hard accuracy.

## Practical run order

For a new server, do this first:

```bash
make install

make storage

make video-debug

make diagnose

make perturb-mcq
```

This gives you:

- `outputs/no_train_diagnostics_table.csv`: all-encoder representation diagnostics
- `outputs/zeroshot_perturbation_mcq_table.csv`: original/reverse/shuffle MCQ counts for text-aligned encoders

Run the no-training diagnostic before full projector training:

```bash
python scripts/analyze_encoder_no_train.py \
  --manifest data/manifests/hf_video_debug.jsonl \
  --encoder clip-vit-l-14-336 \
  --out-jsonl outputs/no_train_diagnostics/clip-vit-l-14-336/per_example.jsonl \
  --out-csv outputs/no_train_diagnostics/clip-vit-l-14-336/summary.csv \
  --limit 20
```

## Interpreting results

Use precise wording:

- Good: **controlled frozen-encoder VLM comparison**
- Good: **encoder utility under identical projector-only alignment**
- Avoid: **pure encoder comparison**

The projector is trained for every encoder, so the result includes how well the frozen representation can be aligned to the same frozen LLM under the same budget. That is still a fair and useful comparison, but it is not a projector-free intrinsic representation test.

## Creating the GitHub repository

From your local machine:

```bash
git init
git add .
git commit -m "Initial controlled encoder benchmark scaffold"
gh repo create video-encoder-vlm-benchmark --private --source=. --remote=origin --push
```

If `gh` is not authenticated:

```bash
gh auth login -h github.com
```

Or create an empty repo on GitHub web UI, then:

```bash
git remote add origin git@github.com:<YOUR_USER>/video-encoder-vlm-benchmark.git
git branch -M main
git push -u origin main
```
