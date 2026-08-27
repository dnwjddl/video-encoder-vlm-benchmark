# Video Encoder VLM Benchmark

Controlled benchmark harness for comparing frozen visual encoders under the same LLM and projector-only alignment setup.

The main research question is:

> If we freeze the visual encoder and freeze `Qwen/Qwen2.5-7B-Instruct`, which image/video encoder transfers best to temporal video understanding benchmarks under the same alignment budget?

This repo intentionally separates three things:

1. **Encoder feature extraction**
2. **Projector-only training**
3. **Benchmark evaluation**

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
- `internvideo2-clip-1b`: `OpenGVLab/InternVideo2-CLIP-1B-224p-f8`

Note: `internvideo2-clip-1b` is gated on Hugging Face. Accept the model terms before running it on a server.

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

## 2. Prepare training manifest

The script below creates a unified JSONL manifest from public annotation datasets.

```bash
python scripts/download_data.py \
  --out data/manifests/train_230k.jsonl \
  --streaming \
  --caption-count 100000 \
  --qa-count 100000 \
  --mcq-count 30000 \
  --video-root /data/videos
```

Important: this downloads/streams **annotations**. Many video datasets do not redistribute all raw videos in one simple archive. Put videos on the server, then set `--video-root` so each row's `media_path` points to the correct local file.

If you already have local annotation JSON/JSONL files:

```bash
python scripts/download_data.py \
  --out data/manifests/train_230k.jsonl \
  --local-json /path/to/local_annotations.jsonl \
  --video-root /data/videos \
  --caption-count 100000 \
  --qa-count 100000 \
  --mcq-count 30000
```

The unified schema is documented in [docs/MANIFEST_SCHEMA.md](docs/MANIFEST_SCHEMA.md).

## 3. Extract features for each encoder

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

## 4. Train projector only

Default setup:

- visual encoder: frozen
- `Qwen/Qwen2.5-7B-Instruct`: frozen
- projector: trainable 2-layer MLP

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

## 5. Prepare benchmark manifests

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

## 6. Extract benchmark features

Use the same encoder names, but a benchmark manifest:

```bash
bash scripts/run_all_extract.sh \
  data/benchmarks/mcq_all.jsonl \
  features/benchmarks
```

## 7. Evaluate and aggregate

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

## Practical run order

For a new server, do this first:

```bash
make install

python scripts/download_data.py \
  --out data/manifests/train_debug.jsonl \
  --streaming \
  --max-rows-per-dataset 2000 \
  --caption-count 1000 \
  --qa-count 1000 \
  --mcq-count 500 \
  --video-root /data/videos

python scripts/extract_features.py \
  --manifest data/manifests/train_debug.jsonl \
  --encoder clip-vit-l-14-336 \
  --out-dir features/debug/clip-vit-l-14-336 \
  --allow-missing-media \
  --limit 10
```

Then remove `--allow-missing-media` once your video paths are correct. It is better to catch path problems early than to discover them three hours into a run. Tiny tragedy, preventable.

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
