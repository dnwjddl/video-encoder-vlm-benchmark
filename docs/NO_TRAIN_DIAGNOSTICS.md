# No-training encoder diagnostics

These diagnostics run before projector training. They do not answer the final VQA benchmark question directly. They answer a narrower representation question:

> Does the frozen visual encoder respond to temporal order and preserve segment-level changes before any LLM alignment?

## Metrics

`order_distance`

- Cosine distance between the original video representation and reversed-frame representation.
- For image encoders, this should be near zero because frames are encoded independently and averaged.
- For native video encoders, a larger value means the representation is more sensitive to temporal direction/order.

`shuffle_distance`

- Cosine distance between original and randomly shuffled-frame representation.
- Larger value means the encoder reacts more strongly when temporal coherence is broken.

`segment_diversity`

- Mean pairwise cosine distance among segment embeddings from the same video.
- Larger value means different parts of the video remain more distinguishable after compression.

`segment_adjacent_distance`

- Mean distance between neighboring segment embeddings.

`segment_far_distance`

- Mean distance between non-neighboring segment embeddings.

`segment_temporal_margin`

- `segment_far_distance - segment_adjacent_distance`.
- Positive values suggest farther-apart moments are represented as more different than nearby moments.

## How to run

```bash
python scripts/analyze_encoder_no_train.py \
  --manifest data/manifests/train_debug.jsonl \
  --encoder vjepa2-vith-256 \
  --out-jsonl outputs/no_train_diagnostics/vjepa2-vith-256/per_example.jsonl \
  --out-csv outputs/no_train_diagnostics/vjepa2-vith-256/summary.csv
```

Run all configured encoders:

```bash
bash scripts/run_all_no_train_analysis.sh \
  data/manifests/train_debug.jsonl \
  outputs/no_train_diagnostics

python scripts/aggregate_diagnostics.py \
  --diagnostics-root outputs/no_train_diagnostics \
  --out outputs/no_train_diagnostics_table.csv
```

## Interpretation

Good use:

- Filter out encoders that are almost insensitive to temporal order.
- Check whether a video encoder compresses all segments into nearly the same vector.
- Compare image encoders against native video encoders before LLM/projector training.

Bad use:

- Do not treat this as final temporal reasoning accuracy.
- Do not compare these values directly with benchmark QA accuracy.

The no-training diagnostics are a microscope, not the scoreboard.
