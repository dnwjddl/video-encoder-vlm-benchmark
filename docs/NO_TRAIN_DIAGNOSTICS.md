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

`cycle_shift_distance`

- Cosine distance between original and circularly shifted-frame representation.
- Useful for detecting whether an encoder cares about absolute temporal phase.

`half_swap_distance`

- Cosine distance between original and swapped first/second half representation.
- Useful for seeing whether beginning/end ordering is preserved.

`stride_distance`

- Cosine distance between original and sparse-frame representation.
- Large values suggest the representation depends heavily on dense temporal coverage.

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

`segment_distance_correlation`

- Correlation between temporal gap and segment representation distance.
- Positive values mean farther-apart moments tend to be farther apart in representation space.

`token_effective_rank`

- Effective rank of the visual token matrix after encoder compression.
- Higher values suggest more independent visual directions are preserved.

`token_rank_ratio`

- `token_effective_rank / min(num_tokens, hidden_dim)`.
- Easier to compare across encoders with different token counts.

`token_top1_energy_ratio`

- Fraction of token variance captured by the strongest direction.
- Very high values can indicate representation collapse or over-compression.

`token_mean_pairwise_distance`

- Mean pairwise cosine distance among visual tokens.
- Higher values suggest tokens are less redundant.

`knn_top1` / `knn_topK`

- Label consistency using nearest neighbors in frozen pooled-embedding space.
- This is still no-training: no probe weights are fitted.
- It only appears when the manifest contains a useful `label` field.

## How to run

```bash
python scripts/analyze_encoder_no_train.py \
  --manifest data/manifests/hf_video_debug.jsonl \
  --encoder vjepa2-vith-256 \
  --out-jsonl outputs/no_train_diagnostics/vjepa2-vith-256/per_example.jsonl \
  --out-csv outputs/no_train_diagnostics/vjepa2-vith-256/summary.csv
```

Run all configured encoders:

```bash
bash scripts/run_all_no_train_analysis.sh \
  data/manifests/hf_video_debug.jsonl \
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
- Estimate representation label consistency with kNN when labels are available.

Bad use:

- Do not treat this as final temporal reasoning accuracy.
- Do not compare these values directly with benchmark QA accuracy.
- Do not claim an encoder can reason just because it has high temporal sensitivity.

The no-training diagnostics are a microscope, not the scoreboard.

## Recommended scale

- `100` videos: smoke test only.
- `1,000` videos: first meaningful representation diagnostic.
- `5,000+` videos: more stable encoder comparison.

If you want reasoning accuracy without projector training, use zero-shot multiple-choice scoring only for text-aligned encoders such as CLIP/SigLIP-style models. DINOv2, VideoMAE, and V-JEPA-style encoders do not have a native text space, so they need a probe/projector for true QA evaluation.
