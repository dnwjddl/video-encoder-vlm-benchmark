# Perturbation MCQ Evaluation

This evaluation checks whether a model still answers correctly when the video order is destroyed.

It runs three versions of each multiple-choice question:

- `original`: original video order
- `reverse`: reversed frame order
- `shuffle`: randomly shuffled frame order

For each question, the script records:

- `correct_original`
- `correct_reverse`
- `correct_shuffle`
- `temporal_sensitive_reverse`
- `temporal_sensitive_shuffle`
- `temporal_sensitive_any`

The key definition is:

```text
temporal_sensitive_any =
  correct_original
  and (not correct_reverse or not correct_shuffle)
```

This matches the intended filtering rule:

> If the model gets the original video correct but fails after reverse/shuffle, treat that question as temporal-sensitive for that encoder.

The opposite case is also counted:

```text
robust_correct_all =
  correct_original and correct_reverse and correct_shuffle
```

Those are examples where the model still answers correctly after order perturbation. They may be non-temporal, answerable from static frames, or answerable through shortcuts.

## No-training scope

Without projector/LLM training, this can only run on text-aligned visual encoders that expose a text embedding space, such as:

- CLIP
- SigLIP
- SigLIP2
- some CLIP-style video encoders

It does not work directly for:

- DINOv2
- VideoMAE V2
- V-JEPA2

Those encoders do not have a native text option scoring space, so true MCQ correctness needs a trained probe, projector, or LLM connector.

## Run

```bash
HF_HOME=/data/hf_cache python scripts/evaluate_zeroshot_perturbation_mcq.py \
  --manifest data/benchmarks/mcq_all.jsonl \
  --encoder siglip2-so400m \
  --out-jsonl outputs/zeroshot_perturbation_mcq/siglip2-so400m/predictions.jsonl \
  --out-csv outputs/zeroshot_perturbation_mcq/siglip2-so400m/summary.csv
```

Run the default text-aligned image encoders:

```bash
HF_HOME=/data/hf_cache bash scripts/run_text_aligned_perturbation_mcq.sh \
  data/benchmarks/mcq_all.jsonl \
  outputs/zeroshot_perturbation_mcq

python scripts/aggregate_perturbation_mcq.py \
  --root outputs/zeroshot_perturbation_mcq \
  --out outputs/zeroshot_perturbation_mcq_table.csv
```

## Summary columns

- `original_correct`: number of questions correct on the original video
- `reverse_correct`: number correct after reversing frames
- `shuffle_correct`: number correct after shuffling frames
- `robust_correct_all`: number correct in all three settings
- `temporal_sensitive_reverse`: original correct, reverse wrong
- `temporal_sensitive_shuffle`: original correct, shuffle wrong
- `temporal_sensitive_any`: original correct, reverse or shuffle wrong
- `perturbation_helped`: original wrong, but reverse or shuffle correct
- `prediction_changed_reverse`: prediction changes under reverse
- `prediction_changed_shuffle`: prediction changes under shuffle

Use `temporal_sensitive_any` as the count of examples that behave like temporal questions under this encoder.
