# Frozen VideoLLM information upper-bound analysis

This folder is an end-to-end diagnostic suite for one concrete question:

> With the VideoLLM and projector held fixed, which information must a visual
> encoder deliver for video understanding performance to improve, and how much
> of each kind is enough?

It is not a synthetic demo. The adapters consume local releases of real video
benchmarks, the view extractor decodes the real videos at annotated timestamps,
the runner uses the repository's frozen encoder/projector/LLM interface, and the
analysis reports paired, cluster-bootstrapped uncertainty. Small fixtures under
`tests/` verify software contracts only; they are never used as scientific data.

## What is being identified

Every base question is evaluated under controlled interventions. The useful
contrasts are:

| Contrast | Interpretation when the second condition improves |
|---|---|
| `question_only -> full_video` | usable visual information is present |
| `single_frame -> full_video` | information beyond a static key frame matters |
| `shuffled_frames -> full_video` | temporal order is useful |
| `atomic_oracle -> ordered_oracle` | timestamps/order/bindings are the missing clue |
| `ordered_timestamp_sham -> ordered_oracle` | real timestamps/order help beyond a timing-shaped but uninformative prompt |
| `atomic_oracle -> ordered_timestamp_sham` | placebo check for timestamp-like formatting alone |
| `ordered_oracle -> reasoning_oracle` | facts are available, but an operator/intermediate step is needed by the frozen LLM |
| `reasoning_operator_sham -> reasoning_oracle` | operator semantics help beyond a character-layout-matched meaningless operator |
| `ordered_oracle -> reasoning_operator_sham` | placebo check for an operator-shaped prompt alone |
| `ordered_oracle -> ordered_embedding_oracle` | checks how much of the oracle gain depends on ordinary text placement rather than continuous-token placement |
| `full_video -> video_plus_ordered_oracle` | the current encoder channel omits usable structured facts |
| `full_video -> evidence_only` | irrelevant context or visual bandwidth hurts |
| `evidence_removed -> evidence_present` | the annotated evidence is necessary within an exactly matched sampled-frame set |
| `random_position_mask -> evidence_present` | generic same-count masking cost on that exact frame grid |
| `evidence_removed -> random_position_mask` | masking annotated evidence hurts more than masking other positions |
| `random_matched -> evidence_only` | improvement comes from the evidence, not shorter duration |

`reverse` and `shuffle` are destructive sensitivity controls with the original
label; they are not relabeled counterfactual videos. Claims about correct answer
flips come only from dataset-provided original/counterfactual video pairs: MVP
and the subset of TempCompass original/reverse rows whose question and semantic
options are identical and whose semantic answer changes. CLEVRER
`counterfactual` is an official *question type*, not a paired-video
intervention; its candidate rows are grouped by `independent_unit_id` and must
not be reported as an original/counterfactual video flip.

The text oracle is an *operational dataset ceiling*, not a mathematical upper
bound. It answers “could this frozen LLM solve the item if these visual facts
were delivered clearly?” A weak oracle score means the encoder alone cannot fix
the item under the current frozen LLM and answer interface. A strong oracle score
shows headroom, not that the same facts are already recoverable from pixels.
`embedding_oracle` inserts the same answer-independent clue through frozen LLM
input embeddings at the visual-token position. It is a channel control that
bypasses the projector; it must not be described as an achievable encoder
representation or compared as if it had passed through the visual projector.

The two text shams make the temporal and reasoning claims more specific. The
timestamp sham uses exactly the selected event IDs and semantic facts, but
replaces time with fixed neutral timestamp-shaped prefixes and uses an
answer-independent hash order. The operator sham uses the same ordered facts
and replaces every letter/digit of the operator with `x`/`0`, preserving its
exact whitespace, punctuation, and character count. Neither sham reads the
question, choices, or answer when constructing its replacement text.

## Folder layout

```text
information_upper_bound/
  pyproject.toml         folder-local package and console entry points
  requirements.txt      pip dependency set
  environment.yml       minimal Python 3.11 Conda environment
  environment-linux.yml Linux environment with a pinned GNU C/C++ runtime
  adapters/              strict converters for official dataset formats
  configs/
    encoders.yaml        visual-encoder registry
    conditions.yaml      input interventions and clue-dose grid
    protocol.yaml        locked model/statistical protocol
    exclusions/          audited known official-annotation defects
  tests/                 contract and metric tests; no benchmark claims
  schema.py              versioned diagnostic manifest contract
  validate.py            pair, split, media, leakage, and coverage audit
  data_lock.py           path-independent official-data release lock
  attestation.py         per-trial build-attestation authentication
  trial_matrix.py        external-memory full condition-matrix replay/closure
  integrity.py           model, tensor, feature, trial-set, and result digests
  split_integrity.py     projector train/evaluation unit and media-byte audit
  conditions.py          trial matrix and option counterbalancing
  protocol.py            locked-protocol loading and frozen-model checks
  media.py               timestamp-aware view sampling and decoding
  encoder_runtime.py     immutable encoder-snapshot resolution
  extract_features.py    content-addressed frozen-encoder cache
  projector_training.py  folder-local authenticated training helpers
  scoring.py             shared frozen MCQ likelihood scorer
  run.py                 resumable trial evaluation
  train_projector.py     packaged strict frozen-projector trainer
  metrics.py             paired metrics and cluster-bootstrap confidence intervals
  cli.py                 command dispatcher
```

Generated annotations, videos, features, and outputs stay in the repository's
ignored `data/`, `features/`, and `outputs/` trees; this tracked folder contains
only code and locked configurations.

## Supported evidence roles

The suite supports the following real-data roles. It does not invent annotations
that a release does not provide.

| Dataset | Primary diagnostic role | Important limitation |
|---|---|---|
| TempCompass | direction, speed, action, order, attribute change; validated original/reverse pairs | `eval_dim` target-semantic metadata is excluded from safe clues, so oracle conditions may be unavailable |
| TVBench | ten temporally difficult task families | not every task supplies gold event facts |
| Perception Test | tracking, state, physics, memory, action segments | cut/frame mappings must be applied before using cup-game tracks |
| NExT-GQA | real-video QA with multiple gold evidence spans | preserve every evidence interval rather than merging to one box |
| CLEVRER | explanatory, predictive, and counterfactual questions | this is not a paired-video benchmark; multi-label choices are expanded to binary Yes/No trials and processed scene annotations are needed for fact/reasoning oracles |
| EgoSchema | long-context selection/dilution | no official gold evidence spans; only the public answer subset is offline-scorable |
| MVP | minimal visual changes with opposite answers | every pair must have exactly two videos, identical question/options, and flipped answer |

Dataset repositories remain the authority for licenses and download procedures.
This suite deliberately does not scrape or redistribute their videos.

## Manifest contract

Adapters retain the benchmark harness's normal top-level MCQ fields and add one
versioned namespace:

```json
{
  "id": "next_gqa:qid",
  "source": "next_gqa",
  "benchmark": "next_gqa",
  "task": "mcq",
  "media_type": "video",
  "media_path": "/absolute/path/to/video.mp4",
  "question": "What happened before ...?",
  "choices": ["He entered", "He sat down", "He opened the door", "He left", "He waved"],
  "answer": "C",
  "diagnostic": {
    "schema_version": "1.1",
    "dataset": "next_gqa",
    "split": "validation",
    "information_family": "temporal_order",
    "question_family": "event_order",
    "reasoning_depth": 1,
    "resampling_unit_id": "next_gqa:video:raw-video-id",
    "adapter_run_id": "adapter-run::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "pair_id": "next_gqa:qid",
    "pair_role": "standalone",
    "evidence_spans": [
      {"start": 4.2, "end": 6.8, "unit": "seconds", "role": "necessary"}
    ],
    "oracles": {
      "static_facts": [],
      "unordered_events": [
        {
          "event_id": "event-1",
          "subject": "person",
          "predicate": "opens the door",
          "access": "safe_visual_gt",
          "source": "official_grounding_annotation",
          "lineage": "official_adapter"
        }
      ],
      "ordered_events": [
        {
          "event_id": "event-1",
          "subject": "person",
          "predicate": "opens the door",
          "start_sec": 4.2,
          "end_sec": 6.8,
          "access": "safe_visual_gt",
          "source": "official_grounding_annotation",
          "lineage": "official_adapter"
        }
      ],
      "temporal_relations": [],
      "state_changes": [],
      "relations": [],
      "operator": null,
      "intermediate": [],
      "answer_derived": false
    },
    "provenance": {"source_id": "raw-qid", "annotation_file": "..."}
  }
}
```

Integer answer labels are never guessed. Each adapter declares whether its
official source is zero- or one-indexed. Ambiguous media matches, duplicate
normalized choices, malformed timestamps, incomplete MVP pairs, or source rows
that cannot be joined are build errors in strict mode.

Every oracle fact is an object with explicit `access`, `source`, and `lineage`.
Safe visual facts use `access: safe_visual_gt`; an optional reasoning operator
is a list of objects using `access: operator_only`, while absence is represented
by JSON `null`. Facts marked `target`, `target_semantic`, `answer`, or
`answer_key` may be retained for provenance but are never rendered as encoder
clues. Safe facts must be derived independently from visual/event annotations,
and `answer_derived` must be explicitly `false`.

## Reproducible run

Create the folder-local Conda environment and install the unchanged parent
benchmark package followed by this diagnostic package. On Linux, use the file
that also pins the GNU C/C++ runtime; on macOS, use `environment.yml`. Run these
commands from the repository root:

```bash
# Linux
conda env create -f information_upper_bound/environment-linux.yml

# macOS instead:
# conda env create -f information_upper_bound/environment.yml

conda activate video-iub
python -m pip install -e .
python -m pip install -e ./information_upper_bound
information-upper-bound --help
information-upper-bound-train-projector --help
```

The first editable install exposes the repository's existing `vlmevalbench`
runtime without changing parent files. The second installs only this folder and
its dependencies. `requirements.txt` remains available for environments
managed from a requirements-file workflow.

If Linux reports `CXXABI_1.3.15 not found` while importing `sqlite3`, the
process is loading an older system `libstdc++.so.6` instead of the Conda
runtime. Repair an existing environment and test it without the inherited
library path:

```bash
conda activate video-iub
conda install -c conda-forge --strict-channel-priority \
  "libgcc-ng>=13" "libstdcxx-ng>=13" -y
env -u LD_LIBRARY_PATH python -c \
  "import sqlite3; print(sqlite3.sqlite_version)"
env -u LD_LIBRARY_PATH information-upper-bound --help
```

If the direct `sqlite3` probe succeeds but the console command still fails,
another native module loaded the host C++ runtime first. Update this folder to
the current `main` version, which keeps PyTorch out of lightweight CLI startup,
and refresh the editable install:

```bash
git fetch origin
git merge --ff-only origin/main
python -m pip install -e ./information_upper_bound
env -u LD_LIBRARY_PATH information-upper-bound --help
```

For an immediate one-command diagnostic or temporary workaround, explicitly
preload the environment-local C++ runtime:

```bash
env -u LD_LIBRARY_PATH \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
  information-upper-bound --help
```

When those commands pass, run `unset LD_LIBRARY_PATH` in that shell before the
benchmark. Conda's own troubleshooting guidance recommends removing an
inherited Linux `LD_LIBRARY_PATH` because it can override environment-local C
libraries.

The package also installs the equivalent `information-upper-bound` console
command and the standalone `information-upper-bound-train-projector` trainer.
The checked-in `configs/protocol.yaml` is intentionally an incomplete template.
Trial construction can leave only the late-bound `projector` hashes
unresolved; its data, coverage, model revision, sampling, analysis, comparisons,
and dataset roles must already be final. Scoring and strict analysis fail closed
until the projector section is also populated from final artifacts.

| Stage | Protocol fields enforced | Audit binding |
|---|---|---|
| `adapt` | no protocol is opened | portable source inventory, exact coverage, and `adapter_run_id` in the adapter report |
| `lock-data` | no protocol is opened | merged semantic records, all adapter runs/source bytes, media bytes, and scientific units in `data_release_sha256` |
| `build-trials` | exact release, adapter-run coverage, conditions SHA, model/sampling/analysis design | v2 trial-build attestation carrying the v1 `trial_build_protocol_sha256` payload (all protocol sections except late-bound `projector`) |
| `extract` | v2 trial-build attestation digest, exact release/media lock, locked conditions/sampling, media-SHA and strong-encoder-identity policy | deterministic full-matrix replay, portable closure/root/count, decoded frames, feature tensors/artifacts, and encoder pipeline |
| `score` | final full protocol SHA, frozen flags, exact LLM revision, projector plus matrix-closure locks, dtype, token limit, overflow policy | closure-bound global signature, run-local trial-set identity, and per-row result digest |
| `analyze` | final full protocol SHA, release/build identity, locked full trial root/count, analysis section, comparison order, score sidecars | expected manifest and authenticated score union must both equal the independently locked complete matrix |

Build, extraction, and scoring reject conflicting command-line overrides.
Confirmatory analysis also takes its values from the locked protocol; analysis
CLI overrides are for explicitly watermarked development output only.

### Confirmatory versus development output

Strict mode is the default. Adapter runs using a missing-data escape hatch or
`--limit` are marked `confirmatory_eligible: false`, and `lock-data` rejects
their reports. `build-trials` and `extract` require the release lock unless
`--development` is explicit. `analyze --development` (or its deprecated
`--allow-incomplete` alias) skips score-sidecar authentication and writes a
prominent non-confirmatory warning to `report.json`.

Development is suitable for parser, decoder, cache, and plotting checks only.
Development trial attestations are rejected by the strict scoring CLI, and
there is deliberately no `score --development` switch. Do not mix development
artifacts into a result bundle or interpret a development report as benchmark
evidence.

### 1. Convert one or more official releases

The `adapt --help` command lists dataset-specific required paths. A typical run
has this shape:

```bash
python -m information_upper_bound adapt \
  --dataset next_gqa \
  --annotations /data/NExT-GQA/val.csv \
  --media-root /data/NExTVideo \
  --grounding /data/NExT-GQA/gsub_val.json \
  --video-map /data/NExT-GQA/map_vid_vidorID.json \
  --exclusions information_upper_bound/configs/exclusions/next_gqa_validation.json \
  --output data/information_upper_bound/next_gqa_validation.jsonl \
  --split validation \
  --source-split val
```

The checked-in NExT-GQA exclusion authenticates the one validation row whose
official options repeat `scarf`; the adapter would otherwise reject that row as
an ambiguous MCQ. Keep the exclusion file and its digest with the result bundle.

Adapters never download data. Use each dataset's official instructions, retain
its official split, and point the converter at that local release. Merge
manifests only after validating them independently so dataset-specific failures
remain visible.

Media existence and schema validation are strict by default. The adapter also
writes `<output>.report.json`. Its path-independent `adapter_run_id` binds the
dataset, canonical split, adapter options, portable source-artifact inventory,
and exact retained record IDs; the ID is also copied into every output row.
`source_artifacts` recursively enumerates every supported annotation/sidecar
file with role, relative path, byte size, and SHA256, while
`source_artifact_root_sha256` authenticates the inventory. The report separately
records split, information-family, question-family, evidence, and safe-oracle
coverage.

`confirmatory_eligible` is true only for a complete strict invocation. It is
false for `--limit`, missing-media/grounding/cut escape hatches, missing required
sidecars, a single-task TVBench build, or an MVP build without one explicit
official category. A malformed or ambiguous official row is an error; adapters
do not silently drop it. `--dry-run` prints an audit summary but writes no
manifest/report, so it cannot be passed to `lock-data`.

If an official release has a known bad row, make that decision explicit with an
audited exclusion file:

```json
{
  "exclusions": [
    {
      "dataset": "next_gqa",
      "source_id": "official-question-id",
      "reason": "Documented annotation defect; see audit issue 17"
    }
  ]
}
```

Pass it with `--exclusions exclusions.json` (JSONL with one entry per line is
also accepted). Matching is scoped by the composite `(dataset, source_id)` key.
For the active dataset, every entry must match exactly one raw row: duplicate
keys, an unused entry, or one entry matching multiple rows fails the build. The
SHA256, reason, raw annotation location, and applied IDs are written to the
adapter report and copied into retained-row provenance. `adapt --help` lists
stable row-index fallbacks for malformed rows whose official ID cannot be read.
Entries for other datasets in a shared exclusion file are counted but are not
required to be consumed by the current adapter invocation.
Exclusions must close over the scientific unit: both members of a
TempCompass/MVP pair and every CLEVRER candidate in one official question must
be excluded together. Partial-unit exclusions are build errors.

For Perception Test train/valid MCQ, pass the official
`--cut-frame-mapping`; the visible interval is `[0, cut_frame)` and the cut is
propagated into every sampled view and oracle annotation. The public
`--allow-missing-cut-mapping` escape hatch is annotation-inspection only: it can
expose post-cut answers and must not be used in a benchmark result.

For CLEVRER, pass `--scene-annotations` pointing to the official
`processed_proposals` directory (or the exact `sim_XXXXX.json`). Without that
sidecar, answer rows remain usable for visual controls, but the adapter does not
invent event facts and the oracle conditions are skipped. Multi-choice
explanatory/predictive/counterfactual questions are expanded into binary
candidate rows sharing one `independent_unit_id`; the analysis clusters them
back to the official question. These rows remain `pair_role=standalone` even
when the official question type is counterfactual.

### 2. Audit before model inference

```bash
python -m information_upper_bound validate \
  --manifest data/information_upper_bound/next_gqa_validation.jsonl \
  --out outputs/information_upper_bound/next_gqa_validation.json \
  --require-media --fail-on-warning
```

The report includes ID uniqueness, answer-position balance, information-family
coverage, pair cardinality, counterfactual answer flips, nuisance invariance,
pair/media split leakage, media presence, and evidence-record/span coverage by
dataset, together with the complete validation issue list. Intentional drops
belong in the adapter exclusion audit described above. Do not use
`--allow-validation-errors` for a reported experiment.

### 3. Merge and lock the official data release

After every adapter output and report passes its own audit, concatenate the
JSONL rows into one final manifest. For the checked-in seven-dataset coverage
contract, MVP contributes four separately adapted official category runs:

```bash
cat \
  data/information_upper_bound/tempcompass_test.jsonl \
  data/information_upper_bound/tvbench_test.jsonl \
  data/information_upper_bound/perception_test_validation.jsonl \
  data/information_upper_bound/next_gqa_validation.jsonl \
  data/information_upper_bound/clevrer_validation.jsonl \
  data/information_upper_bound/egoschema_public_500.jsonl \
  data/information_upper_bound/mvp_human_object_interactions.jsonl \
  data/information_upper_bound/mvp_robot_object_interactions.jsonl \
  data/information_upper_bound/mvp_intuitive_physics.jsonl \
  data/information_upper_bound/mvp_temporal_reasoning.jsonl \
  > data/information_upper_bound/official_merged.jsonl
```

Freeze that exact manifest with every corresponding report (repeat
`--adapter-report` once per adapter run):

```bash
python -m information_upper_bound lock-data \
  --manifest data/information_upper_bound/official_merged.jsonl \
  --adapter-report data/information_upper_bound/tempcompass_test.jsonl.report.json \
  --adapter-report data/information_upper_bound/tvbench_test.jsonl.report.json \
  --adapter-report data/information_upper_bound/perception_test_validation.jsonl.report.json \
  --adapter-report data/information_upper_bound/next_gqa_validation.jsonl.report.json \
  --adapter-report data/information_upper_bound/clevrer_validation.jsonl.report.json \
  --adapter-report data/information_upper_bound/egoschema_public_500.jsonl.report.json \
  --adapter-report data/information_upper_bound/mvp_human_object_interactions.jsonl.report.json \
  --adapter-report data/information_upper_bound/mvp_robot_object_interactions.jsonl.report.json \
  --adapter-report data/information_upper_bound/mvp_intuitive_physics.jsonl.report.json \
  --adapter-report data/information_upper_bound/mvp_temporal_reasoning.jsonl.report.json \
  --out data/information_upper_bound/official_release.lock.json
```

`lock-data` rejects a missing, duplicate, overlapping, debug-ineligible, or
content-mismatched report. It hashes all official source artifacts and media,
binds media blobs back to exact record IDs, and records namespaced scientific
unit summaries. `data_release_sha256` is computed from semantic records,
portable source identities, adapter runs, media bindings, and unit summaries;
absolute mount paths and row order do not affect it. The lock also retains
local audit paths and a separate lock-file digest, so copying an identical
release to a different mount preserves the scientific release identity.

Before building trials, finalize every **non-projector** section in
`information_upper_bound/configs/protocol.yaml`:

1. Copy the lock's `data_release_sha256` into
   `data.data_release_sha256`.
2. Copy the exact `adapter_run_id` values from the reports/lock into each
   dataset's `required_adapter_run_ids`; keep the declared run count, split,
   source roles, families, and minimum coverage consistent with the intended
   final release.
3. Verify the bytes of `configs/conditions.yaml` (for example with
   `shasum -a 256 information_upper_bound/configs/conditions.yaml`) and copy the
   digest to `data.conditions_sha256` if the condition matrix changed.
4. Pin `model.llm_revision` to the exact 40--64 hexadecimal hub commit and
   freeze the model, sampling, analysis, confirmatory-comparison, and
   dataset-role sections. Pin the
   visual encoder revision in
   `information_upper_bound/configs/encoders.yaml` or pass the same immutable
   revision with `extract --model-revision`; unresolved mutable model aliases
   are rejected by the strong-identity policy.

At this point the `projector` section may still contain `REPLACE_...` values.
The v2 trial-build attestation carries a canonical v1
`trial_build_protocol_sha256` payload over every other protocol section,
avoiding a cycle between trial-manifest SHA and projector metadata.

### 4. Freeze the trial matrix

```bash
python -m information_upper_bound build-trials \
  --manifest data/information_upper_bound/official_merged.jsonl \
  --data-lock data/information_upper_bound/official_release.lock.json \
  --config information_upper_bound/configs/conditions.yaml \
  --protocol-config information_upper_bound/configs/protocol.yaml \
  --out data/information_upper_bound/official_trials.jsonl.gz
```

Use this expanded trial file later as the projector's predeclared evaluation
manifest. The build report's full-file SHA is representation audit only. The
portable scientific identity is established before encoder inference by
reconstructing the locked base records, deterministically replaying the entire
condition grid, and hashing the exact set of content-derived trial IDs.

The output records one `trial_id` per base item × condition × clue dose × option
permutation. `visual_id` depends on stable dataset/source identity and the view
specification, not on the clue or option order, so clue and option trials share
one cached encoder tensor. The release lock already binds each base record to
media bytes, and extraction also places the current media identity into the
feature content address. Keep the generated report and manifest hash with the
experiment. Trial expansion is streamed into an atomic JSONL writer; using a
`.gz` suffix produces deterministic gzip output. Base-manifest validation still
materializes the base rows in memory before expansion.
Base inputs may not predeclare expansion-owned fields such as `base_id`,
`condition`, `visual_spec`, `clue_text`, trial hashes, or build-lock fields, so
the reverse reconstruction used by the closure check is unambiguous.

Confirmatory construction re-hashes current media against the lock, enforces the
exact dataset/adapter-run/split/family/source-role coverage contract, and embeds
one v2 `trial_build_attestation` in every row. That attestation binds
confirmatory mode, `data_release_sha256`, the condition-file SHA, the v1
`trial_build_protocol_sha256` payload, seed, option permutations, and shard
count. The protocol payload excludes only the late-bound `projector` section.
The attestation digest and release digest are part of `trial_content_sha256`, so
a trial cannot be moved to
a different release or preregistered design without changing its ID.

Each `trial_id` is `trial::<trial_content_sha256>`. The full content hash binds
the question, choices and answer, clue, condition, analysis-critical diagnostic
metadata, stable visual identity, and view specification. Absolute media and
annotation paths are deliberately excluded, so moving the same release does
not change trial IDs. Both extraction and scoring recompute this hash and reject
stale or manually edited rows.

The default `option_permutations: all` creates one cyclic permutation per answer
position for every item. Thus mixed 2/3/4/5-choice manifests are exactly
counterbalanced *within each item*, while all semantic choices and the gold
answer text remain unchanged. A fixed integer is available for development but
weakens that guarantee when it is not a multiple of the number of choices.

The build command treats `sampling.seed`, `sampling.option_permutations`, and
`sampling.trial_shards` in `protocol.yaml` as locked. Conflicting values in
`conditions.yaml` or on the command line are errors. Although trial construction
supports stable `resampling_unit_id` sharding, the current locked scoring path
requires `sampling.trial_shards: 1`: projector metadata authenticates one full
portable trial-set closure. Keep it at one and score the full authenticated
matrix for a confirmatory run. Sharded construction remains useful for
development/infrastructure work, and analysis can authenticate a union of
multiple score sidecars, but that does not bypass the current scoring lock.

### 5. Extract every unique visual view once

```bash
CUDA_VISIBLE_DEVICES=0 python -m information_upper_bound extract \
  --manifest data/information_upper_bound/official_trials.jsonl.gz \
  --data-lock data/information_upper_bound/official_release.lock.json \
  --encoder internvideo2-clip-s \
  --encoder-config information_upper_bound/configs/encoders.yaml \
  --model-revision REPLACE_WITH_IMMUTABLE_ENCODER_COMMIT \
  --conditions-config information_upper_bound/configs/conditions.yaml \
  --protocol-config information_upper_bound/configs/protocol.yaml \
  --out-dir features/information_upper_bound/official/internvideo2-clip-s \
  --device cuda --dtype bf16 --media-sha256
```

The sampler records source-frame indices, timestamps, resolved evidence spans,
FPS, duration, timestamp source/digest, decoder backend, media fingerprint, view
specification, encoder configuration, and content hashes. With decord,
per-frame presentation timestamp intervals (normalized so the first PTS is
time zero) drive clip and evidence overlap, including variable-frame-rate
video. OpenCV and decord builds exposing only average FPS are marked
unverified; second-based evidence or clip operations fail instead of silently
using average FPS as a frame clock.

`evidence_only`, `evidence_present`, `evidence_removed`, `random_position_mask`,
and `random_matched` use the same requested model frame count.
`evidence_present`, `evidence_removed`, and `random_position_mask` share sampled
frame indices. `evidence_removed` masks only sampled evidence positions;
`random_position_mask` masks exactly the same number of positions, choosing
non-evidence positions first and using evidence positions only when there are
not enough alternatives. The selected evidence/mask positions, target count,
and unavoidable overlap count are recorded, separating evidence necessity from
the generic cost of masking frames.
`random_matched` deterministically selects non-evidence time
support whose duration equals the union of the annotated evidence spans in the
decoder timestamp domain. It prefers one contiguous window and otherwise uses
multiple non-evidence spans. Target duration, actual duration, absolute error,
chosen spans, and strategy are recorded; extraction fails when sufficient
non-evidence time coverage does not exist.

Before loading the encoder, confirmatory extraction inverts each answer-option
permutation, reconstructs one canonical base record per `base_id`, requires its
semantic root and exact ID set to equal the data lock, and regenerates the full
condition × dose × option-permutation matrix from `--conditions-config`. The
actual and regenerated `(trial_id, trial_content_sha256)` sets must be exactly
equal. Deleting one condition, dose, permutation, or candidate therefore fails
even if every base/video is still represented. The validator uses a temporary
SQLite index and streams expanded rows, so million-row matrices are checked
without materializing them in Python memory. It also re-hashes every unique
video against the release lock. This is separate from `--media-sha256`, which
places `{sha256, size_bytes}` in each feature content address. The locked
protocol currently requires that flag. In development mode, omitting it falls
back to `{size_bytes, mtime_ns}` and is not content-safe.

Changing the declared encoder model/configuration, frame count, token
compression, evidence intervals, seed, media identity, or view creates a
different cache key. Cached artifacts are checked against their feature/view
hashes, encoder configuration, decoded RGB-frame hash, media identity, tensor
hash, and full feature-file hash before reuse/scoring. The encoder identity is
resolved to a Hugging Face commit/snapshot or a content hash for a local model;
an unresolved mutable alias is rejected when
`sampling.require_strong_encoder_identity: true`. The extraction metadata also
records the release/build attestation, portable `trial_matrix_closure`, full
trial-set root/count, current full protocol SHA as representation audit,
resolved seed,
decoder/software pipeline identity, feature-index SHA, and feature-artifact
root. A conflicting `--seed` is an error.
`index.jsonl` and `metadata.json` are never replaced implicitly. Use a new
output directory for each run, or pass `--overwrite` deliberately when
rebuilding those two files; content-addressed tensor artifacts may still be
reused after their hashes are verified.

### 6. Finalize and lock the projector

After the evaluation features are extracted, train or select the projector with
authenticated **training** feature index/metadata from the same frozen encoder
pipeline, and pass `official_trials.jsonl.gz` to the packaged
`information-upper-bound-train-projector` command with `--eval-manifest`. The
trainer authenticates every training feature and runs `split_integrity.py`;
training and evaluation may share neither
`resampling_unit_id` nor source-media SHA256.

```bash
VLMEB_LOCAL_FILES_ONLY=1 CUDA_VISIBLE_DEVICES=0 \
information-upper-bound-train-projector \
  --manifest data/information_upper_bound/projector_train.jsonl.gz \
  --feature-index features/information_upper_bound/projector_train/internvideo2-clip-s/index.jsonl \
  --feature-metadata features/information_upper_bound/projector_train/internvideo2-clip-s/metadata.json \
  --eval-manifest data/information_upper_bound/official_trials.jsonl.gz \
  --eval-feature-index features/information_upper_bound/official/internvideo2-clip-s/index.jsonl \
  --eval-feature-metadata features/information_upper_bound/official/internvideo2-clip-s/metadata.json \
  --eval-data-lock data/information_upper_bound/official_release.lock.json \
  --conditions-config information_upper_bound/configs/conditions.yaml \
  --protocol-config information_upper_bound/configs/protocol.yaml \
  --out-dir checkpoints/projectors/information_upper_bound/internvideo2-clip-s/run-001 \
  --encoder-name internvideo2-clip-s \
  --llm-id Qwen/Qwen2.5-7B-Instruct \
  --llm-revision REPLACE_WITH_IMMUTABLE_LLM_COMMIT \
  --dtype bf16 --max-length 4096 --seed 42
```

The equivalent module form is
`python -m information_upper_bound.train_projector`.

The five strict-only provenance arguments (`--feature-metadata`,
`--eval-manifest`, `--eval-feature-index`, `--eval-feature-metadata`, and
`--eval-data-lock`) are all-or-none. Omitting all five preserves the
repository's generic legacy training path, but that path cannot produce a
confirmatory projector lock. Strict training refuses a non-empty output
directory unless `--overwrite` is explicit; use a dedicated run directory so
old checkpoints cannot be mistaken for the current run.

Each saved checkpoint directory contains `protocol_projector_lock.json`. Copy
its checkpoint/metadata, train/evaluation manifest, **both** train/evaluation
feature-index, feature-metadata, and artifact-root hashes, encoder-pipeline,
full evaluation matrix closure/root/count, training-LLM, dtype, maximum-length,
and seed fields into the protocol's
`projector` section. This is the one expected late-bound edit: it does not
change `trial_build_protocol_sha256` or any trial ID. Do not change another
protocol section. Extraction validated only the v2 trial-build attestation
digest plus the data/media lock and recorded its then-current full SHA as audit
metadata; score and
analysis now bind the SHA of this final full protocol file and enforce the
projector lock, including the exact evaluation feature bundle and the complete
portable trial set. The closure is late-bound only under `projector`, so it
does not create a hash cycle with trial IDs.

### 7. Score with the fixed VideoLLM

```bash
VLMEB_LOCAL_FILES_ONLY=1 CUDA_VISIBLE_DEVICES=0 \
python -m information_upper_bound score \
  --trials data/information_upper_bound/official_trials.jsonl.gz \
  --feature-index features/information_upper_bound/official/internvideo2-clip-s/index.jsonl \
  --feature-metadata features/information_upper_bound/official/internvideo2-clip-s/metadata.json \
  --projector-ckpt checkpoints/projectors/internvideo2-clip-s/step_XXXXXX/projector.pt \
  --projector-metadata checkpoints/projectors/internvideo2-clip-s/step_XXXXXX/metadata.json \
  --protocol-config information_upper_bound/configs/protocol.yaml \
  --out outputs/information_upper_bound/official/internvideo2-clip-s/predictions.jsonl \
  --device cuda --resume
```

One shared scorer is used for question-only, text-oracle, embedding-oracle,
visual, and visual-plus-text conditions. Every channel receives the exact same
surface MCQ prefix and answer instruction; only the declared visual tokens or
clue content differ. The scorer compares option letters by summed answer-token
negative log likelihood. Output includes every NLL, `softmax(-NLL)` pseudo
probability, semantic prediction text, gold margin, prompt length, original and
effective visual token counts, and complete trial metadata.

The runner checks projector input/output dimensions, projector checkpoint and
metadata hashes, the current full trial root/count against the late-bound
matrix closure, projector/feature
encoder-pipeline identity, the resolved LLM/tokenizer identity, unique
content-bound trial IDs, and feature availability. It authenticates every
referenced feature file and tensor before use. Visual trials require both
`--feature-index` and `--feature-metadata`. Text is never silently truncated.
Visual truncation is explicit, logged, and controlled by
`model.overflow_policy`; set it to `error` in the protocol *before freezing* for
a strict no-truncation run.

The score command requires all three frozen flags in `protocol.yaml` to be
`true`, a pinned `model.llm_revision`, and a fully populated projector section.
Command-line model values, when supplied, must equal the lock. A global
signature binds the protocol/release/build attestation, full matrix closure,
model and projector,
encoder pipeline, media-hash policy, dtype, token limit, and overflow policy.
A run signature additionally binds the current JSONL representation/trial-set,
feature-index/metadata/artifact root, and execution subset. Every prediction
row carries both signature digests plus `result_content_sha256`, which covers
its trial design, NLLs, derived predictions/probabilities/margins, token counts,
and scoring provenance. The adjacent `<predictions>.metadata.json` contains the
canonical signature payloads, exact trial-set identity, status, and failure
count. Resume authenticates the sidecar and every durable result row before it
skips work.

Predictions must be written to plain `.jsonl`: every completed row is flushed
and `fsync`'d so `--resume` is durable, and `.gz` output is rejected. Compress a
completed prediction file afterward if desired; analysis accepts gzip. A run
using `--continue-on-error` produces an incomplete sidecar and is rejected by
strict analysis. Preserve feature/result artifacts read-only after scoring;
digest verification detects modification but cannot prevent deletion.

### 8. Compute paired results

```bash
python -m information_upper_bound analyze \
  --expected-trials data/information_upper_bound/official_trials.jsonl.gz \
  --predictions outputs/information_upper_bound/official/internvideo2-clip-s/predictions.jsonl \
  --score-metadata outputs/information_upper_bound/official/internvideo2-clip-s/predictions.jsonl.metadata.json \
  --out-dir outputs/information_upper_bound/official/internvideo2-clip-s/analysis \
  --config information_upper_bound/configs/protocol.yaml
```

Confirmatory analysis is strict by default. A successful run requires a
content-equivalent complete expanded trial manifest, the locked protocol, and
the completed score metadata
sidecar. For a single score file the sidecar defaults to
`<predictions>.metadata.json`; pass `--score-metadata` explicitly as above for
clarity. When analyzing a deliberately concatenated multi-run prediction file,
repeat `--score-metadata` once per constituent run. All sidecars must share one
authenticated global signature, and their disjoint trial-set identities must
exactly cover the supplied trial manifest. The supplied expected manifest's
root/count must first equal the independent closure pinned in
`protocol.projector`; a self-consistent subset cannot redefine completeness.

The analysis produces:

- condition summaries with `row_micro_accuracy`, `cluster_macro_accuracy`,
  `cluster_all_rows_correct`, gold margin, option pseudo-probability metrics,
  independent-cluster count, and cluster-bootstrap confidence intervals;
- pre-registered `left - right` confirmatory contrasts from `protocol.yaml`, plus
  labeled exploratory contrasts, on aligned rows from the exact shared item
  intersection (requested dose is matched when both sides have dose grids);
- original/counterfactual both-correct and correct semantic-flip rates;
- original/nuisance both-correct semantic invariance;
- semantic option-permutation consistency and all-permutations-correct rate;
- evidence sufficiency, comprehensiveness, same-grid random-mask placebo, and
  random-span controls;
- sham-adjusted timestamp and reasoning-operator effects, alongside their
  format-only placebo contrasts;
- dose curves and K90, the smallest clue dose reaching 90% of the maximum
  observed channel-appropriate gain (`question_only` for text/embedding oracles,
  `full_video` for video-plus-text);
- explicit missing, duplicate, malformed, metadata-mismatched, and
  per-comparison coverage. Missing rows are not silently interpreted as either
  success or failure.

`row_micro_accuracy` gives every valid scored row equal weight.
`cluster_macro_accuracy` first averages rows within each independent cluster and
then weights clusters equally; the historical `accuracy` column is an alias for
this cluster-macro estimand, not row-micro accuracy.
`cluster_all_rows_correct` is stricter: every observed row in the cluster must
have valid correctness and be correct. For CLEVRER, the primary metric is
`official_question_exact_set_accuracy`. For each official candidate, the
analyzer normalizes semantic option text, maps probability mass back from every
authenticated option permutation, averages that semantic probability, and
requires a unique maximum. One official question is correct only when every
candidate in its authenticated complete set is then classified correctly.
`official_question_permutation_robustness_accuracy` retains the stronger
every-candidate/every-permutation requirement as a separate diagnostic.
Candidate-row accuracy and cluster-macro accuracy also remain diagnostics
because the expanded binary labels are dominated by negative candidates; they
are not the CLEVRER primary estimand. Both official-question metrics and the
paired exact-set gain use scene-level `resampling_unit_id` bootstrap intervals.

Point-estimate aggregation uses `pair_id` for paired interventions, then
`independent_unit_id` for multi-row official questions such as CLEVRER, and
finally `base_id`. Bootstrap resampling instead starts from the higher-level
`resampling_unit_id`: one raw video, CLEVRER scene, or paired-video family is
one dependence cluster. Option permutations, candidate rows, multiple
questions from one source, and multiple conditions never count as independent
bootstrap samples. `cluster_all_rows_correct` still operates at the official
question/pair aggregation unit and requires every candidate/permutation row of
that unit to be correct.
Compare encoders on the same locked trial intersection, not on each encoder's
individually available subset.

Strict completeness is automatic; `--require-complete` remains only as a
deprecated compatibility flag. The command writes the diagnostic report and
exits non-zero when the trial manifest is absent/incomplete, a score run is not
complete, sidecar authentication fails, or any trial/result binding is
malformed. It recomputes trial and result hashes and derives probabilities,
prediction, gold/best-distractor NLL, margin, and correctness from the
authenticated `choice_nll` values instead of trusting editable convenience
fields. It also checks choices/labels/semantic prediction text, finite values,
release/build identity, exact trial-set coverage, and global/run signatures.
Projected-visual rows additionally require positive token counts, no visual
token truncation, and a consistent token budget for each `visual_id`.
Comparison-specific intersection sizes are still reported because evidence and
safe-oracle conditions are legitimately unavailable for some items.

The default protocol requests 10,000 ordinary cluster-bootstrap resamples. The
implementation processes replicate counts in deterministic NumPy batches using
per-cluster sufficient statistics; it does not materialize expanded rows for
every replicate. Both expected trials and predictions may be plain JSONL or
gzip. Analysis itself loads the joined rows in memory and writes
`summary.csv`, `comparisons.csv`, `pair_metrics.csv`, `dose_curves.csv`, and
`report.json` atomically. Existing analysis artifacts are refused by default;
pass `--overwrite` only for an intentional rerun. Output paths may never alias
predictions, the expected manifest, score metadata, or the protocol.

In strict mode, analysis values and comparison order come from the frozen
protocol; a conflicting CLI analysis value is rejected. `CLI > config >
defaults` precedence is available only with `--development`, whose
`report.json` is watermarked `DEVELOPMENT RESULT` and is never confirmatory.
Confirmatory pairs are always reported in the configured `left - right`
direction.

K90 is conditional on the pre-registered, answer-independent clue ordering in
`conditions.yaml`; it is an operational dose threshold, not a proof that no
smaller or differently phrased clue set exists.

## Known boundaries

- The suite does not download benchmark media, model weights, or projector
  checkpoints; none are bundled. It also ships no benchmark predictions or
  score. A real result requires each official release/sidecar, actual videos, a
  compatible frozen encoder/projector pair, and the exact frozen LLM revisions
  named by the locked artifacts.
- Evidence and oracle conditions exist only where the official release or an
  explicitly supplied official sidecar supports them. Missing evidence or safe
  facts skips that condition and dose with an aggregated reason plus a bounded
  preview in the build report; the code does not synthesize labels, timestamps,
  event facts, or rationales.
- Text and embedding oracles measure annotation-conditioned **clue usability by
  the frozen LLM**, not realizability by the visual encoder/projector interface.
  Text clues enter the prompt and embedding clues bypass the projector. They are
  sensitive to annotation completeness, wording, tokenization, and clue order;
  they are neither an information-theoretic optimum nor proof that a trainable
  encoder can attain the gain. A projected-oracle realizability experiment would
  require a separately trained, held-out visual-token mapper and is intentionally
  not implemented here.
- `--limit`, `--allow-missing-media`, `--allow-missing-grounding`,
  `--allow-missing-cut-mapping`, `--allow-incomplete-diagnostic`,
  `--allow-validation-errors`, and `--continue-on-error` are debugging/audit
  affordances. Adapter reports expose these choices and release locking or
  strict analysis rejects the resulting incomplete artifacts. Do not report a
  run using them as a complete benchmark result.
- Trial writing, matrix authentication, extraction input, and scoring are
  streaming. Matrix authentication spills trial identities and reconstructed
  bases to temporary SQLite storage; base-manifest validation, feature-index
  assembly, and statistical analysis still retain substantial state in memory.
  The current confirmatory scorer requires one full trial manifest
  (`trial_shards: 1`), so provision storage and temporary disk accordingly.
- Release creation/extraction authenticate video bytes; feature extraction and
  scoring authenticate decoded frames, tensor/file identities, and resolved
  pretrained content; scoring/analyze authenticate result rows and sidecars.
  These checks detect substitution or mutation, but cannot recover a deleted
  artifact. Preserve the complete release, protocol, model snapshots, feature
  cache, predictions, and sidecars read-only.

## Question families and recommended data

Use `information_family` for the required visual information and
`reasoning_depth` for the operator applied to it:

| Information family | Example question | Best-supported sources |
|---|---|---|
| `static` | What color/object/location is visible? | Perception Test, NExT-GQA controls |
| `local_motion` | Is the drawer opening or closing? | TempCompass, TVBench, MVP |
| `temporal_order` | Did A occur before B? | TempCompass, TVBench, NExT-GQA |
| `metric_temporal` | Which lasted longer / how many times / who moved faster? | TempCompass, TVBench, Perception Test |
| `binding_tracking` | What did the person holding X do next? | Perception Test, NExT-GQA, CLEVRER |
| `causal_compositional` | What caused the collision / what if X were absent? | CLEVRER, NExT-GQA |
| `long_range_selection` | Which early event explains the final outcome? | EgoSchema, NExT-GQA |

Within each family, use depth `0` for direct facts, `1` for one relation, `2`
for binding/composition, and `3` for aggregation, causal, predictive, or
counterfactual reasoning. Do not use a dataset name as a question taxonomy.

For a compact but identifiable core, combine TempCompass (short temporal
conflicts), CLEVRER (symbolic causal controls), NExT-GQA (gold evidence
ablation), and EgoSchema (long context). Add Perception Test when tracking and
binding are central. A pilot should contain at least 100 independent families
per information type; estimate the confirmatory sample size from pilot paired
discordance with a McNemar power analysis. The protocol does not impose a
generic “300 families per type” target; dataset-specific record/evidence/oracle
minimums in `coverage_contract` are scope guards, not a power calculation.

## Leakage and reporting rules

1. Keep every original, counterfactual, nuisance, paraphrase, transformed clip,
   and question from one source family in the same split.
2. Group all clips from one raw source video in the same split, even if their
   filenames differ.
3. Create clues from visual/event annotations without exposing the answer or
   choices to the clue generator. Keep target-semantic metadata inaccessible.
4. Lock item IDs, prompts, conditions, clue ordering, frame budget, seed, and
   analysis before scoring or inspecting final model outputs.
5. Use the same frozen LLM, projector architecture/checkpoint policy, scoring
   prompt, token budget, and trial intersection for every encoder.
6. Report row-micro accuracy, cluster-macro accuracy, all-rows-correct, and
   paired-family metrics with their units named explicitly. The paired metrics
   are the main evidence for minimal visual changes and nuisance invariance.
7. Report each dataset and information family separately before any macro
   average. A gain on CLEVRER does not establish a gain on real long-form video.
8. Require the same underlying safe facts for temporal/reasoning contrasts, and
   interpret the sham-adjusted effects (`ordered_oracle - ordered_timestamp_sham`
   and `reasoning_oracle - reasoning_operator_sham`) as the primary evidence for
   timestamp/operator information. Report the corresponding format placebos and
   the broader total effects against `atomic_oracle`/`ordered_oracle` together.

## Test the implementation

```bash
python -m unittest discover -s information_upper_bound/tests -p 'test_*.py' -v
python -m compileall -q information_upper_bound
```

The tests cover deterministic view selection, evidence/complement disjointness,
matched random controls, answer normalization, option remapping, pair invariants,
adapter reports, relocatable release locks, media/source mutation detection,
trial attestations, feature/result integrity, authenticated resume/analysis,
CLEVRER exact-set scoring, paired metrics, and bootstrap determinism. A passing
test suite verifies the experiment machinery; it is not a substitute for
running the official benchmark data and frozen model checkpoints.
