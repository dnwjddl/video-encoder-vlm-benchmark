from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from information_upper_bound.conditions import trial_content_sha256
from information_upper_bound.integrity import (
    canonical_sha256,
    scored_result_sha256,
    trial_set_identity,
)
from information_upper_bound.io import iter_jsonl, sha256_file, write_jsonl
from information_upper_bound.metrics import _authenticate_score_metadata
from information_upper_bound.run import (
    SCORING_PARTITION_ALGORITHM,
    SCORING_PARTITION_SCHEMA_VERSION,
    _score_worker_index,
    _validate_score_worker,
    parse_args,
    run_trials,
)
from information_upper_bound.scoring import ScoreResult


DATA_RELEASE_SHA256 = "d" * 64
TRIAL_BUILD_ATTESTATION_SHA256 = "e" * 64
ENCODER_PIPELINE_SHA256 = "3" * 64
FEATURE_INDEX_SHA256 = "4" * 64
FEATURE_ARTIFACT_ROOT_SHA256 = "5" * 64
LLM_PRETRAINED_IDENTITY_SHA256 = "6" * 64
LLM_ID = "test/frozen-llm"
LLM_REVISION = "7" * 40
TRIAL_MATRIX_CLOSURE_SHA256 = "8" * 64


class _FakeFeatureStore:
    def __init__(self, visual_ids: set[str]) -> None:
        self.paths = {visual_id: Path(f"{visual_id}.pt") for visual_id in visual_ids}
        self.index_sha256 = FEATURE_INDEX_SHA256
        self.artifact_root_sha256 = FEATURE_ARTIFACT_ROOT_SHA256

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, visual_id: str) -> torch.Tensor:
        if visual_id not in self.paths:
            raise KeyError(visual_id)
        return torch.ones((2, 4), dtype=torch.float32)


class _FakeScorer:
    def __init__(
        self,
        *,
        llm_id: str | None,
        llm_revision: str | None,
        device: str,
        **_kwargs: object,
    ) -> None:
        self.llm_id = str(llm_id)
        self.llm_revision = llm_revision
        self.pretrained_identity = {"identity_sha256": LLM_PRETRAINED_IDENTITY_SHA256}
        self.device = torch.device(device)

    def score(
        self, trial: dict[str, object], features: torch.Tensor | None
    ) -> ScoreResult:
        assert features is not None
        return ScoreResult(
            prediction="A",
            prediction_text=str((trial.get("choices") or [""])[0]),
            choice_nll={"A": 0.1, "B": 1.1},
            choice_probability={"A": 0.7310585786300049, "B": 0.2689414213699951},
            gold_nll=0.1,
            best_distractor_nll=1.1,
            gold_margin=1.0,
            correct=True,
            prompt_tokens=12,
            original_visual_tokens=2,
            effective_visual_tokens=2,
            token_source="projected_visual",
        )


class _InterruptAfterOneScorer(_FakeScorer):
    calls = 0

    def score(
        self, trial: dict[str, object], features: torch.Tensor | None
    ) -> ScoreResult:
        if type(self).calls >= 1:
            raise RuntimeError("simulated scorer interruption")
        type(self).calls += 1
        return super().score(trial, features)


def _trial(index: int) -> dict[str, object]:
    row: dict[str, object] = {
        "base_id": f"base-{index}",
        "data_release_sha256": DATA_RELEASE_SHA256,
        "trial_build_attestation": {
            "attestation_sha256": TRIAL_BUILD_ATTESTATION_SHA256
        },
        "media_type": "video",
        "question": f"Question {index}?",
        "choices": ["yes", "no"],
        "answer": "A",
        "answer_text": "yes",
        "clue_text": "",
        "visual_id": f"visual-{index}",
        "visual_spec": {"view": "full_video"},
        "condition": {
            "name": "full_video",
            "input_channel": "visual",
            "visual_view": "full_video",
            "requested_dose": 0,
            "effective_dose": 0,
            "permutation_index": 0,
            "seed": 42,
        },
        "diagnostic": {
            "dataset": "fixture",
            "split": "validation",
            "information_family": "temporal_order",
            "question_family": "fixture",
            "reasoning_depth": 1,
            "resampling_unit_id": f"video-{index}",
            "pair_id": f"standalone-{index}",
            "pair_role": "standalone",
            "independent_unit_id": f"question-{index}",
        },
    }
    content_sha256 = trial_content_sha256(row)
    row["trial_content_sha256"] = content_sha256
    row["trial_id"] = f"trial::{content_sha256}"
    return row


def _balanced_trials(
    *, worker_count: int, minimum_per_worker: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bucket_counts = [0] * worker_count
    index = 0
    while min(bucket_counts) < minimum_per_worker:
        row = _trial(index)
        content_sha256 = str(row["trial_content_sha256"])
        bucket = _score_worker_index(content_sha256, worker_count=worker_count)
        rows.append(row)
        bucket_counts[bucket] += 1
        index += 1
    return rows


def _fixture(root: Path) -> SimpleNamespace:
    trials = _balanced_trials(worker_count=4, minimum_per_worker=2)
    trials_path = root / "trials.jsonl"
    write_jsonl(trials_path, trials)
    full_trial_set = trial_set_identity(trials)

    checkpoint_path = root / "projector.pt"
    checkpoint_path.write_bytes(b"projector checkpoint")
    projector_metadata_path = root / "projector.metadata.json"
    projector_metadata = {
        "llm_id": LLM_ID,
        "llm_pretrained_identity_sha256": LLM_PRETRAINED_IDENTITY_SHA256,
        "encoder_name": "fixture-encoder",
        "encoder_extraction_pipeline_identity_sha256": ENCODER_PIPELINE_SHA256,
        "evaluation_manifest_sha256": sha256_file(trials_path),
    }
    projector_metadata_path.write_text(json.dumps(projector_metadata), encoding="utf-8")

    feature_index_path = root / "features.index.jsonl"
    feature_index_path.write_text("fixture\n", encoding="utf-8")
    feature_metadata_path = root / "features.metadata.json"
    feature_metadata = {
        "encoder": "fixture-encoder",
        "extraction_pipeline_identity_sha256": ENCODER_PIPELINE_SHA256,
        "data_release_sha256": DATA_RELEASE_SHA256,
        "trial_build_attestation_sha256": TRIAL_BUILD_ATTESTATION_SHA256,
        "media_sha256_enabled": False,
    }
    feature_metadata_path.write_text(json.dumps(feature_metadata), encoding="utf-8")

    locked_projector = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metadata_sha256": sha256_file(projector_metadata_path),
        "encoder_extraction_pipeline_identity_sha256": ENCODER_PIPELINE_SHA256,
        "llm_pretrained_identity_sha256": LLM_PRETRAINED_IDENTITY_SHA256,
        "evaluation_feature_index_sha256": FEATURE_INDEX_SHA256,
        "evaluation_feature_metadata_sha256": sha256_file(feature_metadata_path),
        "evaluation_feature_artifact_root_sha256": FEATURE_ARTIFACT_ROOT_SHA256,
        "evaluation_trial_matrix_closure_sha256": TRIAL_MATRIX_CLOSURE_SHA256,
        "evaluation_trial_set_root_sha256": full_trial_set["root_sha256"],
        "evaluation_trial_count": full_trial_set["trial_count"],
    }
    protocol = {
        "schema_version": "information_upper_bound.test_protocol.v1",
        "model": {
            "llm_id": LLM_ID,
            "llm_revision": LLM_REVISION,
            "llm_frozen": True,
            "visual_encoder_frozen": True,
            "projector_frozen_during_evaluation": True,
            "dtype": "bf16",
            "max_length": 4096,
            "overflow_policy": "error",
        },
        "projector": locked_projector,
        "sampling": {
            "seed": 42,
            "option_permutations": "all",
            "trial_shards": 1,
            "require_media_sha256": False,
        },
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    visual_ids = {str(row["visual_id"]) for row in trials}
    store = _FakeFeatureStore(visual_ids)
    common = {
        "trials_path": trials_path,
        "projector_checkpoint": checkpoint_path,
        "projector_metadata_path": projector_metadata_path,
        "feature_index_path": feature_index_path,
        "feature_metadata_path": feature_metadata_path,
        "llm_id": LLM_ID,
        "llm_revision": LLM_REVISION,
        "device": "cpu",
        "dtype": "bf16",
        "max_length": 4096,
        "overflow_policy": "error",
        "protocol_config_path": protocol_path,
    }
    return SimpleNamespace(
        trials=trials,
        trials_path=trials_path,
        full_trial_set=full_trial_set,
        locked_projector=locked_projector,
        protocol=protocol,
        protocol_path=protocol_path,
        store=store,
        common=common,
    )


@contextmanager
def _patched_score_runtime(
    fixture: SimpleNamespace, *, scorer_class: type[_FakeScorer] = _FakeScorer
):
    with (
        patch(
            "information_upper_bound.run.load_protocol",
            return_value=(
                fixture.protocol,
                {"sha256": sha256_file(fixture.protocol_path)},
            ),
        ),
        patch(
            "information_upper_bound.run.validate_data_protocol",
            return_value={"data_release_sha256": DATA_RELEASE_SHA256},
        ),
        patch(
            "information_upper_bound.run.validate_locked_projector_protocol",
            return_value=fixture.locked_projector,
        ),
        patch(
            "information_upper_bound.run.validate_trial_build_attestation",
            return_value={"attestation_sha256": TRIAL_BUILD_ATTESTATION_SHA256},
        ),
        patch("information_upper_bound.run.FeatureStore", return_value=fixture.store),
        patch("information_upper_bound.run._validate_evaluation_feature_lock") as lock,
        patch("information_upper_bound.run.FrozenMultipleChoiceScorer", scorer_class),
    ):
        yield lock


def _resign_partitioned_sidecar(
    sidecar: dict[str, object],
    prediction_rows: list[dict[str, object]],
    **partition_updates: object,
) -> None:
    old_run_sha256 = str(sidecar["run_signature_sha256"])
    run_signature = sidecar["run_signature"]
    assert isinstance(run_signature, dict)
    score_partition = run_signature["score_partition"]
    assert isinstance(score_partition, dict)
    score_partition.update(partition_updates)
    sidecar["score_partition"] = dict(score_partition)
    new_run_sha256 = canonical_sha256(run_signature)
    sidecar["run_signature_sha256"] = new_run_sha256
    for row in prediction_rows:
        if row.get("scoring_run_signature_sha256") != old_run_sha256:
            continue
        row["scoring_run_signature_sha256"] = new_run_sha256
        row["result_content_sha256"] = scored_result_sha256(row)


def _authenticate_fixture(
    *,
    fixture: SimpleNamespace,
    prediction_rows: list[dict[str, object]],
    metadata_paths: list[Path],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    return _authenticate_score_metadata(
        prediction_rows=prediction_rows,
        expected_rows=fixture.trials,
        expected_path=fixture.trials_path,
        metadata_paths=metadata_paths,
        protocol_sha256=sha256_file(fixture.protocol_path),
        data_release_sha256=DATA_RELEASE_SHA256,
        trial_build_attestation_sha256=TRIAL_BUILD_ATTESTATION_SHA256,
        locked_protocol=fixture.protocol,
    )


class ScoreWorkerPartitionTest(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_complete(self) -> None:
        digests = [
            hashlib.sha256(f"trial-{index}".encode()).hexdigest()
            for index in range(500)
        ]
        first = [
            {
                digest
                for digest in digests
                if _score_worker_index(digest, worker_count=4) == worker_index
            }
            for worker_index in range(4)
        ]
        second = [
            {
                digest
                for digest in reversed(digests)
                if _score_worker_index(digest, worker_count=4) == worker_index
            }
            for worker_index in range(4)
        ]
        self.assertEqual(first, second)
        self.assertEqual(set().union(*first), set(digests))
        self.assertEqual(sum(map(len, first)), len(digests))

    def test_worker_coordinates_are_strictly_validated(self) -> None:
        _validate_score_worker(worker_count=4, worker_index=3)
        for worker_count, worker_index in ((0, 0), (4, -1), (4, 4), (True, 0)):
            with self.subTest(worker_count=worker_count, worker_index=worker_index):
                with self.assertRaises(ValueError):
                    _validate_score_worker(
                        worker_count=worker_count, worker_index=worker_index
                    )

    def test_score_cli_exposes_worker_partition(self) -> None:
        required = [
            "--trials",
            "trials.jsonl",
            "--out",
            "predictions.jsonl",
            "--projector-ckpt",
            "projector.pt",
            "--projector-metadata",
            "projector.json",
        ]
        defaults = parse_args(required)
        self.assertEqual((defaults.worker_count, defaults.worker_index), (1, 0))
        partitioned = parse_args(
            [*required, "--worker-count", "4", "--worker-index", "3"]
        )
        self.assertEqual((partitioned.worker_count, partitioned.worker_index), (4, 3))

    def test_default_single_worker_keeps_legacy_signature_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            output_path = root / "predictions.jsonl"
            with _patched_score_runtime(fixture):
                report = run_trials(
                    **fixture.common,
                    output_path=output_path,
                )
            self.assertNotIn("score_partition", report)
            self.assertNotIn("score_partition", report["run_signature"])
            self.assertNotIn("num_trials_in_full_manifest", report)
            self.assertEqual(report["trial_set_identity"], fixture.full_trial_set)
            self.assertEqual(report["num_trials_requested"], len(fixture.trials))
            prediction_rows = list(iter_jsonl(output_path))
            self.assertEqual(len(prediction_rows), len(fixture.trials))
            authentication, issues = _authenticate_fixture(
                fixture=fixture,
                prediction_rows=prediction_rows,
                metadata_paths=[output_path.with_suffix(".jsonl.metadata.json")],
            )
            self.assertEqual(issues, [])
            self.assertTrue(authentication["authenticated"])
            self.assertFalse(authentication["authenticated_sharded_run"])
            self.assertFalse(authentication["partitioned_run"])

    def test_four_workers_emit_one_authenticated_sharded_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            output_paths: list[Path] = []
            metadata_paths: list[Path] = []
            reports: list[dict[str, object]] = []
            prediction_rows: list[dict[str, object]] = []

            with _patched_score_runtime(fixture) as evaluation_lock:
                for worker_index in range(4):
                    output_path = root / f"predictions.worker-{worker_index}-of-4.jsonl"
                    report = run_trials(
                        **fixture.common,
                        output_path=output_path,
                        worker_count=4,
                        worker_index=worker_index,
                    )
                    worker_rows = list(iter_jsonl(output_path))
                    output_paths.append(output_path)
                    metadata_paths.append(
                        output_path.with_suffix(".jsonl.metadata.json")
                    )
                    reports.append(report)
                    prediction_rows.extend(worker_rows)

                    expected_ids = {
                        str(row["trial_id"])
                        for row in fixture.trials
                        if _score_worker_index(
                            str(row["trial_content_sha256"]), worker_count=4
                        )
                        == worker_index
                    }
                    self.assertEqual(
                        {str(row["trial_id"]) for row in worker_rows}, expected_ids
                    )
                    self.assertEqual(
                        report["trial_set_identity"], trial_set_identity(worker_rows)
                    )
                    self.assertEqual(report["num_trials_requested"], len(worker_rows))
                    self.assertEqual(
                        report["num_trials_in_full_manifest"], len(fixture.trials)
                    )
                    self.assertEqual(
                        report["run_signature"]["trials_manifest_sha256"],
                        sha256_file(fixture.trials_path),
                    )
                    self.assertEqual(
                        report["score_partition"],
                        {
                            "schema_version": SCORING_PARTITION_SCHEMA_VERSION,
                            "algorithm": SCORING_PARTITION_ALGORITHM,
                            "worker_count": 4,
                            "worker_index": worker_index,
                        },
                    )

                self.assertEqual(evaluation_lock.call_count, 4)
                self.assertEqual(
                    {str(row["trial_id"]) for row in prediction_rows},
                    {str(row["trial_id"]) for row in fixture.trials},
                )
                self.assertEqual(len(prediction_rows), len(fixture.trials))
                self.assertEqual(
                    len({str(report["global_signature_sha256"]) for report in reports}),
                    1,
                )
                self.assertEqual(
                    len({str(report["run_signature_sha256"]) for report in reports}),
                    4,
                )
                self.assertTrue(
                    all(
                        report["global_signature"]["full_trial_set_root_sha256"]
                        == fixture.full_trial_set["root_sha256"]
                        and report["global_signature"]["full_trial_count"]
                        == fixture.full_trial_set["trial_count"]
                        for report in reports
                    )
                )

                authentication, issues = _authenticate_fixture(
                    fixture=fixture,
                    prediction_rows=prediction_rows,
                    metadata_paths=metadata_paths,
                )
                self.assertEqual(issues, [])
                self.assertTrue(authentication["authenticated"])
                self.assertTrue(authentication["authenticated_sharded_run"])
                self.assertTrue(authentication["partitioned_run"])
                self.assertEqual(authentication["score_partition_worker_count"], 4)
                self.assertEqual(
                    authentication["score_partition_worker_indices"], [0, 1, 2, 3]
                )
                self.assertEqual(authentication["sidecar_count"], 4)

                first_output = output_paths[0]
                first_count = reports[0]["num_trials_requested"]
                resumed = run_trials(
                    **fixture.common,
                    output_path=first_output,
                    worker_count=4,
                    worker_index=0,
                    resume=True,
                )
                self.assertEqual(resumed["num_written_this_run"], 0)
                self.assertEqual(resumed["num_skipped_completed"], first_count)
                with self.assertRaisesRegex(ValueError, "resume signature differs"):
                    run_trials(
                        **fixture.common,
                        output_path=first_output,
                        worker_count=4,
                        worker_index=1,
                        resume=True,
                    )

    def test_partition_metadata_authentication_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            metadata_paths: list[Path] = []
            prediction_rows: list[dict[str, object]] = []
            with _patched_score_runtime(fixture):
                for worker_index in range(4):
                    output_path = root / f"worker-{worker_index}.jsonl"
                    run_trials(
                        **fixture.common,
                        output_path=output_path,
                        worker_count=4,
                        worker_index=worker_index,
                    )
                    metadata_paths.append(
                        output_path.with_suffix(".jsonl.metadata.json")
                    )
                    prediction_rows.extend(iter_jsonl(output_path))

            original_sidecars = [
                json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths
            ]
            original_rows = json.loads(json.dumps(prediction_rows))

            def authenticate_case(
                sidecars: list[dict[str, object]],
                rows: list[dict[str, object]],
                paths: list[Path] | None = None,
            ) -> set[str]:
                selected_paths = metadata_paths if paths is None else paths
                for path, sidecar in zip(metadata_paths, sidecars, strict=True):
                    path.write_text(json.dumps(sidecar), encoding="utf-8")
                authentication, issues = _authenticate_fixture(
                    fixture=fixture,
                    prediction_rows=rows,
                    metadata_paths=selected_paths,
                )
                self.assertFalse(authentication["authenticated"])
                return {str(issue["kind"]) for issue in issues}

            with self.subTest("top-level descriptor must equal signed descriptor"):
                sidecars = json.loads(json.dumps(original_sidecars))
                rows = json.loads(json.dumps(original_rows))
                sidecars[0]["score_partition"]["worker_index"] = 1
                self.assertIn(
                    "score_metadata_partition_top_level_mismatch",
                    authenticate_case(sidecars, rows),
                )

            for field in ("schema_version", "algorithm"):
                with self.subTest(field=field):
                    sidecars = json.loads(json.dumps(original_sidecars))
                    rows = json.loads(json.dumps(original_rows))
                    _resign_partitioned_sidecar(
                        sidecars[0], rows, **{field: f"invalid-{field}"}
                    )
                    self.assertIn(
                        "score_metadata_invalid_partition",
                        authenticate_case(sidecars, rows),
                    )

            with self.subTest("worker counts must match"):
                sidecars = json.loads(json.dumps(original_sidecars))
                rows = json.loads(json.dumps(original_rows))
                _resign_partitioned_sidecar(sidecars[3], rows, worker_count=5)
                self.assertIn(
                    "score_metadata_partition_worker_count_mismatch",
                    authenticate_case(sidecars, rows),
                )

            with self.subTest("worker indices must be unique"):
                sidecars = json.loads(json.dumps(original_sidecars))
                rows = json.loads(json.dumps(original_rows))
                _resign_partitioned_sidecar(sidecars[1], rows, worker_index=0)
                issue_kinds = authenticate_case(sidecars, rows)
                self.assertIn(
                    "score_metadata_duplicate_partition_worker_index", issue_kinds
                )
                self.assertIn(
                    "score_metadata_incomplete_partition_worker_indices", issue_kinds
                )

            with self.subTest("worker index set must be exact"):
                sidecars = json.loads(json.dumps(original_sidecars))
                rows = [
                    row
                    for row in json.loads(json.dumps(original_rows))
                    if row["scoring_run_signature_sha256"]
                    != original_sidecars[3]["run_signature_sha256"]
                ]
                self.assertIn(
                    "score_metadata_incomplete_partition_worker_indices",
                    authenticate_case(sidecars, rows, metadata_paths[:3]),
                )

            with self.subTest("partition metadata cannot mix with legacy metadata"):
                sidecars = json.loads(json.dumps(original_sidecars))
                rows = json.loads(json.dumps(original_rows))
                old_run_sha256 = str(sidecars[3]["run_signature_sha256"])
                del sidecars[3]["run_signature"]["score_partition"]
                del sidecars[3]["score_partition"]
                new_run_sha256 = canonical_sha256(sidecars[3]["run_signature"])
                sidecars[3]["run_signature_sha256"] = new_run_sha256
                for row in rows:
                    if row["scoring_run_signature_sha256"] == old_run_sha256:
                        row["scoring_run_signature_sha256"] = new_run_sha256
                        row["result_content_sha256"] = scored_result_sha256(row)
                self.assertIn(
                    "score_metadata_mixed_partition_presence",
                    authenticate_case(sidecars, rows),
                )

            with self.subTest("prediction digest must belong to signed worker"):
                sidecars = json.loads(json.dumps(original_sidecars))
                rows = json.loads(json.dumps(original_rows))
                target = rows[0]
                target_run_sha256 = str(target["scoring_run_signature_sha256"])
                target_sidecar = next(
                    sidecar
                    for sidecar in sidecars
                    if sidecar["run_signature_sha256"] == target_run_sha256
                )
                target_worker_index = int(
                    target_sidecar["score_partition"]["worker_index"]
                )
                replacement_index = 0
                while True:
                    replacement = hashlib.sha256(
                        f"wrong-owner-{replacement_index}".encode()
                    ).hexdigest()
                    if (
                        _score_worker_index(replacement, worker_count=4)
                        != target_worker_index
                    ):
                        break
                    replacement_index += 1
                target["trial_content_sha256"] = replacement
                target["result_content_sha256"] = scored_result_sha256(target)
                self.assertIn(
                    "prediction_partition_owner_mismatch",
                    authenticate_case(sidecars, rows),
                )

    def test_resume_recovers_partial_partition_and_running_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            output_path = root / "worker-0.jsonl"
            metadata_path = output_path.with_suffix(".jsonl.metadata.json")
            _InterruptAfterOneScorer.calls = 0
            with _patched_score_runtime(fixture, scorer_class=_InterruptAfterOneScorer):
                with self.assertRaisesRegex(
                    RuntimeError, "simulated scorer interruption"
                ):
                    run_trials(
                        **fixture.common,
                        output_path=output_path,
                        worker_count=4,
                        worker_index=0,
                    )

            partial_rows = list(iter_jsonl(output_path))
            running_sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(len(partial_rows), 1)
            self.assertEqual(running_sidecar["status"], "running")
            self.assertEqual(running_sidecar["num_completed_before_run"], 0)
            self.assertEqual(running_sidecar["score_partition"]["worker_index"], 0)
            self.assertGreater(running_sidecar["num_trials_requested"], 1)

            with _patched_score_runtime(fixture):
                resumed = run_trials(
                    **fixture.common,
                    output_path=output_path,
                    worker_count=4,
                    worker_index=0,
                    resume=True,
                )
            final_rows = list(iter_jsonl(output_path))
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["num_skipped_completed"], 1)
            self.assertEqual(
                resumed["num_written_this_run"],
                resumed["num_trials_requested"] - 1,
            )
            self.assertEqual(len(final_rows), resumed["num_trials_requested"])
            self.assertEqual(
                len({str(row["trial_id"]) for row in final_rows}), len(final_rows)
            )

    def test_non_owned_trial_is_still_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            non_owned = next(
                row
                for row in fixture.trials
                if _score_worker_index(str(row["trial_content_sha256"]), worker_count=4)
                != 0
            )
            non_owned["trial_content_sha256"] = "0" * 64
            write_jsonl(fixture.trials_path, fixture.trials)
            with _patched_score_runtime(fixture):
                with self.assertRaisesRegex(ValueError, "stale/invalid"):
                    run_trials(
                        **fixture.common,
                        output_path=root / "worker-0.jsonl",
                        worker_count=4,
                        worker_index=0,
                    )

    def test_projector_closure_is_checked_against_full_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            worker_zero_rows = [
                row
                for row in fixture.trials
                if _score_worker_index(str(row["trial_content_sha256"]), worker_count=4)
                == 0
            ]
            wrong_lock = {
                **fixture.locked_projector,
                "evaluation_trial_set_root_sha256": trial_set_identity(
                    worker_zero_rows
                )["root_sha256"],
                "evaluation_trial_count": len(worker_zero_rows),
            }
            with (
                _patched_score_runtime(fixture),
                patch(
                    "information_upper_bound.run.validate_locked_projector_protocol",
                    return_value=wrong_lock,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "complete evaluation matrix"):
                    run_trials(
                        **fixture.common,
                        output_path=root / "worker-0.jsonl",
                        worker_count=4,
                        worker_index=0,
                    )


if __name__ == "__main__":
    unittest.main()
