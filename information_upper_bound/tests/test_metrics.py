from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

from information_upper_bound.conditions import (
    trial_content_sha256 as compute_trial_content_sha256,
)
from information_upper_bound.metrics import (
    _bootstrap_count_batches,
    _bootstrap_grouped_stat_map,
    _bootstrap_paired_stat_map,
    _bootstrap_stat_map,
    _bootstrap_summary_stat_map,
    _paired_metrics_from_units,
    _require_complete_failures,
    _summary_metrics_from_units,
    analyze_predictions,
    bootstrap_mean_ci,
    calibration_metrics,
    load_protocol_config,
    main,
)
from information_upper_bound.integrity import (
    RESULT_INTEGRITY_SCHEMA_VERSION,
    canonical_sha256,
    scored_result_sha256,
    trial_set_identity,
)
from information_upper_bound.io import sha256_file
from information_upper_bound.protocol import trial_build_protocol_sha256
from information_upper_bound.scoring import SCORING_PROTOCOL_VERSION


TEST_DATA_RELEASE_SHA256 = "d" * 64
TEST_PROJECTOR_CHECKPOINT_SHA256 = "1" * 64
TEST_PROJECTOR_METADATA_SHA256 = "2" * 64
TEST_ENCODER_PIPELINE_SHA256 = "3" * 64
TEST_LLM_IDENTITY_SHA256 = "4" * 64


def prediction_row(
    trial_id: str,
    base_id: str,
    condition: str,
    *,
    correct: bool,
    requested_dose: int | str = 0,
    effective_dose: int = 0,
    permutation_index: int = 0,
    pair_id: str | None = None,
    pair_role: str = "standalone",
    resampling_unit_id: str | None = None,
    input_channel: str | None = None,
    prediction: str | None = None,
    prediction_text: str | None = None,
    answer: str = "A",
    answer_text: str = "yes",
    gold_margin: float | None = None,
    choices: list[str] | None = None,
    trial_content_sha256: str | None = None,
    scoring_run_signature_sha256: str = "a" * 64,
    scoring_global_signature_sha256: str = "c" * 64,
) -> dict[str, object]:
    predicted = prediction or (answer if correct else ("B" if answer != "B" else "A"))
    if input_channel is None:
        if condition == "question_only":
            input_channel = "question_only"
        elif "video_plus" in condition:
            input_channel = "visual_plus_text"
        elif "oracle" in condition:
            input_channel = "text_oracle"
        else:
            input_channel = "visual"
    resolved_choices = list(choices) if choices is not None else ["other", "other"]
    gold_index = ord(answer) - ord("A")
    resolved_choices[gold_index] = answer_text
    for index, value in enumerate(resolved_choices):
        if index != gold_index and value == "other":
            resolved_choices[index] = "no" if answer_text.casefold() != "no" else "yes"
    probability = {"A": 0.15, "B": 0.15}
    probability[predicted] = 0.85
    if predicted == "A":
        probability["B"] = 0.15
    else:
        probability["A"] = 0.15
    row: dict[str, object] = {
        "trial_id": trial_id,
        "data_release_sha256": TEST_DATA_RELEASE_SHA256,
        "base_id": base_id,
        "visual_id": f"visual::{base_id}::{condition}",
        "dataset": "diagnostic",
        "information_family": "temporal_order",
        "question_family": "before_after",
        "reasoning_depth": 1,
        "resampling_unit_id": resampling_unit_id
        or (
            f"pair-family::{pair_id}"
            if pair_id is not None
            and pair_role in {"original", "counterfactual", "nuisance"}
            else f"video::{base_id}"
        ),
        "pair_id": pair_id or f"standalone::{base_id}",
        "pair_role": pair_role,
        "condition": condition,
        "input_channel": input_channel,
        "requested_dose": requested_dose,
        "effective_dose": effective_dose,
        "permutation_index": permutation_index,
        "choices": resolved_choices,
        "prediction": predicted,
        "prediction_text": prediction_text
        or (
            resolved_choices[ord(predicted) - ord("A")]
            if predicted in {"A", "B"}
            else "<invalid>"
        ),
        "answer": answer,
        "answer_text": answer_text,
        "correct": correct,
        "choice_probability": probability,
        "choice_nll": {label: -math.log(value) for label, value in probability.items()},
        "scoring_run_signature_sha256": scoring_run_signature_sha256,
        "scoring_global_signature_sha256": scoring_global_signature_sha256,
    }
    choice_nll = row["choice_nll"]
    assert isinstance(choice_nll, dict)
    row["gold_nll"] = choice_nll[answer]
    row["best_distractor_nll"] = min(
        value for label, value in choice_nll.items() if label != answer
    )
    derived_margin = row["best_distractor_nll"] - row["gold_nll"]
    row["gold_margin"] = derived_margin if gold_margin is None else gold_margin
    row["trial_content_sha256"] = trial_content_sha256 or compute_trial_content_sha256(
        row
    )
    row["result_content_sha256"] = scored_result_sha256(row)
    return row


def manifest_row(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "prediction",
            "prediction_text",
            "correct",
            "choice_probability",
            "gold_margin",
            "choice_nll",
            "gold_nll",
            "best_distractor_nll",
            "scoring_global_signature_sha256",
            "scoring_run_signature_sha256",
            "result_content_sha256",
        }
    }


def set_choice_probabilities(
    row: dict[str, object], probabilities: dict[str, float]
) -> None:
    """Set an internally authenticated score vector for semantic MCQ tests."""

    choices = row["choices"]
    assert isinstance(choices, list)
    answer = str(row["answer"])
    prediction = max(sorted(probabilities), key=probabilities.__getitem__)
    choice_nll = {
        label: -math.log(probability) for label, probability in probabilities.items()
    }
    row.update(
        {
            "prediction": prediction,
            "prediction_text": choices[ord(prediction) - ord("A")],
            "correct": prediction == answer,
            "choice_probability": probabilities,
            "choice_nll": choice_nll,
            "gold_nll": choice_nll[answer],
            "best_distractor_nll": min(
                value for label, value in choice_nll.items() if label != answer
            ),
        }
    )
    row["gold_margin"] = float(row["best_distractor_nll"]) - float(row["gold_nll"])
    row["trial_content_sha256"] = compute_trial_content_sha256(row)
    row["result_content_sha256"] = scored_result_sha256(row)


def write_locked_analysis_protocol(
    path: Path, *, bootstrap_replicates: int = 10
) -> str:
    payload = {
        "schema_version": "test-1",
        "data": {
            "data_release_sha256": TEST_DATA_RELEASE_SHA256,
            "conditions_sha256": "9" * 64,
            "required_datasets": ["diagnostic"],
            "coverage_contract": {
                "schema_version": "information_upper_bound.coverage_contract.v1",
                "datasets": {
                    "diagnostic": {
                        "required_adapter_runs": 1,
                        "required_adapter_run_ids": ["adapter-run::" + "a" * 64],
                        "required_splits": ["eval"],
                        "required_information_families": ["temporal_order"],
                        "required_question_families": [],
                        "required_source_roles": ["annotations"],
                        "minimum_records": 0,
                        "minimum_records_with_evidence": 0,
                        "minimum_records_with_safe_oracles": 0,
                    }
                },
            },
        },
        "dataset_roles": {
            "diagnostic": {
                "information_families": ["temporal_order"],
                "primary_use": "unit-test integrity fixture",
            }
        },
        "model": {
            "llm_id": "test/frozen-llm",
            "llm_revision": "a" * 40,
            "llm_frozen": True,
            "visual_encoder_frozen": True,
            "projector_frozen_during_evaluation": True,
            "dtype": "bf16",
            "max_length": 4096,
            "overflow_policy": "error",
        },
        "projector": {
            "checkpoint_sha256": TEST_PROJECTOR_CHECKPOINT_SHA256,
            "metadata_sha256": TEST_PROJECTOR_METADATA_SHA256,
            "training_manifest_sha256": "5" * 64,
            "evaluation_manifest_sha256": "6" * 64,
            "training_feature_index_sha256": "7" * 64,
            "training_feature_metadata_sha256": "8" * 64,
            "training_feature_artifact_root_sha256": "9" * 64,
            "evaluation_feature_index_sha256": "a" * 64,
            "evaluation_feature_metadata_sha256": "b" * 64,
            "evaluation_feature_artifact_root_sha256": "c" * 64,
            "evaluation_trial_matrix_closure_sha256": "d" * 64,
            "evaluation_trial_set_root_sha256": "e" * 64,
            "evaluation_trial_count": 1,
            "encoder_extraction_pipeline_identity_sha256": TEST_ENCODER_PIPELINE_SHA256,
            "llm_pretrained_identity_sha256": TEST_LLM_IDENTITY_SHA256,
            "training_dtype": "bf16",
            "training_max_length": 4096,
            "training_seed": 42,
        },
        "sampling": {
            "require_media_sha256": False,
            "seed": 42,
            "option_permutations": "all",
            "trial_shards": 1,
        },
        "analysis": {
            "bootstrap_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
            "seed": 42,
            "reference_condition": "full_video",
            "ece_bins": 10,
            "minimum_confirmatory_resampling_units": 1,
        },
        "confirmatory_comparisons": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return sha256_file(path)


def write_score_sidecar(
    *,
    path: Path,
    predictions: list[dict[str, object]],
    manifest: list[dict[str, object]],
    manifest_path: Path,
    protocol_path: Path,
    full_manifest: list[dict[str, object]] | None = None,
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    attestation_payload = {
        "schema_version": "information_upper_bound.trial_build_attestation.v2",
        "mode": "confirmatory",
        "data_release_sha256": TEST_DATA_RELEASE_SHA256,
        "condition_config_sha256": "9" * 64,
        "trial_build_protocol_sha256": trial_build_protocol_sha256(protocol),
        "sampling": {
            "seed": 42,
            "option_permutations": "all",
            "trial_shards": 1,
        },
    }
    attestation = {
        **attestation_payload,
        "attestation_sha256": canonical_sha256(attestation_payload),
    }
    locked_manifest = manifest if full_manifest is None else full_manifest
    prediction_by_id = {str(row["trial_id"]): row for row in predictions}
    authenticated_expected: set[int] = set()
    for expected_rows in (locked_manifest, manifest):
        for expected in expected_rows:
            object_identity = id(expected)
            if object_identity in authenticated_expected:
                continue
            authenticated_expected.add(object_identity)
            expected["data_release_sha256"] = TEST_DATA_RELEASE_SHA256
            expected["trial_build_attestation"] = attestation
            expected["trial_content_sha256"] = compute_trial_content_sha256(expected)
            prediction = prediction_by_id.get(str(expected["trial_id"]))
            if prediction is not None:
                prediction["data_release_sha256"] = TEST_DATA_RELEASE_SHA256
                prediction["trial_build_attestation_sha256"] = attestation[
                    "attestation_sha256"
                ]
                prediction["trial_content_sha256"] = expected["trial_content_sha256"]

    full_trial_set = trial_set_identity(locked_manifest)
    closure_payload = {
        "schema_version": "information_upper_bound.test_trial_matrix_closure.v1",
        "status": "exact",
        "data_release_sha256": TEST_DATA_RELEASE_SHA256,
        "trial_build_attestation_sha256": attestation["attestation_sha256"],
        "trial_count": full_trial_set["trial_count"],
        "trial_set_root_sha256": full_trial_set["root_sha256"],
    }
    closure_sha256 = canonical_sha256(closure_payload)
    protocol["projector"].update(
        {
            "evaluation_trial_matrix_closure_sha256": closure_sha256,
            "evaluation_trial_set_root_sha256": full_trial_set["root_sha256"],
            "evaluation_trial_count": full_trial_set["trial_count"],
        }
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha256 = sha256_file(protocol_path)
    if manifest_path.suffix.casefold() == ".gz":
        with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(row) + "\n" for row in manifest))
    else:
        manifest_path.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
        )
    global_signature = {
        "schema_version": "information_upper_bound.scoring_global_signature.v2",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "protocol_config_sha256": protocol_sha256,
        "data_release_sha256": TEST_DATA_RELEASE_SHA256,
        "trial_build_attestation_sha256": attestation["attestation_sha256"],
        "trial_matrix_closure_sha256": closure_sha256,
        "full_trial_set_root_sha256": full_trial_set["root_sha256"],
        "full_trial_count": full_trial_set["trial_count"],
        "projector_checkpoint_sha256": TEST_PROJECTOR_CHECKPOINT_SHA256,
        "projector_metadata_sha256": TEST_PROJECTOR_METADATA_SHA256,
        "encoder_extraction_pipeline_identity_sha256": TEST_ENCODER_PIPELINE_SHA256,
        "media_sha256_required": False,
        "llm_id": "test/frozen-llm",
        "llm_revision_requested": "a" * 40,
        "llm_pretrained_identity": {
            "identity_sha256": TEST_LLM_IDENTITY_SHA256,
        },
        "dtype": "bf16",
        "max_length": 4096,
        "overflow_policy": "error",
    }
    global_sha256 = canonical_sha256(global_signature)
    trial_set = trial_set_identity(manifest)
    run_signature = {
        "schema_version": "information_upper_bound.scoring_run_signature.v2",
        **{
            field: global_signature[field]
            for field in (
                "scoring_protocol_version",
                "protocol_config_sha256",
                "projector_checkpoint_sha256",
                "projector_metadata_sha256",
                "llm_id",
                "llm_revision_requested",
                "llm_pretrained_identity",
                "dtype",
                "max_length",
                "overflow_policy",
                "trial_matrix_closure_sha256",
                "full_trial_set_root_sha256",
                "full_trial_count",
            )
        },
        "scoring_global_signature_sha256": global_sha256,
        "data_release_sha256": TEST_DATA_RELEASE_SHA256,
        "trial_build_attestation_sha256": attestation["attestation_sha256"],
        "trials_manifest_sha256": sha256_file(manifest_path),
        "trial_set_identity": trial_set,
        "feature_index_sha256": "a" * 64,
        "feature_metadata_sha256": "b" * 64,
        "feature_artifact_root_sha256": "c" * 64,
    }
    run_sha256 = canonical_sha256(run_signature)
    for row in predictions:
        row["scoring_global_signature_sha256"] = global_sha256
        row["scoring_run_signature_sha256"] = run_sha256
        row["result_content_sha256"] = scored_result_sha256(row)
    sidecar = {
        "status": "complete",
        "result_integrity_schema_version": RESULT_INTEGRITY_SCHEMA_VERSION,
        "global_signature": global_signature,
        "global_signature_sha256": global_sha256,
        "run_signature": run_signature,
        "run_signature_sha256": run_sha256,
        "trial_set_identity": trial_set,
        "trials_manifest_sha256": sha256_file(manifest_path),
        "num_trials_requested": len(manifest),
        "num_failures": 0,
    }
    path.write_text(json.dumps(sidecar), encoding="utf-8")


class CalibrationFormulaTest(unittest.TestCase):
    def test_multiclass_brier_and_ece(self) -> None:
        rows = [
            {
                "answer": "A",
                "prediction": "A",
                "correct": True,
                "choice_probability": {"A": 0.8, "B": 0.2},
            },
            {
                "answer": "A",
                "prediction": "B",
                "correct": False,
                "choice_probability": {"A": 0.3, "B": 0.7},
            },
        ]
        metrics = calibration_metrics(rows, ece_bins=2)
        # (0.8-1)^2 + 0.2^2 = .08; (0.3-1)^2 + .7^2 = .98
        self.assertAlmostEqual(metrics["brier"], 0.53)
        # Both confidences occupy the upper bin: |accuracy .5 - confidence .75|.
        self.assertAlmostEqual(metrics["ece"], 0.25)
        self.assertAlmostEqual(metrics["gold_probability"], 0.55)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_mean_ci(
            [0.0, 0.0, 0.0, 1.0],
            ["base-a", "base-a", "base-a", "base-b"],
            seed=17,
            bootstrap_replicates=400,
        )
        second = bootstrap_mean_ci(
            [0.0, 0.0, 0.0, 1.0],
            ["base-a", "base-a", "base-a", "base-b"],
            seed=17,
            bootstrap_replicates=400,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.5)
        self.assertLessEqual(first[1], 0.5)
        self.assertGreaterEqual(first[2], 0.5)


class VectorizedBootstrapTest(unittest.TestCase):
    def assert_stats_almost_equal(
        self, first: dict[str, float | None], second: dict[str, float | None]
    ) -> None:
        self.assertEqual(set(first), set(second))
        for name in first:
            if first[name] is None or second[name] is None:
                self.assertIs(first[name], second[name], name)
            else:
                self.assertAlmostEqual(first[name], second[name], places=12, msg=name)

    def test_summary_vectorization_matches_legacy_cluster_resampling(self) -> None:
        units: list[list[dict[str, object]]] = []
        for unit_index, size in enumerate((1, 2, 3, 1, 2)):
            unit = [
                prediction_row(
                    f"summary-{unit_index}-{row_index}",
                    f"summary-{unit_index}-{row_index}",
                    "full_video",
                    correct=(unit_index + row_index) % 3 != 0,
                    gold_margin=float(unit_index - row_index),
                )
                for row_index in range(size)
            ]
            units.append(unit)
        units[1][0]["gold_margin"] = None
        units[2][1]["choice_probability"] = None

        seed = 314159
        replicates = 300
        legacy = _bootstrap_stat_map(
            units,
            lambda sample: _summary_metrics_from_units(sample, ece_bins=4),
            seed=seed,
            bootstrap_replicates=replicates,
            confidence_level=0.9,
        )
        optimized = _bootstrap_summary_stat_map(
            units,
            ece_bins=4,
            seed=seed,
            bootstrap_replicates=replicates,
            confidence_level=0.9,
        )
        repeated = _bootstrap_summary_stat_map(
            units,
            ece_bins=4,
            seed=seed,
            bootstrap_replicates=replicates,
            confidence_level=0.9,
        )
        self.assert_stats_almost_equal(legacy, optimized)
        self.assertEqual(optimized, repeated)

    def test_paired_vectorization_matches_legacy_cluster_resampling(self) -> None:
        units = [
            [
                {
                    "left_correct": (unit_index + row_index) % 2 == 0,
                    "right_correct": row_index % 3 != 0,
                    "left_margin": float(unit_index - row_index),
                    "right_margin": float(row_index - 1),
                }
                for row_index in range(size)
            ]
            for unit_index, size in enumerate((1, 3, 2, 4, 1))
        ]
        units[2][0]["left_margin"] = None
        units[3][1]["right_correct"] = None

        legacy = _bootstrap_stat_map(
            units,
            _paired_metrics_from_units,
            seed=2718,
            bootstrap_replicates=300,
            confidence_level=0.95,
        )
        optimized = _bootstrap_paired_stat_map(
            units,
            seed=2718,
            bootstrap_replicates=300,
            confidence_level=0.95,
        )
        self.assert_stats_almost_equal(legacy, optimized)

    def test_metric_missing_from_a_resample_is_omitted_like_legacy(self) -> None:
        units = [
            [
                {
                    "left_correct": True,
                    "right_correct": False,
                    "left_margin": None,
                    "right_margin": None,
                }
            ],
            [
                {
                    "left_correct": None,
                    "right_correct": None,
                    "left_margin": 2.0,
                    "right_margin": 1.0,
                }
            ],
        ]
        legacy = _bootstrap_stat_map(
            units,
            _paired_metrics_from_units,
            seed=7,
            bootstrap_replicates=200,
            confidence_level=0.95,
        )
        optimized = _bootstrap_paired_stat_map(
            units,
            seed=7,
            bootstrap_replicates=200,
            confidence_level=0.95,
        )
        self.assert_stats_almost_equal(legacy, optimized)

    def test_count_batches_are_deterministic_and_memory_bounded(self) -> None:
        # This checks the production-scale invariant without a flaky wall-clock
        # assertion: memory is bounded by count cells, independent of the full
        # replicate count, and every count row still contains exactly N draws.
        kwargs = {
            "num_units": 137,
            "bootstrap_replicates": 513,
            "seed": 99,
            "max_count_cells": 137 * 7,
        }
        first = list(_bootstrap_count_batches(**kwargs))
        second = list(_bootstrap_count_batches(**kwargs))
        differently_chunked = list(
            _bootstrap_count_batches(**{**kwargs, "max_count_cells": 137 * 3})
        )
        self.assertEqual(sum(batch.shape[0] for batch in first), 513)
        for left, right in zip(first, second):
            self.assertLessEqual(left.size, 137 * 7)
            self.assertTrue((left.sum(axis=1) == 137).all())
            self.assertTrue((left == right).all())
        first_rows = [row.tolist() for batch in first for row in batch]
        chunked_rows = [row.tolist() for batch in differently_chunked for row in batch]
        self.assertEqual(first_rows, chunked_rows)


class SummaryAndComparisonTest(unittest.TestCase):
    def test_independent_unit_clusters_multiple_candidate_rows(self) -> None:
        first = prediction_row("candidate-a", "base-a", "full_video", correct=True)
        second = prediction_row("candidate-b", "base-b", "full_video", correct=False)
        first["independent_unit_id"] = "clevrer:scene:question"
        second["independent_unit_id"] = "clevrer:scene:question"
        first["resampling_unit_id"] = "clevrer:scene"
        second["resampling_unit_id"] = "clevrer:scene"
        result = analyze_predictions([first, second], bootstrap_replicates=20)
        summary = result["summary"][0]
        self.assertEqual(summary["observed_unique_base_ids"], 2)
        self.assertEqual(summary["observed_unique_clusters"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["row_micro_accuracy"], 0.5)
        self.assertEqual(summary["cluster_macro_accuracy"], 0.5)
        self.assertEqual(summary["cluster_all_rows_correct"], 0.0)

    def test_official_question_aggregation_is_nested_in_scene_resampling(self) -> None:
        rows = []
        for question_id, correctness in (("q1", (True, True)), ("q2", (True, False))):
            for candidate_index, correct in enumerate(correctness):
                row = prediction_row(
                    f"{question_id}-{candidate_index}",
                    f"{question_id}-{candidate_index}",
                    "full_video",
                    correct=correct,
                    resampling_unit_id="clevrer:scene:10000",
                )
                row["independent_unit_id"] = f"clevrer:10000:{question_id}"
                rows.append(row)
        summary = analyze_predictions(rows, bootstrap_replicates=20)["summary"][0]
        self.assertEqual(summary["observed_unique_clusters"], 2)
        self.assertEqual(summary["observed_unique_resampling_units"], 1)
        self.assertEqual(summary["cluster_macro_accuracy"], 0.75)
        self.assertEqual(summary["cluster_all_rows_correct"], 0.5)

    def test_grouped_bootstrap_resamples_source_families_not_child_questions(
        self,
    ) -> None:
        units = [[{"value": 0.0}], [{"value": 1.0}], [{"value": 1.0}]]

        def calculate(sample: list[list[dict[str, float]]]) -> dict[str, float]:
            return {"mean": sum(unit[0]["value"] for unit in sample) / len(sample)}

        grouped = _bootstrap_grouped_stat_map(
            units,
            [[0, 1], [2]],
            calculate,
            seed=17,
            bootstrap_replicates=200,
            confidence_level=0.95,
        )
        independent = _bootstrap_grouped_stat_map(
            units,
            [[0], [1], [2]],
            calculate,
            seed=17,
            bootstrap_replicates=200,
            confidence_level=0.95,
        )
        self.assertEqual(grouped["mean"], 2.0 / 3.0)
        self.assertNotEqual(
            (grouped["mean_ci_low"], grouped["mean_ci_high"]),
            (independent["mean_ci_low"], independent["mean_ci_high"]),
        )

    def test_summary_separates_row_micro_from_cluster_macro_accuracy(self) -> None:
        rows = [
            prediction_row(
                f"large-{index}", f"large-{index}", "full_video", correct=False
            )
            for index in range(3)
        ]
        for row in rows:
            row["independent_unit_id"] = "large-cluster"
            row["resampling_unit_id"] = "large-source-video"
        small = prediction_row("small", "small", "full_video", correct=True)
        small["independent_unit_id"] = "small-cluster"
        rows.append(small)

        summary = analyze_predictions(rows, bootstrap_replicates=20)["summary"][0]
        self.assertEqual(summary["row_micro_accuracy"], 0.25)
        self.assertEqual(summary["cluster_macro_accuracy"], 0.5)
        self.assertEqual(summary["accuracy"], summary["cluster_macro_accuracy"])

    def test_group_summary_and_paired_gain(self) -> None:
        rows = [
            prediction_row("full-a", "a", "full_video", correct=False),
            prediction_row("full-b", "b", "full_video", correct=True),
            prediction_row(
                "oracle-a",
                "a",
                "ordered_oracle",
                correct=True,
                requested_dose=2,
                effective_dose=2,
            ),
            prediction_row(
                "oracle-b",
                "b",
                "ordered_oracle",
                correct=True,
                requested_dose=2,
                effective_dose=2,
            ),
        ]
        first = analyze_predictions(rows, seed=9, bootstrap_replicates=200)
        second = analyze_predictions(rows, seed=9, bootstrap_replicates=200)
        self.assertEqual(first["summary"], second["summary"])
        oracle_summary = next(
            row for row in first["summary"] if row["condition"] == "ordered_oracle"
        )
        self.assertEqual(oracle_summary["accuracy"], 1.0)
        self.assertEqual(oracle_summary["observed_unique_base_ids"], 2)
        comparison = next(
            row
            for row in first["comparisons"]
            if row["comparison_type"] == "condition_vs_reference"
            and row["condition"] == "ordered_oracle"
        )
        self.assertEqual(comparison["aligned_rows"], 2)
        self.assertAlmostEqual(comparison["accuracy_gain"], 0.5)
        self.assertLessEqual(comparison["accuracy_gain_ci_low"], 0.5)
        self.assertGreaterEqual(comparison["accuracy_gain_ci_high"], 0.5)

    def test_evidence_formulas_have_explicit_orientation(self) -> None:
        rows: list[dict[str, object]] = []
        patterns = {
            "full_video": (True, False),
            "evidence_only": (True, True),
            "evidence_present": (True, False),
            "evidence_removed": (False, False),
            "random_position_mask": (True, False),
            "random_matched": (True, False),
        }
        for condition, values in patterns.items():
            for base_id, correct in zip(("a", "b"), values):
                rows.append(
                    prediction_row(
                        f"{condition}-{base_id}", base_id, condition, correct=correct
                    )
                )
        result = analyze_predictions(rows, seed=3, bootstrap_replicates=100)
        evidence = {
            row["contrast"]: row
            for row in result["comparisons"]
            if row["comparison_type"] == "evidence_metric"
        }
        self.assertAlmostEqual(evidence["evidence_sufficiency"]["accuracy_gain"], 0.5)
        self.assertAlmostEqual(
            evidence["evidence_comprehensiveness"]["accuracy_gain"], 0.5
        )
        self.assertAlmostEqual(
            evidence["random_mask_placebo_cost"]["accuracy_gain"], 0.0
        )
        self.assertAlmostEqual(
            evidence["evidence_mask_specificity"]["accuracy_gain"], 0.5
        )
        self.assertAlmostEqual(evidence["random_control"]["accuracy_gain"], 0.0)
        self.assertAlmostEqual(evidence["evidence_specificity"]["accuracy_gain"], 0.5)

    def test_text_placebo_formulas_isolate_information_from_format(self) -> None:
        patterns = {
            "atomic_oracle": (False, False),
            "ordered_timestamp_sham": (False, False),
            "ordered_oracle": (True, False),
            "reasoning_operator_sham": (True, False),
            "reasoning_oracle": (True, True),
        }
        rows = [
            prediction_row(
                f"{condition}-{base_id}", base_id, condition, correct=correct
            )
            for condition, values in patterns.items()
            for base_id, correct in zip(("a", "b"), values)
        ]
        result = analyze_predictions(rows, seed=5, bootstrap_replicates=40)
        placebo = {
            row["contrast"]: row
            for row in result["comparisons"]
            if row["comparison_type"] == "placebo_metric"
        }
        self.assertAlmostEqual(
            placebo["timestamp_information_over_sham"]["accuracy_gain"], 0.5
        )
        self.assertAlmostEqual(
            placebo["timestamp_format_placebo"]["accuracy_gain"], 0.0
        )
        self.assertAlmostEqual(
            placebo["operator_information_over_sham"]["accuracy_gain"], 0.5
        )
        self.assertAlmostEqual(placebo["operator_format_placebo"]["accuracy_gain"], 0.0)

    def test_confirmatory_orientation_is_left_minus_right(self) -> None:
        rows = [
            prediction_row("full-a", "a", "full_video", correct=True),
            prediction_row("full-b", "b", "full_video", correct=False),
            prediction_row("question-a", "a", "question_only", correct=False),
            prediction_row("question-b", "b", "question_only", correct=False),
        ]
        result = analyze_predictions(
            rows,
            confirmatory_comparisons=[["full_video", "question_only"]],
            bootstrap_replicates=40,
        )
        comparison = next(
            row
            for row in result["comparisons"]
            if row["comparison_type"] == "confirmatory"
        )
        self.assertEqual(comparison["contrast"], "full_video_minus_question_only")
        self.assertEqual(comparison["condition"], "full_video")
        self.assertEqual(comparison["reference_condition"], "question_only")
        self.assertAlmostEqual(comparison["accuracy_gain"], 0.5)
        self.assertEqual(comparison["dose_alignment"], "broadcast_single_dose")
        self.assertEqual(
            result["report"]["protocol"]["confirmatory_comparisons"],
            [["full_video", "question_only"]],
        )

    def test_confirmatory_multi_dose_conditions_match_requested_dose(self) -> None:
        rows: list[dict[str, object]] = []
        # At dose 1 reasoning beats ordered; at dose 2 it loses.  Cross-dose
        # matching would either be ambiguous or erase this intended contrast.
        for dose, reasoning_correct, ordered_correct in (
            (1, True, False),
            (2, False, True),
        ):
            for base_id in ("a", "b"):
                rows.extend(
                    [
                        prediction_row(
                            f"reasoning-{dose}-{base_id}",
                            base_id,
                            "reasoning_oracle",
                            correct=reasoning_correct,
                            requested_dose=dose,
                            effective_dose=dose,
                        ),
                        prediction_row(
                            f"ordered-{dose}-{base_id}",
                            base_id,
                            "ordered_oracle",
                            correct=ordered_correct,
                            requested_dose=dose,
                            effective_dose=dose,
                        ),
                    ]
                )
        expected = [manifest_row(row) for row in rows]
        result = analyze_predictions(
            rows,
            expected,
            confirmatory_comparisons=[["reasoning_oracle", "ordered_oracle"]],
            bootstrap_replicates=40,
        )
        comparisons = [
            row
            for row in result["comparisons"]
            if row["comparison_type"] == "confirmatory"
        ]
        self.assertEqual(len(comparisons), 2)
        by_dose = {row["requested_dose"]: row for row in comparisons}
        self.assertEqual(set(by_dose), {1, 2})
        self.assertAlmostEqual(by_dose[1]["accuracy_gain"], 1.0)
        self.assertAlmostEqual(by_dose[2]["accuracy_gain"], -1.0)
        for dose, comparison in by_dose.items():
            self.assertEqual(comparison["reference_requested_dose"], dose)
            self.assertEqual(comparison["dose_alignment"], "matched_requested_dose")
            self.assertEqual(comparison["aligned_rows"], 2)
        diagnostics = result["report"]["analysis_coverage"]["comparisons"][
            "confirmatory::reasoning_oracle_minus_ordered_oracle"
        ]
        self.assertEqual(diagnostics["ambiguous_left_alignment_keys"], 0)
        self.assertEqual(diagnostics["ambiguous_right_alignment_keys"], 0)

    def test_confirmatory_pools_effective_dose_within_requested_dose(self) -> None:
        rows: list[dict[str, object]] = []
        for base_id, condition_effective, reference_effective in (
            ("a", 1, 1),
            ("b", 2, 1),
        ):
            rows.extend(
                [
                    prediction_row(
                        f"reasoning-{base_id}",
                        base_id,
                        "reasoning_oracle",
                        correct=True,
                        requested_dose="all",
                        effective_dose=condition_effective,
                    ),
                    prediction_row(
                        f"ordered-{base_id}",
                        base_id,
                        "ordered_oracle",
                        correct=False,
                        requested_dose="all",
                        effective_dose=reference_effective,
                    ),
                ]
            )
        expected = [manifest_row(row) for row in rows]
        result = analyze_predictions(
            rows,
            expected,
            confirmatory_comparisons=[["reasoning_oracle", "ordered_oracle"]],
            minimum_confirmatory_resampling_units=2,
            bootstrap_replicates=0,
        )
        comparisons = [
            row
            for row in result["comparisons"]
            if row["comparison_type"] == "confirmatory"
        ]
        self.assertEqual(len(comparisons), 1)
        comparison = comparisons[0]
        self.assertEqual(comparison["requested_dose"], "all")
        self.assertIsNone(comparison["effective_dose"])
        self.assertEqual(comparison["aligned_rows"], 2)
        self.assertEqual(comparison["condition_effective_dose_min"], 1.0)
        self.assertEqual(comparison["condition_effective_dose_max"], 2.0)
        self.assertEqual(comparison["reference_effective_dose_mean"], 1.0)
        failures = _require_complete_failures(result["report"], expected_requested=True)
        self.assertNotIn(
            "confirmatory_comparison_below_minimum_resampling_units::"
            "reasoning_oracle_minus_ordered_oracle",
            failures,
        )

        reverse = analyze_predictions(
            rows,
            expected,
            confirmatory_comparisons=[["ordered_oracle", "reasoning_oracle"]],
            minimum_confirmatory_resampling_units=2,
            bootstrap_replicates=0,
        )
        reverse_comparisons = [
            row
            for row in reverse["comparisons"]
            if row["comparison_type"] == "confirmatory"
        ]
        self.assertEqual(len(reverse_comparisons), 1)
        reverse_comparison = reverse_comparisons[0]
        self.assertEqual(reverse_comparison["aligned_rows"], 2)
        self.assertEqual(reverse_comparison["accuracy_gain"], -1.0)
        self.assertEqual(reverse_comparison["condition_effective_dose_mean"], 1.0)
        self.assertEqual(reverse_comparison["reference_effective_dose_mean"], 1.5)

    def test_confirmatory_completeness_enforces_locked_resampling_minimum(self) -> None:
        rows = [
            prediction_row(f"full-{base}", base, "full_video", correct=True)
            for base in ("a", "b")
        ] + [
            prediction_row(f"question-{base}", base, "question_only", correct=False)
            for base in ("a", "b")
        ]
        expected = [manifest_row(row) for row in rows]
        complete = analyze_predictions(
            rows,
            expected,
            confirmatory_comparisons=[["full_video", "question_only"]],
            minimum_confirmatory_resampling_units=2,
            bootstrap_replicates=0,
        )
        complete_failures = _require_complete_failures(
            complete["report"], expected_requested=True
        )
        self.assertFalse(
            any(
                value.startswith("confirmatory_comparison_")
                for value in complete_failures
            )
        )

        underpowered = analyze_predictions(
            rows,
            expected,
            confirmatory_comparisons=[["full_video", "question_only"]],
            minimum_confirmatory_resampling_units=3,
            bootstrap_replicates=0,
        )
        underpowered_failures = _require_complete_failures(
            underpowered["report"], expected_requested=True
        )
        self.assertIn(
            "confirmatory_comparison_below_minimum_resampling_units::full_video_minus_question_only",
            underpowered_failures,
        )

    def test_confirmatory_completeness_requires_clevrer_primary_authentication(
        self,
    ) -> None:
        report = {
            "coverage": {
                "expected_trials": 1,
                "joined_coverage": 1.0,
                "prediction_input": {"duplicate_trial_ids": 0},
                "manifest_input": {"duplicate_trial_ids": 0},
                "issues": {"counts": {}},
            },
            "score_metadata_authentication": {"authenticated": True},
            "protocol": {
                "minimum_confirmatory_resampling_units": 1,
                "confirmatory_comparisons": [],
            },
            "analysis_coverage": {
                "comparisons": {},
                "clevrer_primary": {
                    "summary_groups": 1,
                    "authenticated_summary_groups": 0,
                    "confirmatory_comparison_rows": 1,
                    "authenticated_confirmatory_comparison_rows": 0,
                },
            },
        }
        failures = _require_complete_failures(report, expected_requested=True)
        self.assertIn("clevrer_primary_summary_not_authenticated", failures)
        self.assertIn("clevrer_primary_comparison_not_authenticated", failures)

    def test_confirmatory_completeness_rejects_missing_and_unaligned_conditions(
        self,
    ) -> None:
        left = prediction_row("full-a", "a", "full_video", correct=True)
        missing = analyze_predictions(
            [left],
            [manifest_row(left)],
            confirmatory_comparisons=[["full_video", "question_only"]],
            bootstrap_replicates=0,
        )
        missing_failures = _require_complete_failures(
            missing["report"], expected_requested=True
        )
        self.assertIn(
            "confirmatory_comparison_not_computed::full_video_minus_question_only",
            missing_failures,
        )

        right = prediction_row("question-b", "b", "question_only", correct=False)
        unaligned = analyze_predictions(
            [left, right],
            [manifest_row(left), manifest_row(right)],
            confirmatory_comparisons=[["full_video", "question_only"]],
            bootstrap_replicates=0,
        )
        unaligned_failures = _require_complete_failures(
            unaligned["report"], expected_requested=True
        )
        for expected_failure in (
            "confirmatory_comparison_zero_expected_alignment_units::full_video_minus_question_only",
            "confirmatory_comparison_zero_aligned_units::full_video_minus_question_only",
            "confirmatory_comparison_incomplete_expected_alignment::full_video_minus_question_only",
            "confirmatory_comparison_unmatched_expected_alignment::full_video_minus_question_only",
        ):
            self.assertIn(expected_failure, unaligned_failures)

    def test_confirmatory_completeness_rejects_incomplete_or_ambiguous_alignment(
        self,
    ) -> None:
        left = prediction_row("full-a", "a", "full_video", correct=True)
        right = prediction_row("question-a", "a", "question_only", correct=False)
        incomplete = analyze_predictions(
            [left],
            [manifest_row(left), manifest_row(right)],
            confirmatory_comparisons=[["full_video", "question_only"]],
            bootstrap_replicates=0,
        )
        incomplete_failures = _require_complete_failures(
            incomplete["report"], expected_requested=True
        )
        for expected_failure in (
            "confirmatory_comparison_zero_aligned_units::full_video_minus_question_only",
            "confirmatory_comparison_incomplete_observed_alignment::full_video_minus_question_only",
            "confirmatory_comparison_unmatched_observed_alignment::full_video_minus_question_only",
        ):
            self.assertIn(expected_failure, incomplete_failures)

        duplicate_right = prediction_row(
            "question-a-duplicate", "a", "question_only", correct=False
        )
        ambiguous_rows = [left, right, duplicate_right]
        ambiguous = analyze_predictions(
            ambiguous_rows,
            [manifest_row(row) for row in ambiguous_rows],
            confirmatory_comparisons=[["full_video", "question_only"]],
            bootstrap_replicates=0,
        )
        ambiguous_failures = _require_complete_failures(
            ambiguous["report"], expected_requested=True
        )
        self.assertIn(
            "confirmatory_comparison_ambiguous_expected_alignment::full_video_minus_question_only",
            ambiguous_failures,
        )
        self.assertIn(
            "confirmatory_comparison_ambiguous_observed_alignment::full_video_minus_question_only",
            ambiguous_failures,
        )

    def test_confirmatory_resampling_minimum_must_be_positive_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            analyze_predictions(
                [], minimum_confirmatory_resampling_units=0, bootstrap_replicates=0
            )


class PairAndPermutationTest(unittest.TestCase):
    def test_counterfactual_nuisance_and_permutation_metrics(self) -> None:
        rows: list[dict[str, object]] = []
        for permutation, answer_label in ((0, "A"), (1, "B")):
            rows.extend(
                [
                    prediction_row(
                        f"orig-{permutation}",
                        "original-base",
                        "full_video",
                        correct=True,
                        permutation_index=permutation,
                        pair_id="pair-1",
                        pair_role="original",
                        prediction=answer_label,
                        prediction_text="yes",
                        answer=answer_label,
                        answer_text="yes",
                    ),
                    prediction_row(
                        f"cf-{permutation}",
                        "counterfactual-base",
                        "full_video",
                        correct=True,
                        permutation_index=permutation,
                        pair_id="pair-1",
                        pair_role="counterfactual",
                        prediction=answer_label,
                        prediction_text="no",
                        answer=answer_label,
                        answer_text="no",
                    ),
                    prediction_row(
                        f"nuisance-{permutation}",
                        "nuisance-base",
                        "full_video",
                        correct=True,
                        permutation_index=permutation,
                        pair_id="pair-1",
                        pair_role="nuisance",
                        prediction=answer_label,
                        prediction_text="yes",
                        answer=answer_label,
                        answer_text="yes",
                    ),
                ]
            )
        expected = [manifest_row(row) for row in rows]
        result = analyze_predictions(rows, expected, seed=11, bootstrap_replicates=80)
        metrics = result["pair_metrics"][0]
        self.assertEqual(metrics["counterfactual_complete_units"], 2)
        self.assertEqual(metrics["counterfactual_both_correct"], 1.0)
        self.assertEqual(metrics["counterfactual_correct_semantic_flip"], 1.0)
        self.assertEqual(metrics["nuisance_both_correct"], 1.0)
        self.assertEqual(metrics["nuisance_invariance"], 1.0)
        self.assertEqual(metrics["permutation_complete_units"], 3)
        self.assertEqual(metrics["permutation_semantic_consistency"], 1.0)
        self.assertEqual(metrics["permutation_all_permutations_correct"], 1.0)

    def test_incomplete_permutation_is_coverage_not_success(self) -> None:
        complete = prediction_row(
            "p0", "base", "full_video", correct=True, permutation_index=0
        )
        missing = prediction_row(
            "p1", "base", "full_video", correct=True, permutation_index=1
        )
        result = analyze_predictions(
            [complete],
            [manifest_row(complete), manifest_row(missing)],
            bootstrap_replicates=20,
        )
        metrics = result["pair_metrics"][0]
        self.assertEqual(metrics["permutation_expected_units"], 1)
        self.assertEqual(metrics["permutation_complete_units"], 0)
        self.assertEqual(metrics["permutation_coverage"], 0.0)
        self.assertIsNone(metrics.get("permutation_all_permutations_correct"))


class DoseAndCoverageTest(unittest.TestCase):
    def test_k90_uses_first_requested_and_effective_dose(self) -> None:
        rows = [
            # Deliberately make full-video stronger than the low-dose text
            # oracle.  Using full_video as K90 baseline would yield no positive
            # curve; the correct question-only baseline yields .5 then 1.0.
            prediction_row("full-a", "a", "full_video", correct=True),
            prediction_row("full-b", "b", "full_video", correct=True),
            prediction_row("question-a", "a", "question_only", correct=False),
            prediction_row("question-b", "b", "question_only", correct=False),
            prediction_row(
                "d1-a",
                "a",
                "reasoning_oracle",
                correct=True,
                requested_dose=1,
                effective_dose=1,
            ),
            prediction_row(
                "d1-b",
                "b",
                "reasoning_oracle",
                correct=False,
                requested_dose=1,
                effective_dose=1,
            ),
            prediction_row(
                "d2-a",
                "a",
                "reasoning_oracle",
                correct=True,
                requested_dose=2,
                effective_dose=2,
            ),
            prediction_row(
                "d2-b",
                "b",
                "reasoning_oracle",
                correct=True,
                requested_dose=2,
                effective_dose=2,
            ),
        ]
        result = analyze_predictions(rows, seed=5, bootstrap_replicates=100)
        curves = [
            row
            for row in result["dose_curves"]
            if row["condition"] == "reasoning_oracle"
        ]
        self.assertEqual(len(curves), 2)
        self.assertEqual(curves[0]["oracle_gain"], 0.5)
        self.assertEqual(curves[1]["oracle_gain"], 1.0)
        self.assertEqual(curves[0]["k90_requested_dose"], 2)
        self.assertEqual(curves[0]["k90_effective_dose"], 2.0)
        self.assertEqual(curves[0]["reference_condition"], "question_only")
        comparisons = [
            row
            for row in result["comparisons"]
            if row["condition"] == "reasoning_oracle"
        ]
        self.assertEqual(
            {row["comparison_type"] for row in comparisons},
            {"condition_vs_reference", "oracle_vs_question_only"},
        )

    def test_visual_plus_text_dose_uses_full_video_reference(self) -> None:
        rows = [
            prediction_row("full-a", "a", "full_video", correct=False),
            prediction_row("full-b", "b", "full_video", correct=False),
            prediction_row("question-a", "a", "question_only", correct=True),
            prediction_row("question-b", "b", "question_only", correct=True),
            prediction_row(
                "plus-a",
                "a",
                "video_plus_reasoning_oracle",
                correct=True,
                requested_dose=2,
                effective_dose=2,
            ),
            prediction_row(
                "plus-b",
                "b",
                "video_plus_reasoning_oracle",
                correct=True,
                requested_dose=2,
                effective_dose=2,
            ),
        ]
        result = analyze_predictions(rows, bootstrap_replicates=40)
        curve = next(
            row
            for row in result["dose_curves"]
            if row["condition"] == "video_plus_reasoning_oracle"
        )
        self.assertEqual(curve["reference_condition"], "full_video")
        self.assertEqual(curve["oracle_gain"], 1.0)

    def test_manifest_reports_duplicate_missing_and_group_coverage(self) -> None:
        first = prediction_row("trial-1", "a", "full_video", correct=True)
        second = prediction_row("trial-2", "b", "full_video", correct=True)
        result = analyze_predictions(
            [first, dict(first)],
            [manifest_row(first), manifest_row(second)],
            bootstrap_replicates=20,
        )
        coverage = result["report"]["coverage"]
        self.assertEqual(
            coverage["prediction_input"]["duplicate_trial_ids"], ["trial-1"]
        )
        self.assertEqual(coverage["joined_prediction_trials"], 0)
        self.assertIn("trial-2", coverage["missing_prediction_trial_ids"])
        summary = result["summary"][0]
        self.assertEqual(summary["expected_rows"], 2)
        self.assertEqual(summary["observed_rows"], 0)
        self.assertEqual(summary["row_coverage"], 0.0)

    def test_nested_trial_manifest_supplies_design_fields(self) -> None:
        expected = {
            "trial_id": "nested-1",
            "base_id": "base-1",
            "visual_id": "visual-1",
            "answer": "A",
            "answer_text": "yes",
            "condition": {
                "name": "full_video",
                "input_channel": "visual",
                "requested_dose": 0,
                "effective_dose": 0,
                "permutation_index": 0,
            },
            "diagnostic": {
                "dataset": "nested-dataset",
                "information_family": "static",
                "question_family": "attribute",
                "reasoning_depth": 0,
                "pair_id": "standalone-1",
                "pair_role": "standalone",
            },
        }
        prediction = {
            "trial_id": "nested-1",
            "prediction": "A",
            "prediction_text": "yes",
            "correct": True,
            "choice_probability": {"A": 0.9, "B": 0.1},
            "gold_margin": 2.0,
        }
        result = analyze_predictions([prediction], [expected], bootstrap_replicates=20)
        summary = result["summary"][0]
        self.assertEqual(summary["dataset"], "nested-dataset")
        self.assertEqual(summary["condition"], "full_video")
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["row_coverage"], 1.0)


class StrictScoredRowValidationTest(unittest.TestCase):
    def test_choice_nll_rederives_every_reported_score_field(self) -> None:
        mutations = {
            "choice_probability_nll_inconsistency": lambda row: row.update(
                {"choice_probability": {"A": 0.6, "B": 0.4}}
            ),
            "prediction_nll_inconsistency": lambda row: row.update(
                {"prediction": "B", "prediction_text": "no", "correct": False}
            ),
            "gold_nll_inconsistency": lambda row: row.update({"gold_nll": 99.0}),
            "best_distractor_nll_inconsistency": lambda row: row.update(
                {"best_distractor_nll": 99.0}
            ),
            "gold_margin_nll_inconsistency": lambda row: row.update(
                {"gold_margin": 99.0}
            ),
            "correct_nll_inconsistency": lambda row: row.update({"correct": False}),
        }
        for expected_issue, mutate in mutations.items():
            with self.subTest(expected_issue=expected_issue):
                row = prediction_row("nll", "nll", "full_video", correct=True)
                expected = manifest_row(row)
                mutate(row)
                # Authenticate the modified row itself so this test isolates
                # score algebra from the independent row-digest check.
                row["result_content_sha256"] = scored_result_sha256(row)
                result = analyze_predictions([row], [expected], bootstrap_replicates=0)
                issues = result["report"]["coverage"]["issues"]["counts"]
                self.assertIn(expected_issue, issues)

    def test_result_digest_detects_post_run_score_edit(self) -> None:
        row = prediction_row("digest", "digest", "full_video", correct=True)
        expected = manifest_row(row)
        row["gold_margin"] = float(row["gold_margin"]) + 0.25
        result = analyze_predictions([row], [expected], bootstrap_replicates=0)
        issues = result["report"]["coverage"]["issues"]["counts"]
        self.assertEqual(issues["result_content_sha256_mismatch"], 1)

    def test_clevrer_primary_metric_is_authenticated_official_question_exact_set(
        self,
    ) -> None:
        rows = [
            prediction_row("candidate-0", "candidate-0", "full_video", correct=True),
            prediction_row("candidate-1", "candidate-1", "full_video", correct=False),
        ]
        for index, row in enumerate(rows):
            row.update(
                {
                    "dataset": "clevrer",
                    "independent_unit_id": "clevrer:scene:question",
                    "resampling_unit_id": "clevrer:scene",
                    "official_candidate_id": str(index),
                    "official_candidate_count": 2,
                }
            )
            row["result_content_sha256"] = scored_result_sha256(row)
        result = analyze_predictions(
            rows, [manifest_row(row) for row in rows], bootstrap_replicates=10
        )
        summary = result["summary"][0]
        self.assertTrue(summary["official_question_exact_set_authenticated"])
        self.assertEqual(summary["official_question_exact_set_questions"], 1)
        self.assertEqual(summary["official_question_exact_set_accuracy"], 0.0)
        self.assertEqual(
            summary["primary_accuracy_metric"],
            "official_question_exact_set_accuracy",
        )
        self.assertEqual(summary["primary_accuracy"], 0.0)
        self.assertFalse(summary["candidate_level_accuracy_primary"])

    def test_clevrer_semantic_aggregation_recovers_one_letter_position_error(
        self,
    ) -> None:
        first = prediction_row(
            "candidate-0-permutation-0",
            "question-0-candidate-0",
            "full_video",
            correct=True,
            permutation_index=0,
            answer="A",
            answer_text="Yes",
            choices=["Yes", "No"],
        )
        second = prediction_row(
            "candidate-0-permutation-1",
            "question-0-candidate-0",
            "full_video",
            correct=False,
            permutation_index=1,
            answer="B",
            answer_text="Yes",
            choices=["No", "Yes"],
        )
        for row in (first, second):
            row.update(
                {
                    "dataset": "clevrer",
                    "independent_unit_id": "clevrer:scene-0:question-0",
                    "resampling_unit_id": "clevrer:scene-0",
                    "official_candidate_id": "0",
                    "official_candidate_count": 1,
                }
            )
        # Yes wins 0.90/0.10 in the first permutation. In the second,
        # letter A (No) narrowly wins 0.55/0.45. Semantic averaging still
        # predicts Yes: mean 0.675 versus 0.325.
        set_choice_probabilities(first, {"A": 0.90, "B": 0.10})
        set_choice_probabilities(second, {"A": 0.55, "B": 0.45})

        result = analyze_predictions(
            [first, second],
            [manifest_row(first), manifest_row(second)],
            seed=19,
            bootstrap_replicates=40,
        )
        summary = result["summary"][0]
        self.assertTrue(summary["official_question_exact_set_authenticated"])
        self.assertEqual(summary["official_question_exact_set_accuracy"], 1.0)
        self.assertEqual(
            summary["official_question_permutation_robustness_accuracy"], 0.0
        )
        self.assertEqual(summary["cluster_all_rows_correct"], 0.0)
        self.assertEqual(summary["official_question_exact_set_accuracy_ci_low"], 1.0)
        self.assertEqual(summary["official_question_exact_set_accuracy_ci_high"], 1.0)
        self.assertIn(
            "mean the probability assigned to each semantic option",
            result["report"]["protocol"]["clevrer_semantic_aggregation_rule"],
        )

    def test_clevrer_semantic_primary_uses_canonical_nll_near_tie(self) -> None:
        row = prediction_row(
            "candidate-0-permutation-0",
            "question-0-candidate-0",
            "full_video",
            correct=True,
            answer="A",
            answer_text="Yes",
            choices=["Yes", "No"],
        )
        row.update(
            {
                "dataset": "clevrer",
                "independent_unit_id": "clevrer:scene-0:question-0",
                "resampling_unit_id": "clevrer:scene-0",
                "official_candidate_id": "0",
                "official_candidate_count": 1,
                "choice_nll": {"A": 1.0, "B": 1.000002},
                # This reverses the semantic argmax but remains within the
                # serialization tolerance of the NLL-derived probabilities.
                "choice_probability": {"A": 0.4999995, "B": 0.5000005},
                "prediction": "A",
                "prediction_text": "Yes",
                "correct": True,
                "gold_nll": 1.0,
                "best_distractor_nll": 1.000002,
                "gold_margin": 0.000002,
            }
        )
        row["result_content_sha256"] = scored_result_sha256(row)
        result = analyze_predictions(
            [row], [manifest_row(row)], bootstrap_replicates=10
        )
        issues = result["report"]["coverage"]["issues"]["counts"]
        self.assertNotIn("choice_probability_nll_inconsistency", issues)
        summary = result["summary"][0]
        self.assertEqual(summary["official_question_exact_set_accuracy"], 1.0)
        self.assertGreater(summary["gold_probability"], 0.5)

    def test_clevrer_attested_all_requires_exact_candidate_permutation_indices(
        self,
    ) -> None:
        rows = [
            prediction_row(
                f"candidate-0-permutation-{permutation_index}",
                "question-0-candidate-0",
                "full_video",
                correct=True,
                permutation_index=permutation_index,
                answer=answer,
                answer_text="Yes",
                choices=choices,
            )
            for permutation_index, answer, choices in (
                (0, "A", ["Yes", "No"]),
                # Two rows alone are insufficient: for a binary candidate the
                # authenticated all-permutation design must use indices {0, 1}.
                (2, "B", ["No", "Yes"]),
            )
        ]
        expected = []
        attestation = {
            "sampling": {"option_permutations": "all"},
        }
        for row in rows:
            row.update(
                {
                    "dataset": "clevrer",
                    "independent_unit_id": "clevrer:scene-0:question-0",
                    "resampling_unit_id": "clevrer:scene-0",
                    "official_candidate_id": "0",
                    "official_candidate_count": 1,
                }
            )
            manifest = manifest_row(row)
            manifest["trial_build_attestation"] = attestation
            manifest["trial_content_sha256"] = compute_trial_content_sha256(manifest)
            row["trial_content_sha256"] = manifest["trial_content_sha256"]
            row["result_content_sha256"] = scored_result_sha256(row)
            expected.append(manifest)

        result = analyze_predictions(rows, expected, bootstrap_replicates=10)
        summary = result["summary"][0]
        self.assertFalse(summary["official_question_exact_set_authenticated"])
        self.assertEqual(
            summary["official_question_semantic_aggregation_failures"][
                "attested_all_permutation_set_mismatch"
            ],
            1,
        )

    def test_clevrer_confirmatory_exact_set_gain_is_not_candidate_gain(self) -> None:
        rows: list[dict[str, object]] = []
        for condition, correctness in (
            ("ordered_oracle", (True, True)),
            ("full_video", (True, False)),
        ):
            for candidate_index, correct in enumerate(correctness):
                row = prediction_row(
                    f"{condition}-candidate-{candidate_index}",
                    f"question-0-candidate-{candidate_index}",
                    condition,
                    correct=correct,
                    answer="A",
                    answer_text="Yes",
                    choices=["Yes", "No"],
                )
                row.update(
                    {
                        "dataset": "clevrer",
                        "independent_unit_id": "clevrer:scene-0:question-0",
                        "resampling_unit_id": "clevrer:scene-0",
                        "official_candidate_id": str(candidate_index),
                        "official_candidate_count": 2,
                    }
                )
                set_choice_probabilities(
                    row,
                    {"A": 0.90, "B": 0.10} if correct else {"A": 0.20, "B": 0.80},
                )
                rows.append(row)

        result = analyze_predictions(
            rows,
            [manifest_row(row) for row in rows],
            seed=23,
            bootstrap_replicates=40,
            confirmatory_comparisons=(("ordered_oracle", "full_video"),),
        )
        comparison = next(
            row
            for row in result["comparisons"]
            if row["comparison_type"] == "confirmatory"
            and row["contrast"] == "ordered_oracle_minus_full_video"
        )
        self.assertTrue(
            comparison["official_question_exact_set_comparison_authenticated"]
        )
        self.assertEqual(
            comparison["condition_official_question_exact_set_accuracy"], 1.0
        )
        self.assertEqual(
            comparison["reference_official_question_exact_set_accuracy"], 0.0
        )
        self.assertEqual(comparison["official_question_exact_set_accuracy_gain"], 1.0)
        self.assertEqual(comparison["accuracy_gain"], 0.5)
        self.assertNotEqual(
            comparison["official_question_exact_set_accuracy_gain"],
            comparison["accuracy_gain"],
        )
        self.assertEqual(
            comparison["official_question_exact_set_accuracy_gain_ci_low"], 1.0
        )
        self.assertEqual(
            comparison["official_question_exact_set_accuracy_gain_ci_high"], 1.0
        )
        self.assertEqual(
            comparison["official_question_exact_set_comparison_questions"], 1
        )
        self.assertFalse(comparison["candidate_level_accuracy_metrics_primary"])

    def test_clevrer_incomplete_candidate_set_is_rejected(self) -> None:
        row = prediction_row("candidate-0", "candidate-0", "full_video", correct=True)
        row.update(
            {
                "dataset": "clevrer",
                "independent_unit_id": "clevrer:scene:question",
                "official_candidate_id": "0",
                "official_candidate_count": 2,
            }
        )
        row["result_content_sha256"] = scored_result_sha256(row)
        result = analyze_predictions([row], [manifest_row(row)], bootstrap_replicates=0)
        issues = result["report"]["coverage"]["issues"]["counts"]
        self.assertEqual(issues["clevrer_incomplete_official_candidate_set"], 1)
        self.assertFalse(
            result["summary"][0]["official_question_exact_set_authenticated"]
        )

    def test_clevrer_requires_explicit_scene_resampling_unit(self) -> None:
        row = prediction_row("candidate-0", "candidate-0", "full_video", correct=True)
        row.update(
            {
                "dataset": "clevrer",
                "independent_unit_id": "clevrer:scene:question",
                "official_candidate_id": "0",
                "official_candidate_count": 1,
            }
        )
        row.pop("resampling_unit_id")
        row["trial_content_sha256"] = compute_trial_content_sha256(row)
        row["result_content_sha256"] = scored_result_sha256(row)
        result = analyze_predictions(
            [row], [manifest_row(row)], bootstrap_replicates=10
        )
        issues = result["report"]["coverage"]["issues"]["counts"]
        self.assertEqual(issues["clevrer_missing_resampling_unit_id"], 1)
        self.assertFalse(
            result["summary"][0]["official_question_exact_set_authenticated"]
        )
        failures = _require_complete_failures(result["report"], expected_requested=True)
        self.assertIn("clevrer_missing_resampling_unit_id", failures)
        self.assertIn("clevrer_primary_summary_not_authenticated", failures)

    def test_projected_visual_token_budget_is_strictly_validated(self) -> None:
        first = prediction_row("tokens-a", "tokens-a", "full_video", correct=True)
        second = prediction_row("tokens-b", "tokens-b", "full_video", correct=True)
        for row in (first, second):
            row["visual_id"] = "shared-visual"
            row["token_source"] = "projected_visual_features"
        first["original_visual_tokens"] = 32
        first["effective_visual_tokens"] = 16
        second["original_visual_tokens"] = 32.0
        second["effective_visual_tokens"] = 32
        expected = [manifest_row(first), manifest_row(second)]

        result = analyze_predictions([first, second], expected, bootstrap_replicates=10)
        coverage = result["report"]["coverage"]
        issue_counts = coverage["issues"]["counts"]
        self.assertEqual(issue_counts["projected_visual_token_truncation"], 1)
        self.assertEqual(issue_counts["invalid_projected_original_visual_tokens"], 1)
        failures = _require_complete_failures(result["report"], expected_requested=True)
        self.assertIn("projected_visual_token_truncation", failures)
        self.assertIn("invalid_projected_original_visual_tokens", failures)

    def test_same_visual_id_must_have_one_projected_token_budget(self) -> None:
        rows = [
            prediction_row("tokens-a", "a", "full_video", correct=True),
            prediction_row("tokens-b", "b", "video_plus_ordered_oracle", correct=True),
        ]
        for count, row in zip((32, 48), rows):
            row["visual_id"] = "shared-visual"
            row["token_source"] = "projected_visual_features"
            row["original_visual_tokens"] = count
            row["effective_visual_tokens"] = count
        result = analyze_predictions(
            rows, [manifest_row(row) for row in rows], bootstrap_replicates=10
        )
        issue_counts = result["report"]["coverage"]["issues"]["counts"]
        self.assertEqual(issue_counts["inconsistent_projected_visual_token_budget"], 1)

    def test_manifest_binding_labels_probabilities_and_semantics_are_validated(
        self,
    ) -> None:
        names = (
            "binding",
            "missing-binding",
            "label",
            "keys",
            "value",
            "total",
            "correctness",
            "semantic",
        )
        predictions = [
            prediction_row(name, name, "full_video", correct=True) for name in names
        ]
        expected = [manifest_row(row) for row in predictions]

        predictions[0]["trial_content_sha256"] = "f" * 64
        predictions[0]["choices"] = ["no", "yes"]
        predictions[1].pop("trial_content_sha256")
        predictions[2]["prediction"] = "C"
        predictions[2]["prediction_text"] = "<invalid>"
        predictions[3]["choice_probability"] = {"A": 0.5, "C": 0.5}
        predictions[4]["choice_probability"] = {"A": 1.1, "B": -0.1}
        predictions[5]["choice_probability"] = {"A": 0.0, "B": 0.0}
        predictions[6]["prediction"] = "B"
        predictions[6]["prediction_text"] = "no"
        predictions[7]["prediction_text"] = "not the selected choice"

        result = analyze_predictions(predictions, expected, bootstrap_replicates=10)
        coverage = result["report"]["coverage"]
        issue_counts = coverage["issues"]["counts"]
        self.assertEqual(coverage["metadata_mismatch_count"], 2)
        self.assertEqual(issue_counts["prediction_manifest_mismatch"], 2)
        self.assertEqual(issue_counts["prediction_manifest_missing_binding_field"], 1)
        self.assertEqual(issue_counts["invalid_prediction_label"], 1)
        self.assertEqual(issue_counts["invalid_choice_probability_key_set"], 1)
        self.assertEqual(issue_counts["invalid_choice_probability_value"], 1)
        self.assertEqual(issue_counts["invalid_choice_probability_total"], 1)
        self.assertEqual(issue_counts["correct_prediction_inconsistency"], 2)
        self.assertEqual(issue_counts["prediction_text_choice_mismatch"], 1)
        self.assertEqual(coverage["field_present_count"]["trial_content_sha256"], 8)
        self.assertEqual(coverage["field_present_count"]["choices"], 8)

    def test_positive_finite_probability_weights_are_normalized(self) -> None:
        row = prediction_row("weights", "weights", "full_video", correct=True)
        row["choice_probability"] = {"A": 2.0, "B": 1.0}
        # Probability-only development inputs retain the documented fallback;
        # authenticated scorer outputs with choice_nll use canonical NLL scores.
        row.pop("choice_nll")
        row.pop("gold_nll")
        row.pop("best_distractor_nll")
        row["result_content_sha256"] = scored_result_sha256(row)
        result = analyze_predictions(
            [row], [manifest_row(row)], bootstrap_replicates=10
        )
        coverage = result["report"]["coverage"]
        self.assertEqual(coverage["valid_probability_rows"], 1)
        self.assertNotIn(
            "invalid_choice_probability_total", coverage["issues"]["counts"]
        )
        self.assertAlmostEqual(result["summary"][0]["gold_probability"], 2.0 / 3.0)

    def test_stale_manifest_hash_and_mixed_scoring_runs_are_reported(self) -> None:
        first = prediction_row("first", "first", "full_video", correct=True)
        second = prediction_row(
            "second",
            "second",
            "full_video",
            correct=True,
            scoring_run_signature_sha256="b" * 64,
        )
        expected = [manifest_row(first), manifest_row(second)]
        expected[0]["clue_text"] = "hand-edited after the hash was created"

        result = analyze_predictions([first, second], expected, bootstrap_replicates=10)
        coverage = result["report"]["coverage"]
        issue_counts = coverage["issues"]["counts"]
        self.assertEqual(issue_counts["stale_manifest_trial_content_sha256"], 1)
        self.assertEqual(issue_counts["mixed_scoring_run_signatures"], 1)
        self.assertEqual(coverage["scoring_run_signature_count"], 2)


class CliAndConfigTest(unittest.TestCase):
    def test_development_escape_hatch_is_visibly_watermarked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            out_dir = root / "analysis"
            predictions.write_text(
                json.dumps(prediction_row("dev", "dev", "full_video", correct=True))
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "analyze",
                        "--predictions",
                        str(predictions),
                        "--out-dir",
                        str(out_dir),
                        "--bootstrap-replicates",
                        "0",
                        "--development",
                    ]
                ),
                0,
            )
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["analysis_mode"]["mode"], "development")
            self.assertFalse(report["analysis_mode"]["confirmatory"])
            self.assertIn("DEVELOPMENT RESULT", report["analysis_mode"]["warning"])

    def test_analysis_refuses_implicit_replacement_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            out_dir = root / "analysis"
            predictions.write_text(
                json.dumps(prediction_row("dev", "dev", "full_video", correct=True))
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "analyze",
                "--predictions",
                str(predictions),
                "--out-dir",
                str(out_dir),
                "--bootstrap-replicates",
                "0",
                "--development",
            ]
            self.assertEqual(main(argv), 0)
            with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
                main(argv)
            self.assertEqual(main([*argv, "--overwrite"]), 0)

    def test_analysis_rejects_output_input_alias_even_with_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "analysis"
            out_dir.mkdir()
            predictions = out_dir / "report.json"
            predictions.write_text(
                json.dumps(prediction_row("dev", "dev", "full_video", correct=True))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output aliases an input"):
                main(
                    [
                        "analyze",
                        "--predictions",
                        str(predictions),
                        "--out-dir",
                        str(out_dir),
                        "--development",
                        "--overwrite",
                    ]
                )

    def test_confirmatory_cli_override_conflicting_with_protocol_is_refused(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            predictions = root / "predictions.jsonl"
            write_locked_analysis_protocol(protocol, bootstrap_replicates=10)
            predictions.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "conflicts with the locked protocol"
            ):
                main(
                    [
                        "analyze",
                        "--predictions",
                        str(predictions),
                        "--out-dir",
                        str(root / "analysis"),
                        "--config",
                        str(protocol),
                        "--bootstrap-replicates",
                        "9",
                    ]
                )

    def test_confirmatory_cli_requires_positive_bootstrap_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            predictions = root / "predictions.jsonl"
            write_locked_analysis_protocol(protocol, bootstrap_replicates=0)
            predictions.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "bootstrap_replicates must be a positive integer"
            ):
                main(
                    [
                        "analyze",
                        "--predictions",
                        str(predictions),
                        "--out-dir",
                        str(root / "analysis"),
                        "--config",
                        str(protocol),
                    ]
                )

    def test_authenticated_multiple_run_signatures_share_one_global_signature(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_path = root / "merged.jsonl"
            manifest_path = root / "merged-trials.jsonl"
            protocol_path = root / "protocol.json"
            first_metadata = root / "shard-0.metadata.json"
            second_metadata = root / "shard-1.metadata.json"
            out_dir = root / "analysis"
            rows = [
                prediction_row("shard-0", "a", "full_video", correct=True),
                prediction_row("shard-1", "b", "full_video", correct=True),
            ]
            manifest = [manifest_row(row) for row in rows]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            write_locked_analysis_protocol(protocol_path, bootstrap_replicates=3)
            write_score_sidecar(
                path=first_metadata,
                predictions=[rows[0]],
                manifest=[manifest[0]],
                manifest_path=manifest_path,
                protocol_path=protocol_path,
                full_manifest=manifest,
            )
            write_score_sidecar(
                path=second_metadata,
                predictions=[rows[1]],
                manifest=[manifest[1]],
                manifest_path=manifest_path,
                protocol_path=protocol_path,
                full_manifest=manifest,
            )
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            exit_code = main(
                [
                    "analyze",
                    "--predictions",
                    str(predictions_path),
                    "--expected-trials",
                    str(manifest_path),
                    "--score-metadata",
                    str(first_metadata),
                    "--score-metadata",
                    str(second_metadata),
                    "--out-dir",
                    str(out_dir),
                    "--config",
                    str(protocol_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            authentication = report["score_metadata_authentication"]
            self.assertTrue(authentication["authenticated"])
            self.assertTrue(authentication["authenticated_sharded_run"])
            self.assertEqual(len(authentication["run_signatures"]), 2)

    def test_strict_analysis_rejects_proper_subset_of_locked_trial_matrix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_path = root / "predictions.jsonl"
            manifest_path = root / "trials.jsonl"
            protocol_path = root / "protocol.json"
            metadata_path = predictions_path.with_suffix(".jsonl.metadata.json")
            out_dir = root / "analysis"
            rows = [
                prediction_row("full-0", "a", "full_video", correct=True),
                prediction_row("full-1", "b", "full_video", correct=True),
            ]
            full_manifest = [manifest_row(row) for row in rows]
            write_locked_analysis_protocol(protocol_path, bootstrap_replicates=1)
            write_score_sidecar(
                path=metadata_path,
                predictions=[rows[0]],
                manifest=[full_manifest[0]],
                full_manifest=full_manifest,
                manifest_path=manifest_path,
                protocol_path=protocol_path,
            )
            predictions_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "analyze",
                        "--predictions",
                        str(predictions_path),
                        "--expected-trials",
                        str(manifest_path),
                        "--out-dir",
                        str(out_dir),
                        "--config",
                        str(protocol_path),
                    ]
                ),
                2,
            )
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            issue_counts = report["coverage"]["issues"]["counts"]
            self.assertEqual(issue_counts["manifest_trial_matrix_closure_mismatch"], 1)
            self.assertIn(
                "manifest_trial_matrix_closure_mismatch",
                report["require_complete"]["failures"],
            )

    def test_self_consistent_sidecar_with_wrong_locked_projector_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_path = root / "predictions.jsonl"
            manifest_path = root / "trials.jsonl"
            metadata_path = root / "predictions.jsonl.metadata.json"
            protocol_path = root / "protocol.json"
            out_dir = root / "analysis"
            row = prediction_row("tampered", "tampered", "full_video", correct=True)
            manifest = [manifest_row(row)]
            write_locked_analysis_protocol(protocol_path, bootstrap_replicates=1)
            write_score_sidecar(
                path=metadata_path,
                predictions=[row],
                manifest=manifest,
                manifest_path=manifest_path,
                protocol_path=protocol_path,
            )
            sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
            wrong_projector = "f" * 64
            sidecar["global_signature"]["projector_checkpoint_sha256"] = wrong_projector
            new_global = canonical_sha256(sidecar["global_signature"])
            sidecar["global_signature_sha256"] = new_global
            sidecar["run_signature"]["projector_checkpoint_sha256"] = wrong_projector
            sidecar["run_signature"]["scoring_global_signature_sha256"] = new_global
            new_run = canonical_sha256(sidecar["run_signature"])
            sidecar["run_signature_sha256"] = new_run
            row["scoring_global_signature_sha256"] = new_global
            row["scoring_run_signature_sha256"] = new_run
            row["result_content_sha256"] = scored_result_sha256(row)
            metadata_path.write_text(json.dumps(sidecar), encoding="utf-8")
            predictions_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "analyze",
                        "--predictions",
                        str(predictions_path),
                        "--expected-trials",
                        str(manifest_path),
                        "--out-dir",
                        str(out_dir),
                        "--config",
                        str(protocol_path),
                    ]
                ),
                2,
            )
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            authentication = report["score_metadata_authentication"]
            self.assertFalse(authentication["authenticated"])
            self.assertIn(
                "score_metadata_global_protocol_mismatch",
                {issue["kind"] for issue in authentication["issues"]},
            )

    def test_protocol_loader_uses_analysis_seed_not_sampling_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "protocol.yaml"
            config.write_text(
                "analysis:\n"
                "  seed: 9001\n"
                "  bootstrap_replicates: 25\n"
                "  minimum_confirmatory_resampling_units: 7\n"
                "sampling:\n"
                "  seed: 7\n"
                "confirmatory_comparisons:\n"
                "  - [full_video, question_only]\n"
                "  - [reasoning_oracle, ordered_oracle]\n",
                encoding="utf-8",
            )
            values, _ = load_protocol_config(config)
            self.assertEqual(values["seed"], 9001)
            self.assertEqual(values["bootstrap_replicates"], 25)
            self.assertEqual(values["minimum_confirmatory_resampling_units"], 7)
            self.assertEqual(
                values["confirmatory_comparisons"],
                [
                    ["full_video", "question_only"],
                    ["reasoning_oracle", "ordered_oracle"],
                ],
            )

    def test_protocol_loader_accepts_safe_dump_block_comparison_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "protocol.final.yaml"
            config.write_text(
                "analysis:\n"
                "  seed: 9001\n"
                "  bootstrap_replicates: 25\n"
                "confirmatory_comparisons:\n"
                "- - full_video\n"
                "  - question_only\n"
                "- - reasoning_oracle\n"
                "  - ordered_oracle\n",
                encoding="utf-8",
            )
            values, _ = load_protocol_config(config)
            self.assertEqual(values["seed"], 9001)
            self.assertEqual(
                values["confirmatory_comparisons"],
                [
                    ["full_video", "question_only"],
                    ["reasoning_oracle", "ordered_oracle"],
                ],
            )

    def test_confirmatory_completeness_rejects_empty_clevrer_comparisons(
        self,
    ) -> None:
        report = {
            "coverage": {
                "expected_trials": 1,
                "joined_coverage": 1.0,
                "prediction_input": {"duplicate_trial_ids": 0},
                "manifest_input": {"duplicate_trial_ids": 0},
                "issues": {"counts": {}},
            },
            "score_metadata_authentication": {"authenticated": True},
            "protocol": {
                "minimum_confirmatory_resampling_units": 1,
                "confirmatory_comparisons": [],
            },
            "analysis_coverage": {
                "comparisons": {},
                "clevrer_primary": {
                    "summary_groups": 1,
                    "authenticated_summary_groups": 1,
                    "confirmatory_comparison_rows": 0,
                    "authenticated_confirmatory_comparison_rows": 0,
                },
            },
        }
        failures = _require_complete_failures(report, expected_requested=True)
        self.assertIn("clevrer_confirmatory_comparisons_missing", failures)

    def test_scalar_yaml_and_cli_override_write_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            config = root / "protocol.yaml"
            out_dir = root / "analysis"
            row = prediction_row("trial", "base", "full_video", correct=True)
            predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
            config.write_text(
                "analysis:\n"
                "  bootstrap_replicates: 17\n"
                "  confidence_level: 0.9\n"
                "  seed: 123\n"
                "  reference_condition: full_video\n"
                "  ece_bins: 4\n",
                encoding="utf-8",
            )
            values, status = load_protocol_config(config)
            self.assertTrue(status["found"])
            self.assertEqual(values["bootstrap_replicates"], 17)
            exit_code = main(
                [
                    "analyze",
                    "--predictions",
                    str(predictions),
                    "--out-dir",
                    str(out_dir),
                    "--config",
                    str(config),
                    "--seed",
                    "999",
                    "--bootstrap-replicates",
                    "3",
                    "--development",
                ]
            )
            self.assertEqual(exit_code, 0)
            for name in (
                "summary.csv",
                "comparisons.csv",
                "pair_metrics.csv",
                "dose_curves.csv",
                "report.json",
            ):
                self.assertTrue((out_dir / name).is_file(), name)
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["protocol_resolution"]["resolved"]["seed"], 999)
            self.assertEqual(
                report["protocol_resolution"]["resolved"]["bootstrap_replicates"], 3
            )
            with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)

    def test_require_complete_writes_outputs_then_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_path = root / "predictions.jsonl"
            manifest_path = root / "trials.jsonl"
            protocol_path = root / "protocol.json"
            out_dir = root / "analysis"
            present = prediction_row("present", "a", "full_video", correct=True)
            missing = prediction_row("missing", "b", "full_video", correct=True)
            broken = dict(present)
            broken["choice_probability"] = None
            manifest = [manifest_row(present), manifest_row(missing)]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            write_locked_analysis_protocol(protocol_path)
            write_score_sidecar(
                path=predictions_path.with_suffix(".jsonl.metadata.json"),
                predictions=[broken],
                manifest=manifest,
                manifest_path=manifest_path,
                protocol_path=protocol_path,
            )
            predictions_path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
            exit_code = main(
                [
                    "analyze",
                    "--predictions",
                    str(predictions_path),
                    "--expected-trials",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--config",
                    str(protocol_path),
                    "--require-complete",
                ]
            )
            self.assertEqual(exit_code, 2)
            self.assertTrue((out_dir / "summary.csv").is_file())
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["require_complete"]["passed"])
            self.assertIn(
                "expected_trial_join_coverage_below_one",
                report["require_complete"]["failures"],
            )
            self.assertIn(
                "invalid_choice_probability", report["require_complete"]["failures"]
            )

    def test_require_complete_passes_for_exact_valid_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = prediction_row("only", "a", "full_video", correct=True)
            predictions_path = root / "predictions.jsonl"
            manifest_path = root / "trials.jsonl"
            protocol_path = root / "protocol.json"
            out_dir = root / "analysis"
            manifest = [manifest_row(prediction)]
            manifest_path.write_text(json.dumps(manifest[0]) + "\n", encoding="utf-8")
            write_locked_analysis_protocol(protocol_path)
            write_score_sidecar(
                path=predictions_path.with_suffix(".jsonl.metadata.json"),
                predictions=[prediction],
                manifest=manifest,
                manifest_path=manifest_path,
                protocol_path=protocol_path,
            )
            predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            exit_code = main(
                [
                    "analyze",
                    "--predictions",
                    str(predictions_path),
                    "--expected-trials",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--config",
                    str(protocol_path),
                    "--require-complete",
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["require_complete"]["passed"])

    def test_gzipped_prediction_and_manifest_jsonl_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = prediction_row("gzip", "gzip", "full_video", correct=True)
            predictions_path = root / "predictions.jsonl.gz"
            manifest_path = root / "trials.jsonl.gz"
            protocol_path = root / "protocol.json"
            out_dir = root / "analysis"
            manifest = [manifest_row(prediction)]
            with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest[0]) + "\n")
            write_locked_analysis_protocol(protocol_path, bootstrap_replicates=3)
            write_score_sidecar(
                path=predictions_path.with_suffix(".gz.metadata.json"),
                predictions=[prediction],
                manifest=manifest,
                manifest_path=manifest_path,
                protocol_path=protocol_path,
            )
            with gzip.open(predictions_path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(prediction) + "\n")

            exit_code = main(
                [
                    "analyze",
                    "--predictions",
                    str(predictions_path),
                    "--expected-trials",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--config",
                    str(protocol_path),
                    "--require-complete",
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["require_complete"]["passed"])
            self.assertEqual(report["coverage"]["joined_prediction_trials"], 1)

    def test_require_complete_fails_new_scored_row_integrity_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = [
                prediction_row("first", "a", "full_video", correct=True),
                prediction_row(
                    "second",
                    "b",
                    "full_video",
                    correct=True,
                    scoring_run_signature_sha256="b" * 64,
                ),
                prediction_row("third", "c", "full_video", correct=True),
            ]
            expected = [manifest_row(row) for row in predictions]
            predictions_path = root / "predictions.jsonl"
            manifest_path = root / "trials.jsonl"
            protocol_path = root / "protocol.json"
            out_dir = root / "analysis"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in expected), encoding="utf-8"
            )
            write_locked_analysis_protocol(protocol_path)
            write_score_sidecar(
                path=predictions_path.with_suffix(".jsonl.metadata.json"),
                predictions=predictions,
                manifest=expected,
                manifest_path=manifest_path,
                protocol_path=protocol_path,
            )
            expected[1]["clue_text"] = "stale edit"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in expected), encoding="utf-8"
            )
            # Apply the intended corruptions after authenticating the original
            # runner outputs, so digest and semantic checks both have work to do.
            predictions[0]["prediction"] = "B"
            predictions[0]["prediction_text"] = "no"
            predictions[0]["choice_probability"] = {"A": 0.4, "C": 0.6}
            predictions[1]["scoring_run_signature_sha256"] = "b" * 64
            predictions[2].pop("scoring_run_signature_sha256")
            predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
            )

            exit_code = main(
                [
                    "analyze",
                    "--predictions",
                    str(predictions_path),
                    "--expected-trials",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--config",
                    str(protocol_path),
                    "--require-complete",
                ]
            )
            self.assertEqual(exit_code, 2)
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            failures = report["require_complete"]["failures"]
            self.assertIn("correct_prediction_inconsistency", failures)
            self.assertIn("invalid_choice_probability_key_set", failures)
            self.assertIn("missing_scoring_run_signature_sha256", failures)
            self.assertIn("mixed_scoring_run_signatures", failures)
            self.assertIn("stale_manifest_trial_content_sha256", failures)


if __name__ == "__main__":
    unittest.main()
