from __future__ import annotations

from pathlib import Path
import unittest

from information_upper_bound.conditions import (
    ConditionSpec,
    build_trials,
    load_condition_config,
    matched_event_facts,
    render_clue,
)
from information_upper_bound.cli import _parse_option_permutations
from information_upper_bound.schema import (
    SCHEMA_VERSION,
    normalize_answer,
    validate_record,
)
from information_upper_bound.validate import validate_manifest


def base_record(record_id: str = "sample-1") -> dict:
    return {
        "id": record_id,
        "source": "fixture",
        "benchmark": "fixture",
        "task": "mcq",
        "media_type": "video",
        "media_path": "/data/not-required-for-unit-test.mp4",
        "question": "Which event happened first?",
        "choices": ["open", "close"],
        "answer": "A",
        "diagnostic": {
            "schema_version": SCHEMA_VERSION,
            "dataset": "fixture",
            "split": "test",
            "information_family": "temporal_order",
            "question_family": "event_order",
            "reasoning_depth": 1,
            "pair_id": "pair-1",
            "pair_role": "original",
            "resampling_unit_id": "fixture-video-1",
            "evidence_spans": [{"start": 1.0, "end": 3.0, "unit": "seconds"}],
            "oracles": {
                "static_facts": [
                    {
                        "text": "There is a door.",
                        "access": "safe_visual_gt",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    }
                ],
                "unordered_events": [
                    {
                        "event_id": "door-open",
                        "subject": "door",
                        "predicate": "opens",
                        "access": "safe_visual_gt",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    },
                    {
                        "event_id": "door-close",
                        "subject": "door",
                        "predicate": "closes",
                        "access": "safe_visual_gt",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    },
                ],
                "ordered_events": [
                    {
                        "event_id": "door-open",
                        "subject": "door",
                        "predicate": "opens",
                        "start_sec": 1.0,
                        "end_sec": 1.5,
                        "access": "safe_visual_gt",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    },
                    {
                        "event_id": "door-close",
                        "subject": "door",
                        "predicate": "closes",
                        "start_sec": 2.0,
                        "end_sec": 2.5,
                        "access": "safe_visual_gt",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    },
                ],
                "temporal_relations": [
                    {
                        "text": "opening occurs before closing",
                        "access": "safe_visual_gt",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    }
                ],
                "state_changes": [],
                "relations": [],
                "operator": [
                    {
                        "text": "compare event start times",
                        "access": "operator_only",
                        "source": "fixture_annotation",
                        "lineage": "official_adapter",
                    }
                ],
                "intermediate": [],
                "answer_derived": False,
            },
            "provenance": {"source_id": record_id},
        },
    }


class SchemaTests(unittest.TestCase):
    def test_valid_record(self) -> None:
        self.assertEqual(validate_record(base_record()), [])

    def test_integer_answer_is_not_guessed(self) -> None:
        with self.assertRaisesRegex(ValueError, "index base"):
            normalize_answer(0, ["yes", "no"])

    def test_invalid_normalized_span(self) -> None:
        record = base_record()
        record["diagnostic"]["evidence_spans"] = [
            {"start": 0.2, "end": 1.2, "unit": "normalized"}
        ]
        paths = {issue.path for issue in validate_record(record)}
        self.assertIn("diagnostic.evidence_spans[0].end", paths)

    def test_resampling_unit_is_required(self) -> None:
        record = base_record()
        del record["diagnostic"]["resampling_unit_id"]
        paths = {issue.path for issue in validate_record(record)}
        self.assertIn("diagnostic.resampling_unit_id", paths)

    def test_resampling_family_must_not_cross_splits(self) -> None:
        first = base_record("first")
        second = base_record("second")
        second["diagnostic"]["split"] = "validation"
        second["diagnostic"]["pair_id"] = "standalone:second"
        second["diagnostic"]["pair_role"] = "standalone"
        report = validate_manifest([first, second])
        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                issue["path"] == "diagnostic.resampling_unit_id"
                and "crosses splits" in issue["message"]
                for issue in report["issues"]
            )
        )

    def test_official_question_must_map_to_one_resampling_family(self) -> None:
        first = base_record("candidate-a")
        second = base_record("candidate-b")
        for row in (first, second):
            row["diagnostic"]["pair_id"] = f"standalone:{row['id']}"
            row["diagnostic"]["pair_role"] = "standalone"
            row["diagnostic"]["independent_unit_id"] = "official-question-1"
        second["diagnostic"]["resampling_unit_id"] = "different-scene"
        report = validate_manifest([first, second])
        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                "one official aggregation unit maps to multiple resampling families"
                in issue["message"]
                for issue in report["issues"]
            )
        )

    def test_oracle_lineage_and_answer_independence_are_explicit(self) -> None:
        record = base_record()
        record["diagnostic"]["oracles"].pop("answer_derived")
        record["diagnostic"]["oracles"]["mystery_answer_hint"] = ["A"]
        record["diagnostic"]["oracles"]["static_facts"] = ["The answer is open."]
        paths = [issue.path for issue in validate_record(record)]
        self.assertIn("diagnostic.oracles.answer_derived", paths)
        self.assertIn("diagnostic.oracles", paths)
        self.assertIn("diagnostic.oracles.static_facts[0]", paths)


class ConditionTests(unittest.TestCase):
    def test_unknown_clue_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported clue fields"):
            ConditionSpec(
                name="leaky",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("gold_answer_hint",),
            ).validate()

    def test_ordered_event_text_keeps_its_timestamp(self) -> None:
        record = base_record()
        record["diagnostic"]["oracles"]["ordered_events"] = [
            {
                "event_id": "cup-open",
                "text": "The cup opens.",
                "start": 1.25,
                "end": 2.5,
                "unit": "seconds",
                "access": "safe_visual_gt",
                "source": "fixture_annotation",
                "lineage": "official_adapter",
            }
        ]
        clue, count = render_clue(record, ["ordered_events"], "all")
        self.assertEqual(count, 1)
        self.assertIn("[t=1.25-2.5s] The cup opens.", clue)

    def test_option_permutation_preserves_semantic_gold(self) -> None:
        specs = [
            ConditionSpec(
                name="full", input_channel="visual", visual_view="full", doses=(0,)
            )
        ]
        trials, report = build_trials(
            [base_record()], specs, seed=7, option_permutations=2
        )
        self.assertEqual(report["trials"], 2)
        self.assertEqual({trial["answer_text"] for trial in trials}, {"open"})
        for trial in trials:
            gold_index = ord(trial["answer"]) - ord("A")
            self.assertEqual(trial["choices"][gold_index], "open")
        self.assertEqual(len({trial["visual_id"] for trial in trials}), 1)

    def test_all_option_positions_are_exactly_counterbalanced_per_item(self) -> None:
        record = base_record()
        record["choices"] = ["open", "close", "wait"]
        specs = [
            ConditionSpec(
                name="full", input_channel="visual", visual_view="full", doses=(0,)
            )
        ]
        trials, report = build_trials(
            [record], specs, seed=7, option_permutations="all"
        )
        self.assertEqual(report["option_permutations"], "all")
        self.assertEqual(len(trials), 3)
        self.assertEqual({trial["answer"] for trial in trials}, {"A", "B", "C"})
        self.assertTrue(
            all(
                trial["choices"][ord(trial["answer"]) - ord("A")] == "open"
                for trial in trials
            )
        )

    def test_trial_identity_changes_with_scoring_relevant_content(self) -> None:
        spec = ConditionSpec(
            name="ordered",
            input_channel="text_oracle",
            visual_view="none",
            clue_fields=("ordered_events",),
            doses=(1,),
        )
        original, _ = build_trials([base_record()], [spec])
        changed = base_record()
        changed["diagnostic"]["oracles"]["ordered_events"][0]["start_sec"] = 1.25
        modified, _ = build_trials([changed], [spec])
        self.assertNotEqual(original[0]["trial_id"], modified[0]["trial_id"])
        self.assertEqual(
            original[0]["trial_id"],
            "trial::" + original[0]["trial_content_sha256"],
        )
        self.assertEqual(len(original[0]["trial_content_sha256"]), 64)

    def test_trial_identity_binds_official_candidate_membership(self) -> None:
        spec = ConditionSpec(
            name="full",
            input_channel="visual",
            visual_view="full",
            doses=(0,),
        )
        first = base_record()
        first["diagnostic"]["official_candidate_id"] = "candidate-0"
        first["diagnostic"]["official_candidate_count"] = 2
        second = base_record()
        second["diagnostic"]["official_candidate_id"] = "candidate-1"
        second["diagnostic"]["official_candidate_count"] = 3
        first_trials, _ = build_trials([first], [spec])
        second_trials, _ = build_trials([second], [spec])
        self.assertNotEqual(first_trials[0]["trial_id"], second_trials[0]["trial_id"])

    def test_trial_identity_binds_confirmatory_data_lock(self) -> None:
        spec = ConditionSpec(
            name="full",
            input_channel="visual",
            visual_view="full",
            doses=(0,),
        )
        first = {**base_record(), "data_release_sha256": "a" * 64}
        second = {**base_record(), "data_release_sha256": "b" * 64}
        first_trials, _ = build_trials([first], [spec])
        second_trials, _ = build_trials([second], [spec])
        self.assertNotEqual(first_trials[0]["trial_id"], second_trials[0]["trial_id"])
        self.assertEqual(first_trials[0]["data_release_sha256"], "a" * 64)

    def test_trial_and_visual_identity_are_portable_across_mount_paths(self) -> None:
        spec = ConditionSpec(
            name="full",
            input_channel="visual",
            visual_view="full",
            doses=(0,),
        )
        first = base_record()
        first["diagnostic"]["provenance"]["video_id"] = "stable-video-7"
        second = base_record()
        second["diagnostic"]["provenance"]["video_id"] = "stable-video-7"
        second["media_path"] = "/another/mount/of/the/same/release/video.mp4"
        first_trials, _ = build_trials([first], [spec])
        second_trials, _ = build_trials([second], [spec])
        self.assertEqual(first_trials[0]["visual_id"], second_trials[0]["visual_id"])
        self.assertEqual(first_trials[0]["trial_id"], second_trials[0]["trial_id"])

    def test_cli_option_permutation_parser_accepts_all_and_rejects_zero(self) -> None:
        self.assertEqual(_parse_option_permutations("all"), "all")
        self.assertEqual(_parse_option_permutations("5"), 5)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _parse_option_permutations("0")

    def test_official_media_cut_is_carried_into_visual_spec(self) -> None:
        record = base_record()
        record["diagnostic"]["media_clip"] = {
            "start_frame": 0,
            "end_frame_exclusive": 60,
            "expected_total_frames": 100,
            "unit": "frames",
            "source": "official_cut_frame_mapping",
        }
        spec = ConditionSpec(
            name="full", input_channel="visual", visual_view="full", doses=(0,)
        )
        trials, _ = build_trials([record], [spec])
        self.assertEqual(trials[0]["visual_spec"]["clip"]["end_frame_exclusive"], 60)

    def test_clue_dose_is_answer_independent_prefix(self) -> None:
        spec = ConditionSpec(
            name="ordered",
            input_channel="text_oracle",
            visual_view="none",
            clue_fields=("ordered_events", "operator"),
            doses=(1, 2, "all"),
        )
        trials, _ = build_trials([base_record()], [spec], option_permutations=1)
        self.assertEqual(
            [trial["condition"]["effective_dose"] for trial in trials], [1, 2, 3]
        )
        self.assertTrue(all(trial["visual_id"] is None for trial in trials))

    def test_required_operator_is_in_every_reasoning_dose(self) -> None:
        spec = ConditionSpec(
            name="reasoning",
            input_channel="embedding_oracle",
            visual_view="none",
            clue_fields=("ordered_events", "operator"),
            always_include_fields=("operator",),
            required_fields=("operator",),
            doses=(1, 2),
        )
        trials, _ = build_trials([base_record()], [spec], option_permutations=1)
        self.assertEqual(
            [trial["condition"]["effective_dose"] for trial in trials], [2, 3]
        )
        self.assertTrue(
            all("Reasoning operator" in trial["clue_text"] for trial in trials)
        )

    def test_atomic_and_ordered_conditions_use_identical_event_ids_at_each_dose(
        self,
    ) -> None:
        specs = [
            ConditionSpec(
                name="atomic",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("unordered_events",),
                matched_event_view="atomic",
                requires_matched_events=True,
                doses=(1, 2),
            ),
            ConditionSpec(
                name="ordered",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events",),
                matched_event_view="ordered",
                requires_matched_events=True,
                doses=(1, 2),
            ),
        ]
        trials, _ = build_trials([base_record()], specs, option_permutations=1)
        by_key = {
            (trial["condition"]["name"], trial["condition"]["requested_dose"]): trial
            for trial in trials
        }
        for dose in (1, 2):
            atomic = by_key[("atomic", dose)]
            ordered = by_key[("ordered", dose)]
            atomic_audit = atomic["condition"]["clue_audit"]
            ordered_audit = ordered["condition"]["clue_audit"]
            self.assertEqual(
                atomic_audit["selected_fact_ids"], ordered_audit["selected_fact_ids"]
            )
            self.assertEqual(
                atomic_audit["selected_fact_semantic_sha256"],
                ordered_audit["selected_fact_semantic_sha256"],
            )
            self.assertNotIn("t=", atomic["clue_text"])
            self.assertIn("t=", ordered["clue_text"])

    def test_matched_event_semantics_must_differ_only_by_time_and_rendering(
        self,
    ) -> None:
        record = base_record()
        record["diagnostic"]["oracles"]["ordered_events"][0]["predicate"] = "locks"
        with self.assertRaisesRegex(ValueError, "changes non-temporal semantics"):
            matched_event_facts(record)

        record = base_record()
        record["diagnostic"]["oracles"]["unordered_events"][0]["text"] = (
            "The answer is A."
        )
        record["diagnostic"]["oracles"]["ordered_events"][0]["text"] = "The door opens."
        with self.assertRaisesRegex(ValueError, "changes non-temporal semantics"):
            matched_event_facts(record)

    def test_reasoning_condition_adds_operator_to_same_ordered_fact_set(self) -> None:
        specs = [
            ConditionSpec(
                name="ordered",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events",),
                matched_event_view="ordered",
                requires_matched_events=True,
                doses=(1,),
            ),
            ConditionSpec(
                name="reasoning",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events", "operator"),
                always_include_fields=("operator",),
                required_fields=("operator",),
                matched_event_view="ordered",
                requires_matched_events=True,
                doses=(1,),
            ),
        ]
        trials, _ = build_trials([base_record()], specs, option_permutations=1)
        self.assertEqual(
            trials[0]["condition"]["clue_audit"]["selected_fact_ids"],
            trials[1]["condition"]["clue_audit"]["selected_fact_ids"],
        )
        self.assertNotIn("Reasoning operator", trials[0]["clue_text"])
        self.assertIn("Reasoning operator", trials[1]["clue_text"])

    def test_timestamp_and_operator_shams_preserve_design_but_remove_information(
        self,
    ) -> None:
        specs = [
            ConditionSpec(
                name="ordered",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events",),
                matched_event_view="ordered",
                requires_matched_events=True,
                doses=(2,),
            ),
            ConditionSpec(
                name="timestamp_sham",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events",),
                matched_event_view="timestamp_sham",
                requires_matched_events=True,
                doses=(2,),
            ),
            ConditionSpec(
                name="reasoning",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events", "operator"),
                always_include_fields=("operator",),
                required_fields=("operator",),
                matched_event_view="ordered",
                requires_matched_events=True,
                doses=(2,),
            ),
            ConditionSpec(
                name="operator_sham",
                input_channel="text_oracle",
                visual_view="none",
                clue_fields=("ordered_events", "operator"),
                always_include_fields=("operator",),
                required_fields=("operator",),
                sham_fields=("operator",),
                matched_event_view="ordered",
                requires_matched_events=True,
                doses=(2,),
            ),
        ]
        trials, _ = build_trials([base_record()], specs, option_permutations=1)
        by_name = {trial["condition"]["name"]: trial for trial in trials}
        selected_ids = {
            tuple(trial["condition"]["clue_audit"]["selected_fact_ids"])
            for trial in trials
        }
        semantic_hashes = {
            tuple(trial["condition"]["clue_audit"]["selected_fact_semantic_sha256"])
            for trial in trials
        }
        self.assertEqual(len(selected_ids), 1)
        self.assertEqual(len(semantic_hashes), 1)
        self.assertIn("[t=000.000-000.000s]", by_name["timestamp_sham"]["clue_text"])
        self.assertNotIn("t=1-1.5", by_name["timestamp_sham"]["clue_text"])
        self.assertNotIn(
            "compare event start times", by_name["operator_sham"]["clue_text"]
        )
        self.assertEqual(
            len(by_name["reasoning"]["clue_text"]),
            len(by_name["operator_sham"]["clue_text"]),
        )
        sham_counts = by_name["operator_sham"]["condition"]["clue_audit"][
            "sham_character_counts"
        ]
        self.assertEqual(
            sham_counts[0]["source_characters"], sham_counts[0]["sham_characters"]
        )

        changed_answer = base_record()
        changed_answer["answer"] = "B"
        changed_trials, _ = build_trials(
            [changed_answer], [specs[1], specs[3]], option_permutations=1
        )
        self.assertEqual(
            [
                by_name["timestamp_sham"]["clue_text"],
                by_name["operator_sham"]["clue_text"],
            ],
            [trial["clue_text"] for trial in changed_trials],
        )

    def test_default_config_is_loadable(self) -> None:
        config = Path(__file__).parents[1] / "configs" / "conditions.yaml"
        specs, options = load_condition_config(config)
        self.assertGreaterEqual(len(specs), 10)
        self.assertEqual(options["seed"], 42)
        self.assertEqual(options["option_permutations"], "all")
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["atomic_oracle"].clue_fields[0], "unordered_events")
        self.assertEqual(by_name["ordered_oracle"].clue_fields[0], "ordered_events")
        self.assertEqual(
            by_name["ordered_timestamp_sham"].matched_event_view,
            "timestamp_sham",
        )
        self.assertEqual(
            by_name["reasoning_operator_sham"].sham_fields,
            ("operator",),
        )
        self.assertEqual(
            by_name["random_position_mask"].visual_view,
            "random_position_mask",
        )


class PairValidationTests(unittest.TestCase):
    def test_source_video_ids_are_namespaced_by_dataset(self) -> None:
        first = base_record("dataset-a")
        first["diagnostic"]["dataset"] = "dataset-a"
        first["diagnostic"]["split"] = "train"
        first["diagnostic"]["pair_id"] = "standalone:dataset-a"
        first["diagnostic"]["pair_role"] = "standalone"
        first["diagnostic"]["provenance"]["video_id"] = "1"
        first["media_path"] = "/data/dataset-a/1.mp4"
        second = base_record("dataset-b")
        second["diagnostic"]["dataset"] = "dataset-b"
        second["diagnostic"]["split"] = "test"
        second["diagnostic"]["pair_id"] = "standalone:dataset-b"
        second["diagnostic"]["pair_role"] = "standalone"
        second["diagnostic"]["provenance"]["video_id"] = "1"
        second["media_path"] = "/data/dataset-b/1.mp4"
        report = validate_manifest([first, second])
        self.assertTrue(report["valid"], report["issues"])

    def test_counterfactual_must_flip_and_nuisance_must_not(self) -> None:
        original = base_record("original")
        counterfactual = base_record("counterfactual")
        counterfactual["answer"] = "B"
        counterfactual["diagnostic"]["pair_role"] = "counterfactual"
        nuisance = base_record("nuisance")
        nuisance["diagnostic"]["pair_role"] = "nuisance"
        report = validate_manifest([original, counterfactual, nuisance])
        self.assertTrue(report["valid"], report["issues"])

        counterfactual["answer"] = "A"
        report = validate_manifest([original, counterfactual, nuisance])
        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                "counterfactual must change" in issue["message"]
                for issue in report["issues"]
            )
        )


if __name__ == "__main__":
    unittest.main()
