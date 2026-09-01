"""Parser contract tests using small fixtures shaped like the official releases.

The fixtures intentionally exercise official field names and indexing rules.
They validate ingestion code only; they are not toy substitutes for running or
reporting scientific results on the complete benchmark releases.
"""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from information_upper_bound.adapters import (
    AdapterError,
    available_adapters,
    load_records,
)
from information_upper_bound.adapters.cli import main as adapter_main
from information_upper_bound.adapters.common import resolve_media
from information_upper_bound.conditions import render_clue
from information_upper_bound.schema import validate_record
from information_upper_bound.validate import validate_manifest


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"official-schema-shaped-parser-fixture")
    return path


class AdapterTests(unittest.TestCase):
    def test_registry_contains_all_diagnostic_datasets(self) -> None:
        self.assertEqual(
            set(available_adapters()),
            {
                "tempcompass",
                "tvbench",
                "perception_test",
                "next_gqa",
                "clevrer",
                "egoschema",
                "mvp",
            },
        )

    def test_adapter_rejects_output_report_or_input_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = _json(root / "annotations.json", {})
            media_root = root / "media"
            media_root.mkdir()
            same_output = root / "same.jsonl"
            with self.assertRaisesRegex(SystemExit, "path collision"):
                adapter_main(
                    [
                        "--dataset",
                        "tempcompass",
                        "--annotations",
                        str(annotation),
                        "--media-root",
                        str(media_root),
                        "--output",
                        str(same_output),
                        "--report-output",
                        str(same_output),
                    ]
                )
            with self.assertRaisesRegex(SystemExit, "path collision"):
                adapter_main(
                    [
                        "--dataset",
                        "tempcompass",
                        "--annotations",
                        str(annotation),
                        "--media-root",
                        str(media_root),
                        "--output",
                        str(annotation),
                        "--overwrite",
                    ]
                )

    def test_tempcompass_embedded_choices_pair_and_meta_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "media"
            _media(media_root / "100.mp4")
            _media(media_root / "100_reverse.mp4")
            questions = {
                "100": {
                    "direction": [
                        {
                            "question": "Which direction does the ball move?\nA. right to left\nB. left to right",
                            "answer": "B. left to right",
                        }
                    ]
                },
                "100_reverse": {
                    "direction": [
                        {
                            "question": "Which direction does the ball move?\nA. right to left\nB. left to right",
                            "answer": "A. right to left",
                        }
                    ]
                },
            }
            meta = {
                "100": {
                    "eval_dim": {
                        "action": None,
                        "speed": None,
                        "direction": {
                            "subject": "ball",
                            "direction": "left to right",
                            "type": "object motion",
                        },
                        "order": None,
                        "attribute_change": None,
                    },
                    "content": ["artifacts"],
                },
                "100_reverse": {
                    "eval_dim": {
                        "action": None,
                        "speed": None,
                        "direction": {
                            "subject": "ball",
                            "direction": "right to left",
                            "type": "object motion",
                        },
                        "order": None,
                        "attribute_change": None,
                    },
                    "content": ["artifacts"],
                },
            }
            question_path = _json(root / "multi-choice.json", questions)
            meta_path = _json(root / "meta_info.json", meta)
            rows = load_records(
                "tempcompass",
                question_path,
                media_root,
                meta_info_path=meta_path,
            )
            self.assertEqual([row["answer"] for row in rows], ["B", "A"])
            self.assertEqual(
                {row["diagnostic"]["pair_role"] for row in rows},
                {"original", "counterfactual"},
            )
            self.assertEqual(len({row["diagnostic"]["pair_id"] for row in rows}), 1)
            self.assertEqual(
                {row["diagnostic"]["resampling_unit_id"] for row in rows},
                {"tempcompass:video_family:100"},
            )
            self.assertIn(
                "left to right",
                rows[0]["diagnostic"]["oracles"]["relations"][0]["text"],
            )
            self.assertEqual(
                rows[0]["diagnostic"]["oracles"]["relations"][0]["access"],
                "target_semantic",
            )
            self.assertTrue(validate_manifest(rows, require_media=True)["valid"])

    def test_tvbench_text_answer_and_accurate_localization_span(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "release"
            _media(media_root / "video" / "action_localization" / "clip.mp4")
            annotation = _json(
                root / "action_localization.json",
                [
                    {
                        "video": "clip.mp4",
                        "question": "When does the action occur?",
                        "candidates": [
                            "Before the marked interval",
                            "During the marked interval",
                        ],
                        "answer": "During the marked interval",
                        "start": 1.0,
                        "end": 3.0,
                        "accurate_start": 1.25,
                        "accurate_end": 2.75,
                    }
                ],
            )
            row = load_records(
                "tvbench", annotation, media_root, task="action_localization"
            )[0]
            self.assertEqual(row["answer"], "B")
            self.assertEqual(
                row["diagnostic"]["evidence_spans"],
                [
                    {
                        "start": 1.25,
                        "end": 2.75,
                        "unit": "seconds",
                        "role": "dataset_temporal_annotation",
                    }
                ],
            )
            self.assertEqual(
                row["diagnostic"]["provenance"]["answer_index_base"], "text"
            )
            self.assertEqual(
                row["diagnostic"]["resampling_unit_id"],
                "tvbench:video:clip.mp4",
            )

    def test_perception_test_zero_based_answer_and_gt_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "media"
            _media(media_root / "video_1.mp4")
            annotation = _json(
                root / "valid.json",
                {
                    "video_1": {
                        "metadata": {
                            "split": "valid",
                            "video_id": "video_1",
                            "frame_rate": 30.0,
                            "num_frames": 90,
                            "resolution": [480, 640],
                            "audio_samples": 0,
                            "audio_sample_rate": 48000.0,
                            "is_cup_game": 0,
                            "is_camera_moving": 0,
                        },
                        "object_tracking": [
                            {
                                "id": 0,
                                "label": "red cup",
                                "is_occluder": 0,
                                "bounding_boxes": [
                                    [0.1, 0.2, 0.3, 0.4],
                                    [0.2, 0.2, 0.4, 0.4],
                                    [0.3, 0.2, 0.5, 0.4],
                                ],
                                "initial_tracking_box": [1, 0, 0],
                                "frame_ids": [0, 30, 75],
                                "timestamps": [0, 1_000_000, 2_500_000],
                                "is_masked": [0, 0, 0],
                            }
                        ],
                        "point_tracking": [],
                        "action_localisation": [
                            {
                                "id": 0,
                                "label_id": 23,
                                "label": "Opening something",
                                "parent_objects": [0],
                                "timestamps": [100_000, 900_000],
                                "frame_ids": [3, 27],
                            },
                            {
                                "id": 1,
                                "label_id": 24,
                                "label": "Closing something",
                                "parent_objects": [0],
                                "timestamps": [2_166_667, 2_666_667],
                                "frame_ids": [65, 80],
                            },
                            {
                                "id": 2,
                                "label_id": 25,
                                "label": "Touching the cut boundary",
                                "parent_objects": [0],
                                "timestamps": [1_666_667, 2_000_000],
                                "frame_ids": [50, 60],
                            },
                        ],
                        "sound_localisation": [],
                        "mc_question": [
                            {
                                "id": 7,
                                "question": "What does the person open?",
                                "options": ["a door", "a red cup", "a window"],
                                "answer_id": 1,
                                "area": "semantics",
                                "reasoning": "descriptive",
                                "tag": ["action recognition", "distractor object"],
                            }
                        ],
                        "grounded_question": [],
                    }
                },
            )
            # The official map can request a cut even when is_cup_game=0.
            cut_mapping = _json(root / "cut_frame_mapping_valid.json", {"video_1": 60})
            row = load_records(
                "perception_test",
                annotation,
                media_root,
                split="validation",
                cut_frame_mapping_path=cut_mapping,
            )[0]
            self.assertEqual(row["answer"], "B")
            self.assertEqual(row["diagnostic"]["provenance"]["answer_index_base"], 0)
            self.assertEqual(
                row["diagnostic"]["oracles"]["unordered_events"][0]["label"],
                "Opening something",
            )
            self.assertEqual(len(row["diagnostic"]["oracles"]["unordered_events"]), 1)
            self.assertNotIn(
                "Touching the cut boundary",
                {
                    event["label"]
                    for event in row["diagnostic"]["oracles"]["unordered_events"]
                },
            )
            self.assertEqual(
                row["diagnostic"]["oracles"]["ordered_events"][0]["start"], 0.1
            )
            self.assertEqual(
                row["diagnostic"]["oracles"]["relations"][0]["frame_ids"], [0, 30]
            )
            self.assertEqual(row["diagnostic"]["media_clip"]["end_frame_exclusive"], 60)
            self.assertEqual(
                row["diagnostic"]["media_clip"]["expected_total_frames"], 90
            )
            self.assertEqual(row["diagnostic"]["media_clip"]["unit"], "frames")
            self.assertFalse(validate_record(row, require_media=True))
            self.assertEqual(
                row["diagnostic"]["resampling_unit_id"],
                "perception_test:video:video_1",
            )

    def test_next_gqa_answer_text_and_multiple_evidence_spans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "videos"
            _media(media_root / "0101" / "4882821564.mp4")
            csv_path = root / "val.csv"
            columns = [
                "video_id",
                "frame_count",
                "width",
                "height",
                "question",
                "answer",
                "qid",
                "type",
                "a0",
                "a1",
                "a2",
                "a3",
                "a4",
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "video_id": "4882821564",
                        "frame_count": "2697",
                        "width": "640",
                        "height": "480",
                        "question": "Why did the boy move to the sofa?",
                        "answer": "unwrap it",
                        "qid": "1",
                        "type": "CW",
                        "a0": "share with the girl",
                        "a1": "approach the lady",
                        "a2": "unwrap it",
                        "a3": "play with a train",
                        "a4": "gesture something",
                    }
                )
            grounding = _json(
                root / "gsub_val.json",
                {
                    "4882821564": {
                        "duration": 10.0,
                        "fps": 30.0,
                        # The official grounding release contains a handful of
                        # -0.1/-0.2 starts from decimal timestamp rounding.
                        "location": {"1": [[-0.1, 2.5], [5.0, 7.0]]},
                    }
                },
            )
            video_map = _json(
                root / "map_vid_vidorID.json", {"4882821564": "0101/4882821564"}
            )
            row = load_records(
                "next_gqa",
                csv_path,
                media_root,
                grounding_path=grounding,
                video_map_path=video_map,
                source_split="val",
            )[0]
            self.assertEqual(row["answer"], "C")
            self.assertEqual(len(row["diagnostic"]["evidence_spans"]), 2)
            self.assertEqual(row["diagnostic"]["evidence_spans"][0]["start"], 0.0)
            self.assertEqual(
                row["diagnostic"]["evidence_spans"][0]["source_start"], -0.1
            )
            self.assertEqual(row["diagnostic"]["evidence_spans"][1]["end"], 7.0)
            self.assertEqual(
                row["diagnostic"]["resampling_unit_id"],
                "next_gqa:video:0101/4882821564",
            )

    def test_clevrer_multilabel_expands_to_binary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "media"
            _media(media_root / "video_10000.mp4")
            annotation = _json(
                root / "validation.json",
                [
                    {
                        "scene_index": 10000,
                        "video_filename": "video_10000.mp4",
                        "questions": [
                            {
                                "question_id": 11,
                                "question": "Which event caused the collision?",
                                "question_type": "explanatory",
                                "program": ["events", "filter_ancestor", "belong_to"],
                                "choices": [
                                    {
                                        "choice_id": 0,
                                        "choice": "The red sphere entered.",
                                        "program": ["objects", "red", "filter_in"],
                                        "answer": "correct",
                                    },
                                    {
                                        "choice_id": 1,
                                        "choice": "The cube stayed still.",
                                        "program": [
                                            "objects",
                                            "cube",
                                            "filter_stationary",
                                        ],
                                        "answer": "wrong",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            )
            scene_annotations = root / "processed_proposals"
            _json(
                scene_annotations / "sim_10000.json",
                {
                    "ground_truth": {
                        "objects": [
                            {
                                "id": 7,
                                "color": "red",
                                "material": "rubber",
                                "shape": "sphere",
                            },
                            {
                                "id": 2,
                                "color": "blue",
                                "material": "metal",
                                "shape": "cube",
                            },
                        ],
                        # Deliberately unsorted to exercise frame ordering.
                        "collisions": [
                            {"frame": 45, "object": [2, 7]},
                            {"frame": 12, "object": [7, 2]},
                        ],
                    },
                    "frames": [],
                },
            )
            rows = load_records(
                "clevrer",
                annotation,
                media_root,
                source_split="validation",
                scene_annotations_path=scene_annotations,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["answer"] for row in rows], ["B", "A"])
            self.assertEqual(rows[0]["choices"], ["No", "Yes"])
            self.assertEqual(
                rows[0]["diagnostic"]["question_family"],
                rows[1]["diagnostic"]["question_family"],
            )
            self.assertEqual(
                rows[0]["diagnostic"]["question_family"], "clevrer:explanatory"
            )
            self.assertEqual(
                rows[0]["diagnostic"]["independent_unit_id"],
                rows[1]["diagnostic"]["independent_unit_id"],
            )
            self.assertEqual(
                rows[0]["diagnostic"]["independent_unit_id"],
                "clevrer:10000:11",
            )
            self.assertEqual(
                {row["diagnostic"]["resampling_unit_id"] for row in rows},
                {"clevrer:scene:10000"},
            )
            self.assertEqual(
                {row["diagnostic"]["official_candidate_id"] for row in rows},
                {"0", "1"},
            )
            self.assertEqual(
                {row["diagnostic"]["official_candidate_count"] for row in rows},
                {2},
            )
            operator = rows[0]["diagnostic"]["oracles"]["operator"][0]
            self.assertEqual(
                operator["composition"], "choice_program + question_program"
            )
            operator_clue, operator_dose = render_clue(rows[0], ("operator",), "all")
            self.assertEqual(operator_dose, 1)
            self.assertIn(
                "candidate program: objects -> red -> filter in", operator_clue
            )
            self.assertIn(
                "question program: events -> filter ancestor -> belong to",
                operator_clue,
            )
            self.assertIn(
                "composition rule: choice_program + question_program",
                operator_clue,
            )
            self.assertNotIn("official_adapter_annotation", operator_clue)
            self.assertEqual(
                [
                    event["frame"]
                    for event in rows[0]["diagnostic"]["oracles"]["ordered_events"]
                ],
                [12, 45],
            )
            unordered_collisions = rows[0]["diagnostic"]["oracles"]["unordered_events"]
            ordered_collisions = rows[0]["diagnostic"]["oracles"]["ordered_events"]
            self.assertEqual(len(unordered_collisions), 2)
            self.assertTrue(all("frame" not in event for event in unordered_collisions))
            self.assertTrue(
                all("At frame" not in event["text"] for event in unordered_collisions)
            )
            self.assertEqual(
                {event["event_id"] for event in unordered_collisions},
                {event["event_id"] for event in ordered_collisions},
            )
            self.assertIn(
                "red rubber sphere",
                rows[0]["diagnostic"]["oracles"]["ordered_events"][0]["text"],
            )
            self.assertEqual(len(rows[0]["diagnostic"]["oracles"]["static_facts"]), 2)
            self.assertEqual(
                rows[0]["diagnostic"]["provenance"]["scene_annotations_file"],
                str((scene_annotations / "sim_10000.json").resolve()),
            )

    def test_egoschema_public_answer_is_zero_based_and_has_no_fake_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "egoschema"
            release.mkdir()
            uid = "001934bb-81bd-4cd8-a574-0472ef3f6678"
            _json(
                release / "questions.json",
                [
                    {
                        "q_uid": uid,
                        "google_drive_id": "drive-id",
                        "question": "What interrupts the recurring activity?",
                        "option 0": "A phone notification.",
                        "option 1": "A conversation.",
                        "option 2": "A drink.",
                        "option 3": "Typing on a laptop.",
                        "option 4": "Nothing changes.",
                    }
                ],
            )
            _json(release / "subset_answers.json", {uid: 3})
            media_root = root / "videos"
            _media(media_root / f"{uid}.mp4")
            row = load_records("egoschema", release, media_root, split="public_500")[0]
            self.assertEqual(row["answer"], "D")
            self.assertEqual(row["diagnostic"]["evidence_spans"], [])
            self.assertTrue(
                all(
                    value in ([], None, False)
                    for value in row["diagnostic"]["oracles"].values()
                )
            )
            self.assertEqual(
                row["diagnostic"]["resampling_unit_id"],
                f"egoschema:video:{uid}",
            )

    def test_mvp_pair_invariants_and_opposite_semantic_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "videos"
            _media(media_root / "star" / "A.mp4")
            _media(media_root / "star" / "B.mp4")
            annotation = _json(
                root / "mini.json",
                [
                    {
                        "video_id": "star_A_B_question_1",
                        "video_path": "star/A.mp4",
                        "question": "Which object was placed after the clothes were tidied?",
                        "answer": "The blanket.",
                        "candidates": "['The blanket.', 'The bag.']",
                        "source": "star",
                    },
                    {
                        "video_id": "star_A_B_question_2",
                        "video_path": "star/B.mp4",
                        "question": "Which object was placed after the clothes were tidied?",
                        "answer": "The bag.",
                        "candidates": "['The blanket.', 'The bag.']",
                        "source": "star",
                    },
                ],
            )
            rows = load_records(
                "mvp",
                annotation,
                media_root,
                split="mini",
                category="temporal_reasoning",
            )
            self.assertEqual([row["answer"] for row in rows], ["A", "B"])
            self.assertEqual(len({row["diagnostic"]["pair_id"] for row in rows}), 1)
            self.assertEqual(
                len({row["diagnostic"]["resampling_unit_id"] for row in rows}),
                1,
            )
            self.assertEqual(
                [row["diagnostic"]["pair_role"] for row in rows],
                ["original", "counterfactual"],
            )
            self.assertTrue(validate_manifest(rows, require_media=True)["valid"])

    def test_media_resolution_rejects_ambiguous_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _media(root / "one" / "same.mp4")
            _media(root / "two" / "same.mp4")
            with self.assertRaisesRegex(AdapterError, "ambiguous media resolution"):
                resolve_media(root, ("same.mp4",), search_basename="same.mp4")

    def test_exclusion_manifest_skips_malformed_row_and_is_fully_auditable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "release"
            _media(media_root / "video" / "action_localization" / "clip.mp4")
            annotation = _json(
                root / "action_localization.json",
                [
                    {
                        "question_id": "known-malformed",
                        "video": "clip.mp4",
                        "question": "Which interval contains the event?",
                        "candidates": ["During it", "During it"],
                        "answer": "During it",
                    },
                    {
                        "question_id": "valid-row",
                        "video": "clip.mp4",
                        "question": "Which interval contains the event?",
                        "candidates": ["Before it", "During it"],
                        "answer": "During it",
                        "accurate_start": 1.0,
                        "accurate_end": 2.0,
                    },
                ],
            )
            with self.assertRaisesRegex(AdapterError, "duplicate normalized choices"):
                load_records(
                    "tvbench",
                    annotation,
                    media_root,
                    task="action_localization",
                )

            exclusion_path = _json(
                root / "exclusions.json",
                {
                    "exclusions": [
                        {
                            "dataset": "tvbench",
                            "source_id": "action_localization:known-malformed",
                            "reason": "Official release row has duplicate normalized candidates.",
                        },
                        {
                            "dataset": "egoschema",
                            "source_id": "unrelated-dataset-row",
                            "reason": "Dataset-scoped entry for a separate build.",
                        },
                    ]
                },
            )
            expected_sha256 = hashlib.sha256(exclusion_path.read_bytes()).hexdigest()
            rows = load_records(
                "tvbench",
                annotation,
                media_root,
                task="action_localization",
                exclusions_path=exclusion_path,
            )
            self.assertEqual(len(rows), 1)
            audit = rows[0]["diagnostic"]["provenance"]["exclusion_manifest"]
            self.assertEqual(audit["sha256"], expected_sha256)
            self.assertEqual(audit["selected_entries"], 1)
            self.assertEqual(
                audit["applied"],
                [
                    {
                        "dataset": "tvbench",
                        "source_id": "action_localization:known-malformed",
                        "reason": "Official release row has duplicate normalized candidates.",
                        "raw_location": f"{annotation.resolve()}[0]",
                    }
                ],
            )

            output = root / "manifest.jsonl"
            report_output = root / "build-report.json"
            with redirect_stdout(io.StringIO()):
                return_code = adapter_main(
                    [
                        "--dataset",
                        "tvbench",
                        "--annotations",
                        str(annotation),
                        "--media-root",
                        str(media_root),
                        "--output",
                        str(output),
                        "--report-output",
                        str(report_output),
                        "--task",
                        "action_localization",
                        "--exclusions",
                        str(exclusion_path),
                    ]
                )
            self.assertEqual(return_code, 0)
            report = json.loads(report_output.read_text(encoding="utf-8"))
            self.assertEqual(report["exclusions"], audit)
            self.assertFalse(report["confirmatory_eligible"])
            self.assertIn(
                "tvbench_single_task_file",
                report["confirmatory_eligibility_issues"],
            )
            self.assertTrue(report["adapter_run_id"].startswith("adapter-run::"))
            self.assertEqual(report["source_roles"], ["annotations", "exclusions"])
            self.assertEqual(
                report["source_checksums_sha256"][str(exclusion_path.resolve())],
                expected_sha256,
            )
            written_row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(
                written_row["diagnostic"]["provenance"]["exclusion_manifest"],
                audit,
            )

    def test_exclusion_manifest_rejects_unused_and_ambiguous_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_root = root / "release"
            _media(media_root / "video" / "action_localization" / "clip.mp4")
            valid_row = {
                "question_id": "same-id",
                "video": "clip.mp4",
                "question": "When does it happen?",
                "candidates": ["Before", "During"],
                "answer": "During",
                "accurate_start": 1.0,
                "accurate_end": 2.0,
            }
            annotation = _json(root / "action_localization.json", [valid_row])
            unused = _json(
                root / "unused.jsonl",
                {
                    "dataset": "tvbench",
                    "source_id": "action_localization:not-present",
                    "reason": "Expected anomaly is absent in this release checksum.",
                },
            )
            with self.assertRaisesRegex(AdapterError, "unused exclusions"):
                load_records(
                    "tvbench",
                    annotation,
                    media_root,
                    task="action_localization",
                    exclusions_path=unused,
                )

            duplicate_manifest = _json(
                root / "duplicate-exclusions.json",
                [
                    {
                        "dataset": "tvbench",
                        "source_id": "action_localization:same-id",
                        "reason": "First declaration.",
                    },
                    {
                        "dataset": "tvbench",
                        "source_id": "action_localization:same-id",
                        "reason": "Conflicting repeated declaration.",
                    },
                ],
            )
            with self.assertRaisesRegex(AdapterError, "duplicate exclusion"):
                load_records(
                    "tvbench",
                    annotation,
                    media_root,
                    task="action_localization",
                    exclusions_path=duplicate_manifest,
                )

            duplicate_rows = _json(
                root / "duplicate-source.json",
                [valid_row, dict(valid_row)],
            )
            ambiguous = _json(
                root / "ambiguous.json",
                [
                    {
                        "dataset": "tvbench",
                        "source_id": "action_localization:same-id",
                        "reason": "This key must identify exactly one raw row.",
                    }
                ],
            )
            with self.assertRaisesRegex(
                AdapterError, "ambiguous exclusion matched multiple raw rows"
            ):
                load_records(
                    "tvbench",
                    duplicate_rows,
                    media_root,
                    task="action_localization",
                    exclusions_path=ambiguous,
                )

    def test_exclusions_must_close_over_paired_and_official_question_units(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            temp_media = root / "temp-media"
            _media(temp_media / "100.mp4")
            _media(temp_media / "100_reverse.mp4")
            temp_annotations = _json(
                root / "tempcompass.json",
                {
                    "100": {
                        "action": [
                            {
                                "question": "Which action occurs?\nA. opens\nB. closes",
                                "answer": "A. opens",
                            }
                        ]
                    },
                    "100_reverse": {
                        "action": [
                            {
                                "question": "Which action occurs?\nA. opens\nB. closes",
                                "answer": "B. closes",
                            }
                        ]
                    },
                },
            )
            temp_exclusions = _json(
                root / "temp-exclusions.json",
                {
                    "exclusions": [
                        {
                            "dataset": "tempcompass",
                            "source_id": "100:action:0",
                            "reason": "Exclude one corrupted member only.",
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(AdapterError, "must close over"):
                load_records(
                    "tempcompass",
                    temp_annotations,
                    temp_media,
                    exclusions_path=temp_exclusions,
                )

            clevrer_media = root / "clevrer-media"
            _media(clevrer_media / "video_10000.mp4")
            clevrer_annotations = _json(
                root / "clevrer.json",
                [
                    {
                        "scene_index": 10000,
                        "video_filename": "video_10000.mp4",
                        "questions": [
                            {
                                "question_id": 11,
                                "question_type": "explanatory",
                                "question": "What caused the collision?",
                                "program": [],
                                "choices": [
                                    {
                                        "choice_id": 0,
                                        "choice": "The red object moved.",
                                        "answer": "correct",
                                        "program": [],
                                    },
                                    {
                                        "choice_id": 1,
                                        "choice": "Nothing moved.",
                                        "answer": "wrong",
                                        "program": [],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            )
            clevrer_exclusions = _json(
                root / "clevrer-exclusions.json",
                {
                    "exclusions": [
                        {
                            "dataset": "clevrer",
                            "source_id": "10000:11:0",
                            "reason": "Exclude one candidate only.",
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(AdapterError, "must close over"):
                load_records(
                    "clevrer",
                    clevrer_annotations,
                    clevrer_media,
                    source_split="validation",
                    exclusions_path=clevrer_exclusions,
                )

            mvp_media = root / "mvp-media"
            _media(mvp_media / "A.mp4")
            _media(mvp_media / "B.mp4")
            mvp_annotations = _json(
                root / "mvp.json",
                [
                    {
                        "video_id": "family_1",
                        "video_path": "A.mp4",
                        "question": "Which event happened?",
                        "answer": "First",
                        "candidates": ["First", "Second"],
                    },
                    {
                        "video_id": "family_2",
                        "video_path": "B.mp4",
                        "question": "Which event happened?",
                        "answer": "Second",
                        "candidates": ["First", "Second"],
                    },
                ],
            )
            mvp_exclusions = _json(
                root / "mvp-exclusions.json",
                {
                    "exclusions": [
                        {
                            "dataset": "mvp",
                            "source_id": "family_1",
                            "reason": "Exclude one paired video only.",
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(AdapterError, "must close over"):
                load_records(
                    "mvp",
                    mvp_annotations,
                    mvp_media,
                    category="temporal_reasoning",
                    exclusions_path=mvp_exclusions,
                )


if __name__ == "__main__":
    unittest.main()
