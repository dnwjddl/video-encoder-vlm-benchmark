from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from information_upper_bound.media import (
    EvidenceSpan,
    EvidenceSpanError,
    VideoMetadata,
    ViewSamplingError,
    ViewSpec,
    _probe_decord,
    load_video_view,
    sample_frame_indices,
    validate_evidence_spans,
)


class EvidenceSpanValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = VideoMetadata(total_frames=100, fps=10.0, duration_sec=10.0)

    def test_seconds_and_normalized_spans_resolve_to_same_frames(self) -> None:
        seconds = validate_evidence_spans(
            [{"start_sec": 2.0, "end_sec": 4.0}],
            duration_sec=self.metadata.duration_sec,
            total_frames=self.metadata.total_frames,
            fps=self.metadata.fps,
        )
        normalized = validate_evidence_spans(
            [[0.2, 0.4]],
            duration_sec=self.metadata.duration_sec,
            total_frames=self.metadata.total_frames,
            fps=self.metadata.fps,
            default_unit="normalized",
        )
        self.assertEqual(
            (seconds[0].start_frame, seconds[0].end_frame_exclusive), (20, 40)
        )
        self.assertEqual(
            (seconds[0].start_frame, seconds[0].end_frame_exclusive),
            (normalized[0].start_frame, normalized[0].end_frame_exclusive),
        )
        self.assertEqual(normalized[0].start_sec, 2.0)
        self.assertEqual(normalized[0].end_sec, 4.0)

    def test_subframe_span_still_resolves_to_one_overlapping_frame(self) -> None:
        resolved = validate_evidence_spans(
            [{"start": 0.011, "end": 0.012, "unit": "seconds"}],
            duration_sec=self.metadata.duration_sec,
            total_frames=self.metadata.total_frames,
            fps=self.metadata.fps,
        )
        self.assertEqual(resolved[0].start_frame, 0)
        self.assertEqual(resolved[0].end_frame_exclusive, 1)

    def test_rejects_invalid_spans(self) -> None:
        invalid = (
            {"start": -0.1, "end": 0.2, "unit": "normalized"},
            {"start": 0.5, "end": 0.5, "unit": "normalized"},
            {"start": 0.8, "end": 1.1, "unit": "normalized"},
            {"start_sec": 9.0, "end_sec": 10.1},
            {"start_sec": 2.0},
        )
        for span in invalid:
            with self.subTest(span=span), self.assertRaises(EvidenceSpanError):
                validate_evidence_spans(
                    [span],
                    duration_sec=self.metadata.duration_sec,
                    total_frames=self.metadata.total_frames,
                    fps=self.metadata.fps,
                )

    def test_evidence_view_requires_annotation(self) -> None:
        for view in (
            "evidence_only",
            "evidence_present",
            "evidence_removed",
            "random_position_mask",
            "random_matched",
        ):
            with self.subTest(view=view), self.assertRaises(EvidenceSpanError):
                ViewSpec(view=view, num_frames=8)


class DeterministicFrameSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = VideoMetadata(total_frames=100, fps=10.0, duration_sec=10.0)
        self.evidence = (EvidenceSpan(2.0, 4.0, "seconds"),)

    def sample(
        self, view: str, *, num_frames: int = 5, seed: int = 7, visual_id: str = "v1"
    ):
        spans = (
            self.evidence
            if view
            in {
                "evidence_only",
                "evidence_present",
                "evidence_removed",
                "random_position_mask",
                "random_matched",
            }
            else ()
        )
        spec = ViewSpec(
            view=view, num_frames=num_frames, seed=seed, evidence_spans=spans
        )
        return sample_frame_indices(self.metadata, spec, visual_id=visual_id)

    def test_full_single_and_reverse_indices(self) -> None:
        self.assertEqual(self.sample("full").indices, (0, 25, 50, 74, 99))
        self.assertEqual(self.sample("single").indices, (50, 50, 50, 50, 50))
        self.assertEqual(self.sample("reverse").indices, (99, 74, 50, 25, 0))

    def test_declared_media_clip_is_applied_before_every_view(self) -> None:
        spec = ViewSpec.create(
            view="full",
            num_frames=5,
            clip={"start": 1.0, "end": 4.0, "unit": "seconds"},
        )
        selection = sample_frame_indices(self.metadata, spec, visual_id="cup-game")
        self.assertEqual(selection.indices[0], 10)
        self.assertLess(selection.indices[-1], 40)

    def test_shuffle_is_stable_and_keyed_by_visual_identity(self) -> None:
        first = self.sample("shuffle", num_frames=10, seed=91, visual_id="video-a")
        repeated = self.sample("shuffle", num_frames=10, seed=91, visual_id="video-a")
        other_visual = self.sample(
            "shuffle", num_frames=10, seed=91, visual_id="video-b"
        )
        self.assertEqual(first.indices, repeated.indices)
        self.assertNotEqual(first.indices, other_visual.indices)
        self.assertCountEqual(first.indices, self.sample("full", num_frames=10).indices)

    def test_evidence_present_and_removed_share_grid_and_mask_only_evidence(
        self,
    ) -> None:
        evidence_only = self.sample("evidence_only", num_frames=5)
        evidence_present = self.sample("evidence_present", num_frames=10)
        evidence_removed = self.sample("evidence_removed", num_frames=10)
        random_mask = self.sample("random_position_mask", num_frames=10)
        self.assertEqual(evidence_only.indices, (20, 25, 30, 34, 39))
        self.assertTrue(all(20 <= index < 40 for index in evidence_only.indices))
        self.assertEqual(evidence_present.indices, evidence_removed.indices)
        self.assertEqual(evidence_present.indices, random_mask.indices)
        self.assertEqual(evidence_present.masked_input_positions, ())
        self.assertTrue(evidence_removed.masked_input_positions)
        self.assertEqual(
            len(random_mask.masked_input_positions),
            len(evidence_removed.masked_input_positions),
        )
        self.assertFalse(
            set(random_mask.masked_input_positions)
            & set(evidence_removed.masked_input_positions)
        )
        self.assertTrue(
            all(
                20 <= evidence_removed.indices[position] < 40
                for position in evidence_removed.masked_input_positions
            )
        )
        self.assertEqual(
            evidence_removed.mask_strategy,
            "solid_midgray_rgb_on_shared_evidence_grid",
        )
        random_metadata = random_mask.to_dict()
        self.assertEqual(random_metadata["mask_evidence_overlap_count"], 0)
        self.assertEqual(
            random_metadata["mask_non_evidence_count"],
            len(evidence_removed.masked_input_positions),
        )
        self.assertEqual(evidence_only.evidence_source_frame_count, 20)
        self.assertEqual(evidence_only.candidate_source_frame_count, 20)

    def test_random_position_mask_records_unavoidable_evidence_overlap(self) -> None:
        metadata = VideoMetadata(total_frames=10, fps=1.0, duration_sec=10.0)
        spec = ViewSpec(
            view="random_position_mask",
            num_frames=5,
            seed=3,
            evidence_spans=(EvidenceSpan(0.0, 9.0, "seconds"),),
        )
        selection = sample_frame_indices(metadata, spec, visual_id="dense-evidence")
        metadata_row = selection.to_dict()
        self.assertEqual(metadata_row["mask_target_count"], 4)
        self.assertEqual(metadata_row["mask_non_evidence_count"], 1)
        self.assertEqual(metadata_row["mask_evidence_overlap_count"], 3)

    def test_evidence_removed_decodes_same_grid_then_masks_declared_positions(
        self,
    ) -> None:
        spec = ViewSpec(
            view="evidence_removed",
            num_frames=5,
            evidence_spans=self.evidence,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"placeholder")

            def fake_decode(_path, indices, *, metadata, backend):
                return (
                    [
                        Image.new("RGB", (2, 2), color=(index, 0, 0))
                        for index in indices
                    ],
                    metadata,
                )

            with (
                patch(
                    "information_upper_bound.media.probe_video",
                    return_value=self.metadata,
                ),
                patch(
                    "information_upper_bound.media.decode_selected_frames",
                    side_effect=fake_decode,
                ),
            ):
                decoded = load_video_view(
                    path, spec, visual_id="mask-test", backend="decord"
                )
        masked = set(decoded.selection.masked_input_positions)
        self.assertTrue(masked)
        for position, frame in enumerate(decoded.frames):
            expected = (
                (128, 128, 128)
                if position in masked
                else (
                    decoded.selection.indices[position],
                    0,
                    0,
                )
            )
            self.assertEqual(frame.getpixel((0, 0)), expected)

    def test_random_matched_is_deterministic_disjoint_and_coverage_matched(
        self,
    ) -> None:
        first = self.sample("random_matched", num_frames=8, seed=123)
        repeated = self.sample("random_matched", num_frames=8, seed=123)
        self.assertEqual(first.indices, repeated.indices)
        self.assertTrue(all(not 20 <= index < 40 for index in first.indices))
        self.assertEqual(first.evidence_source_frame_count, 20)
        self.assertGreater(first.candidate_source_frame_count, 0)
        self.assertEqual(
            first.random_match_strategy,
            "contiguous_non_evidence_duration_window",
        )
        self.assertAlmostEqual(first.random_match_target_duration_sec, 2.0)
        self.assertAlmostEqual(first.random_match_actual_duration_sec, 2.0)
        self.assertLess(first.random_match_error_sec, 1e-9)

    def test_random_matched_combines_disjoint_time_support_when_needed(self) -> None:
        metadata = VideoMetadata(total_frames=10, fps=1.0, duration_sec=10.0)
        spans = tuple(
            EvidenceSpan(float(index), float(index + 1)) for index in range(0, 10, 2)
        )
        spec = ViewSpec(
            view="random_matched",
            num_frames=5,
            seed=19,
            evidence_spans=spans,
        )
        selected = sample_frame_indices(metadata, spec, visual_id="alternating")
        self.assertEqual(selected.indices, (1, 3, 5, 7, 9))
        self.assertEqual(selected.evidence_source_frame_count, 5)
        self.assertEqual(selected.candidate_source_frame_count, 5)
        self.assertEqual(
            selected.random_match_strategy,
            "multi_span_non_evidence_duration_match",
        )
        self.assertAlmostEqual(selected.random_match_target_duration_sec, 5.0)
        self.assertAlmostEqual(selected.random_match_actual_duration_sec, 5.0)

    def test_random_matched_rejects_insufficient_control_coverage(self) -> None:
        spec = ViewSpec(
            view="random_matched",
            num_frames=5,
            evidence_spans=(EvidenceSpan(0.0, 8.0),),
        )
        with self.assertRaises(ViewSamplingError):
            sample_frame_indices(self.metadata, spec, visual_id="mostly-evidence")

    def test_sampling_metadata_contains_reproducible_indices_and_timestamps(
        self,
    ) -> None:
        selection = self.sample("evidence_only", num_frames=3)
        payload = selection.to_dict()
        self.assertEqual(payload["selected_indices"], [20, 30, 39])
        self.assertEqual(payload["selected_timestamps_sec"], [2.0, 3.0, 3.9])
        self.assertEqual(payload["video"]["total_frames"], 100)
        self.assertEqual(payload["resolved_evidence_spans"][0]["start_frame"], 20)
        self.assertEqual(
            payload["resolved_evidence_spans"][0]["end_frame_exclusive"], 40
        )


class VariableFrameRateTimestampTest(unittest.TestCase):
    def setUp(self) -> None:
        # Presentation intervals deliberately differ enough from average-FPS
        # inference that a regression back to index/fps changes the result.
        self.metadata = VideoMetadata(
            total_frames=5,
            fps=12.0,
            duration_sec=0.44,
            backend="decord",
            frame_intervals_sec=(
                (0.00, 0.04),
                (0.04, 0.12),
                (0.12, 0.16),
                (0.16, 0.40),
                (0.40, 0.44),
            ),
            timestamp_source="decord_frame_timestamp",
        )

    def test_evidence_overlap_uses_decoder_frame_intervals(self) -> None:
        spec = ViewSpec(
            view="evidence_only",
            num_frames=3,
            evidence_spans=(EvidenceSpan(0.10, 0.18),),
        )
        selection = sample_frame_indices(self.metadata, spec, visual_id="vfr-evidence")
        self.assertEqual(selection.indices, (1, 2, 3))
        self.assertEqual(selection.timestamps_sec, (0.04, 0.12, 0.16))
        self.assertEqual(selection.resolved_evidence_spans[0].start_frame, 1)
        self.assertEqual(selection.resolved_evidence_spans[0].end_frame_exclusive, 4)

    def test_explicit_clip_uses_decoder_frame_intervals(self) -> None:
        spec = ViewSpec.create(
            view="full",
            num_frames=3,
            clip={"start": 0.10, "end": 0.17, "unit": "seconds"},
        )
        selection = sample_frame_indices(self.metadata, spec, visual_id="vfr-clip")
        self.assertEqual(selection.indices, (1, 2, 3))

    def test_official_frame_exclusive_clip_is_exact_under_vfr(self) -> None:
        spec = ViewSpec.create(
            view="full",
            num_frames=4,
            clip={
                "start_frame": 0,
                "end_frame_exclusive": 3,
                "expected_total_frames": 5,
                "unit": "frames",
            },
        )
        selection = sample_frame_indices(self.metadata, spec, visual_id="vfr-frame-cut")
        self.assertTrue(all(index < 3 for index in selection.indices))
        self.assertEqual(selection.indices[-1], 2)

        mismatched = ViewSpec.create(
            view="full",
            num_frames=2,
            clip={
                "end_frame_exclusive": 3,
                "expected_total_frames": 6,
                "unit": "frames",
            },
        )
        with self.assertRaisesRegex(ViewSamplingError, "frame count does not match"):
            sample_frame_indices(
                self.metadata, mismatched, visual_id="bad-frame-contract"
            )

    def test_random_control_matches_duration_not_vfr_frame_count(self) -> None:
        spec = ViewSpec(
            view="random_matched",
            num_frames=2,
            seed=7,
            evidence_spans=(EvidenceSpan(0.10, 0.18),),
        )
        selection = sample_frame_indices(self.metadata, spec, visual_id="vfr-control")
        self.assertEqual(selection.evidence_source_frame_count, 3)
        self.assertEqual(selection.candidate_source_frame_count, 2)
        self.assertEqual(selection.indices, (0, 4))
        self.assertAlmostEqual(selection.random_match_target_duration_sec, 0.08)
        self.assertAlmostEqual(selection.random_match_actual_duration_sec, 0.08)
        self.assertLess(selection.random_match_error_sec, 1e-9)

    def test_timestamp_table_is_digest_referenced_not_repeated_per_trial(self) -> None:
        payload = self.metadata.to_dict()
        self.assertTrue(payload["time_alignment_guaranteed"])
        self.assertEqual(payload["timestamp_source"], "decord_frame_timestamp")
        self.assertEqual(len(payload["frame_timestamp_sha256"]), 64)
        self.assertNotIn("frame_intervals_sec", payload)

    def test_invalid_timestamp_timeline_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nondecreasing presentation order"):
            VideoMetadata(
                total_frames=2,
                fps=10.0,
                duration_sec=0.2,
                frame_intervals_sec=((0.1, 0.2), (0.0, 0.1)),
            )

    def test_span_in_real_timestamp_gap_is_not_fabricated(self) -> None:
        metadata = VideoMetadata(
            total_frames=2,
            fps=2.0,
            duration_sec=1.0,
            frame_intervals_sec=((0.0, 0.2), (0.8, 1.0)),
        )
        spec = ViewSpec(
            view="evidence_only",
            num_frames=1,
            evidence_spans=(EvidenceSpan(0.4, 0.6),),
        )
        with self.assertRaisesRegex(EvidenceSpanError, "does not overlap"):
            sample_frame_indices(metadata, spec, visual_id="timestamp-gap")


class UnverifiedAverageFpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = VideoMetadata(
            total_frames=100,
            fps=10.0,
            duration_sec=10.0,
            backend="opencv",
            timestamp_source="opencv_average_fps_unverified",
        )

    def test_rejects_evidence_alignment(self) -> None:
        spec = ViewSpec(
            view="evidence_only",
            num_frames=4,
            evidence_spans=(EvidenceSpan(2.0, 3.0),),
        )
        with self.assertRaisesRegex(
            ViewSamplingError, "average FPS metadata is not sufficient"
        ):
            sample_frame_indices(self.metadata, spec, visual_id="opencv-evidence")

    def test_rejects_explicit_second_based_clip(self) -> None:
        spec = ViewSpec.create(
            view="full",
            num_frames=4,
            clip={"start": 1.0, "end": 2.0, "unit": "seconds"},
        )
        with self.assertRaisesRegex(
            ViewSamplingError, "average FPS metadata cannot guarantee"
        ):
            sample_frame_indices(self.metadata, spec, visual_id="opencv-clip")

    def test_allows_index_only_view_but_marks_timestamps_unverified(self) -> None:
        selection = sample_frame_indices(
            self.metadata,
            ViewSpec(view="full", num_frames=3),
            visual_id="opencv-full",
        )
        self.assertEqual(selection.indices, (0, 50, 99))
        self.assertFalse(selection.to_dict()["video"]["time_alignment_guaranteed"])


class DecoderTimestampProbeTest(unittest.TestCase):
    @staticmethod
    def fake_decord_module(*, timestamps, fps: float = 15.0):
        module = types.ModuleType("decord")

        class ArrayResult:
            def asnumpy(self):
                class ListResult:
                    def tolist(inner_self):
                        return timestamps

                return ListResult()

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return len(timestamps) if not isinstance(timestamps, Exception) else 3

            def get_avg_fps(self):
                return fps

            def get_frame_timestamp(self, indices):
                if isinstance(timestamps, Exception):
                    raise timestamps
                if indices != list(range(len(timestamps))):
                    raise AssertionError(
                        "probe did not request every source-frame timestamp"
                    )
                return ArrayResult()

        module.VideoReader = Reader
        module.cpu = lambda index: ("cpu", index)
        return module

    def test_decord_probe_uses_and_normalizes_decoder_timestamps(self) -> None:
        fake_decord = self.fake_decord_module(
            timestamps=((7.00, 7.04), (7.04, 7.14), (7.14, 7.20))
        )
        with patch.dict(sys.modules, {"decord": fake_decord}):
            metadata = _probe_decord(Path("decoder-probe.mp4"))
        self.assertEqual(metadata.timestamp_source, "decord_frame_timestamp")
        self.assertTrue(metadata.time_alignment_guaranteed)
        self.assertAlmostEqual(metadata.duration_sec, 0.20)
        for actual, expected in zip(
            metadata.frame_intervals_sec,
            ((0.00, 0.04), (0.04, 0.14), (0.14, 0.20)),
        ):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])

    def test_decord_without_timestamp_api_is_explicitly_unverified(self) -> None:
        fake_decord = self.fake_decord_module(
            timestamps=RuntimeError("timestamp API unavailable"),
            fps=25.0,
        )
        with patch.dict(sys.modules, {"decord": fake_decord}):
            metadata = _probe_decord(Path("decoder-probe.mp4"))
        self.assertEqual(metadata.timestamp_source, "decord_average_fps_unverified")
        self.assertFalse(metadata.time_alignment_guaranteed)
        self.assertIsNone(metadata.frame_intervals_sec)


if __name__ == "__main__":
    unittest.main()
