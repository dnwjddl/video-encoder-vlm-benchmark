from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from information_upper_bound.io import write_jsonl
from information_upper_bound.extract_features import (
    SCHEMA_VERSION,
    feature_artifact_identity,
    feature_content_hash,
    media_content_identity,
    media_fingerprint,
)
from information_upper_bound.integrity import tensor_identity
from information_upper_bound.io import sha256_file
from information_upper_bound.run import FeatureStore
from information_upper_bound.scoring import FrozenMultipleChoiceScorer, make_prompt


class PromptTests(unittest.TestCase):
    def _trial(self, channel: str) -> dict:
        return {
            "media_type": "video",
            "question": "What happened?",
            "choices": ["open", "close"],
            "clue_text": "- Timed event: opening at t=1s",
            "condition": {"input_channel": channel},
        }

    def test_visual_and_text_channels_have_one_explicit_difference(self) -> None:
        visual_prefix, marker, suffix = make_prompt(self._trial("visual"))
        self.assertEqual(marker, "<VISUAL>")
        self.assertNotIn("Timed event", suffix)
        self.assertIn("Question: What happened?", suffix)

        visual_text_prefix, marker, suffix = make_prompt(
            self._trial("visual_plus_text")
        )
        self.assertEqual(marker, "<VISUAL>")
        self.assertIn("Timed event", suffix)

        text_prefix, marker, suffix = make_prompt(self._trial("text_oracle"))
        self.assertIsNone(marker)
        self.assertIn("Timed event", suffix)

        embedding_prefix, marker, suffix = make_prompt(self._trial("embedding_oracle"))
        self.assertEqual(marker, "<VISUAL>")
        self.assertNotIn("Timed event", suffix)

        question_prefix, marker, suffix = make_prompt(self._trial("question_only"))
        self.assertIsNone(marker)
        self.assertNotIn("Timed event", suffix)
        self.assertEqual(
            {
                visual_prefix,
                visual_text_prefix,
                text_prefix,
                embedding_prefix,
                question_prefix,
            },
            {visual_prefix},
        )

    def test_sequence_nll_sums_only_answer_positions(self) -> None:
        logits = torch.zeros(1, 5, 3)
        labels = torch.tensor([[-100, -100, -100, 2, 1]])
        # Causal shifts: logits at positions 2 and 3 predict labels 2 and 1.
        logits[0, 2, 2] = 10.0
        logits[0, 3, 1] = 10.0
        good = FrozenMultipleChoiceScorer._sequence_nll(logits, labels)
        logits[0, 2, 2] = -10.0
        bad = FrozenMultipleChoiceScorer._sequence_nll(logits, labels)
        self.assertLess(float(good.item()), float(bad.item()))
        self.assertEqual(tuple(good.shape), (1,))


class FeatureStoreTests(unittest.TestCase):
    def _write_feature(self, root: Path, *, visual_id: str = "visual::1") -> dict:
        features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        encoder_config = {"model_id": "encoder-a", "num_frames": 8}
        media_identity = {"sha256": "a" * 64, "size_bytes": 17}
        extraction_identity = {
            "pipeline": {
                "compute_dtype_requested": "bf16",
                "encoder_pretrained_identity": {"identity_sha256": "b" * 64},
            },
            "decoder": {"actual_backend": "decord", "implementation_version": "0.6.0"},
            "identity_sha256": "c" * 64,
        }
        decoded_identity = {
            "sha256": "d" * 64,
            "num_frames": 8,
            "shapes": [[224, 224, 3]] * 8,
            "pixel_format": "RGB_uint8",
        }
        sampling = {
            "video": {
                "backend": "decord",
                "timestamp_source": "decord_frame_timestamp",
            },
            "selected_indices": list(range(8)),
        }
        view_hash = "e" * 64
        feature_hash = feature_content_hash(
            view_hash,
            encoder_config=encoder_config,
            media_identity=media_identity,
            extraction_identity=extraction_identity,
            decoded_frame_identity=decoded_identity,
            sampling_identity=sampling,
        )
        feature_tensor_identity = tensor_identity(features)
        artifact_identity = feature_artifact_identity(
            visual_id=visual_id,
            view_content_hash_value=view_hash,
            feature_content_hash_value=feature_hash,
            encoder_config=encoder_config,
            extraction_identity=extraction_identity,
            media_identity=media_identity,
            decoded_frame_identity=decoded_identity,
            sampling=sampling,
            feature_tensor_identity=feature_tensor_identity,
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "visual_id": visual_id,
            "view_content_hash": view_hash,
            "feature_content_hash": feature_hash,
            "encoder_config": encoder_config,
            "extraction_identity": extraction_identity,
            "media_content_identity": media_identity,
            "decoded_frame_identity": decoded_identity,
            "sampling": sampling,
            "feature_tensor_identity": feature_tensor_identity,
            "feature_artifact_identity_sha256": artifact_identity,
            "features": features,
        }
        feature_path = root / f"{visual_id.replace(':', '_')}.pt"
        torch.save(artifact, feature_path)
        return {
            **{key: artifact[key] for key in FeatureStore._ARTIFACT_MATCH_FIELDS},
            "visual_id": visual_id,
            "feature_path": feature_path.name,
            "feature_file_sha256": sha256_file(feature_path),
            "shape": list(features.shape),
        }

    def test_confirmatory_media_fingerprint_includes_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"not-a-real-video-but-content-addressable")
            fingerprint = media_fingerprint(path, include_sha256=True)
            self.assertEqual(len(fingerprint["sha256"]), 64)
            self.assertEqual(fingerprint["size_bytes"], path.stat().st_size)

    def test_confirmatory_content_identity_ignores_mtime_when_sha_is_present(
        self,
    ) -> None:
        first = {"sha256": "a" * 64, "size_bytes": 17, "mtime_ns": 1}
        second = {"sha256": "a" * 64, "size_bytes": 17, "mtime_ns": 999}
        self.assertEqual(media_content_identity(first), media_content_identity(second))

    def test_development_content_identity_tracks_mtime_without_sha(self) -> None:
        first = {"size_bytes": 17, "mtime_ns": 1}
        second = {"size_bytes": 17, "mtime_ns": 999}
        self.assertNotEqual(
            media_content_identity(first), media_content_identity(second)
        )

    def test_feature_content_hash_binds_media_bytes_and_encoder_config(self) -> None:
        base = feature_content_hash(
            "view-hash",
            encoder_config={"model_id": "encoder-a", "num_frames": 8},
            media_identity={"sha256": "a" * 64, "size_bytes": 17},
            extraction_identity={"dtype": "bf16", "backend": "decord"},
            decoded_frame_identity={"sha256": "c" * 64},
            sampling_identity={"selected_indices": [0, 1]},
        )
        changed_media = feature_content_hash(
            "view-hash",
            encoder_config={"model_id": "encoder-a", "num_frames": 8},
            media_identity={"sha256": "b" * 64, "size_bytes": 17},
            extraction_identity={"dtype": "bf16", "backend": "decord"},
            decoded_frame_identity={"sha256": "c" * 64},
            sampling_identity={"selected_indices": [0, 1]},
        )
        changed_encoder = feature_content_hash(
            "view-hash",
            encoder_config={"model_id": "encoder-a", "num_frames": 16},
            media_identity={"sha256": "a" * 64, "size_bytes": 17},
            extraction_identity={"dtype": "bf16", "backend": "decord"},
            decoded_frame_identity={"sha256": "c" * 64},
            sampling_identity={"selected_indices": [0, 1]},
        )
        changed_dtype_backend = feature_content_hash(
            "view-hash",
            encoder_config={"model_id": "encoder-a", "num_frames": 8},
            media_identity={"sha256": "a" * 64, "size_bytes": 17},
            extraction_identity={"dtype": "fp16", "backend": "opencv"},
            decoded_frame_identity={"sha256": "c" * 64},
            sampling_identity={"selected_indices": [0, 1]},
        )
        changed_pixels = feature_content_hash(
            "view-hash",
            encoder_config={"model_id": "encoder-a", "num_frames": 8},
            media_identity={"sha256": "a" * 64, "size_bytes": 17},
            extraction_identity={"dtype": "bf16", "backend": "decord"},
            decoded_frame_identity={"sha256": "f" * 64},
            sampling_identity={"selected_indices": [0, 1]},
        )
        changed_sampling = feature_content_hash(
            "view-hash",
            encoder_config={"model_id": "encoder-a", "num_frames": 8},
            media_identity={"sha256": "a" * 64, "size_bytes": 17},
            extraction_identity={"dtype": "bf16", "backend": "decord"},
            decoded_frame_identity={"sha256": "c" * 64},
            sampling_identity={"selected_indices": [1, 0]},
        )
        self.assertEqual(len(base), 64)
        self.assertNotEqual(base, changed_media)
        self.assertNotEqual(base, changed_encoder)
        self.assertNotEqual(base, changed_dtype_backend)
        self.assertNotEqual(base, changed_pixels)
        self.assertNotEqual(base, changed_sampling)

    def test_relative_feature_path_and_unique_visual_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_feature(root)
            write_jsonl(root / "index.jsonl", [row])
            store = FeatureStore(root / "index.jsonl")
            self.assertEqual(tuple(store.load("visual::1").shape), (3, 4))
            with self.assertRaises(KeyError):
                store.load("visual::missing")

    def test_duplicate_visual_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.jsonl"
            root = path.parent
            first = self._write_feature(root, visual_id="same")
            second = dict(first)
            write_jsonl(
                path,
                [first, second],
            )
            with self.assertRaisesRegex(ValueError, "duplicate feature key"):
                FeatureStore(path)

    def test_feature_file_mutation_is_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_feature(root)
            write_jsonl(root / "index.jsonl", [row])
            (root / row["feature_path"]).write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                FeatureStore(root / "index.jsonl")

    def test_index_artifact_metadata_mismatch_is_rejected_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_feature(root)
            row["encoder_config"] = {"model_id": "other", "num_frames": 8}
            write_jsonl(root / "index.jsonl", [row])
            store = FeatureStore(root / "index.jsonl", verify_all_files=False)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                store.load("visual::1")


if __name__ == "__main__":
    unittest.main()
