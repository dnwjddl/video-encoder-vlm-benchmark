from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from information_upper_bound.encoder_runtime import (
    EncoderConfig,
    prepare_encoder_runtime,
    resolve_encoder,
)


class EncoderRuntimeTests(unittest.TestCase):
    def test_registry_accepts_and_overrides_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "encoders.yaml"
            registry.write_text(
                "encoder-a:\n"
                "  family: video\n"
                "  model_id: owner/model\n"
                "  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
                encoding="utf-8",
            )
            config = resolve_encoder(
                "encoder-a",
                registry,
                overrides={"revision": "b" * 40, "num_frames": 16},
            )
            self.assertEqual(config.revision, "b" * 40)
            self.assertEqual(config.num_frames, 16)

    def test_revision_resolves_to_snapshot_without_changing_declared_identity(
        self,
    ) -> None:
        revision = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / revision
            snapshot.mkdir()
            config = EncoderConfig(
                name="encoder-a",
                family="video",
                model_id="owner/model",
                revision=revision,
            )
            with patch(
                "information_upper_bound.encoder_runtime._download_snapshot",
                return_value=str(snapshot),
            ) as download:
                runtime, declared = prepare_encoder_runtime(config)
            self.assertEqual(runtime.model_id, str(snapshot))
            self.assertEqual(declared["model_id"], "owner/model")
            self.assertEqual(declared["revision"], revision)
            download.assert_called_once_with(
                repo_id="owner/model",
                revision=revision,
                local_files_only=False,
            )

    def test_local_model_path_never_calls_hub(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = EncoderConfig(
                name="local",
                family="video",
                model_id=directory,
                revision=None,
            )
            with patch(
                "information_upper_bound.encoder_runtime._download_snapshot",
                side_effect=AssertionError("must not download"),
            ):
                runtime, declared = prepare_encoder_runtime(config)
            self.assertEqual(runtime.model_id, str(Path(directory).resolve()))
            self.assertEqual(declared["model_id"], directory)

    def test_local_files_only_flag_is_forwarded(self) -> None:
        revision = "d" * 40
        config = EncoderConfig(
            name="encoder-a",
            family="video",
            model_id="owner/model",
            revision=revision,
        )
        with (
            patch.dict(os.environ, {"VLMEB_LOCAL_FILES_ONLY": "1"}),
            patch(
                "information_upper_bound.encoder_runtime._download_snapshot",
                return_value="/resolved/snapshot",
            ) as download,
        ):
            prepare_encoder_runtime(config)
        self.assertTrue(download.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
