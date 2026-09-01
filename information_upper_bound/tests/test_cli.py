from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from information_upper_bound.cli import _build_trials_command
from information_upper_bound.io import write_jsonl
from information_upper_bound.tests.test_schema_conditions import base_record


class TrialBuildCliTests(unittest.TestCase):
    def test_confirmatory_build_requires_data_lock_and_development_is_explicit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, [base_record()])
            locked_output = root / "locked.jsonl"
            with self.assertRaisesRegex(ValueError, "requires --data-lock"):
                _build_trials_command(
                    ["--manifest", str(manifest), "--out", str(locked_output)]
                )
            self.assertFalse(locked_output.exists())

            development_output = root / "development.jsonl"
            with redirect_stdout(io.StringIO()):
                _build_trials_command(
                    [
                        "--manifest",
                        str(manifest),
                        "--out",
                        str(development_output),
                        "--development",
                    ]
                )
            self.assertTrue(development_output.is_file())
            report = json.loads(
                development_output.with_suffix(".jsonl.report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["execution_mode"], "development")
            self.assertIsNone(report["data_lock"])
            self.assertEqual(report["trial_build_attestation"]["mode"], "development")

    def test_build_rejects_output_report_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, [base_record()])
            output = root / "same.jsonl"
            with self.assertRaisesRegex(ValueError, "path collision"):
                _build_trials_command(
                    [
                        "--manifest",
                        str(manifest),
                        "--out",
                        str(output),
                        "--report-out",
                        str(output),
                        "--development",
                    ]
                )

    def test_build_rejects_reserved_attestation_fields_in_base_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {**base_record(), "data_release_sha256": "a" * 64}
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, [record])
            with self.assertRaisesRegex(ValueError, "reserved"):
                _build_trials_command(
                    [
                        "--manifest",
                        str(manifest),
                        "--out",
                        str(root / "trials.jsonl"),
                        "--development",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
