from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from information_upper_bound.io import sha256_file, write_json
from information_upper_bound.clevrer_pilot_contract import expected_clevrer_unit_ids
from information_upper_bound.pilot_protocol import main as prepare_protocol_main
from information_upper_bound.protocol import load_protocol, validate_data_protocol
from information_upper_bound.unit_sampling import (
    RESAMPLING_UNIT_FIELD,
    RESAMPLING_UNIT_SELECTION_ALGORITHM,
    RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION,
    resampling_unit_set_sha256,
)


class PilotProtocolTests(unittest.TestCase):
    def test_preparer_binds_authenticated_pilot_release_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = "adapter-run::" + "a" * 64
            release_id = "b" * 64
            population_ids = expected_clevrer_unit_ids("validation")
            report_path = root / "adapter.report.json"
            write_json(
                report_path,
                {
                    "dataset": "clevrer",
                    "adapter_run_id": run_id,
                    "confirmatory_eligible": True,
                    "resampling_unit_selection": {
                        "schema_version": RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION,
                        "algorithm": RESAMPLING_UNIT_SELECTION_ALGORITHM,
                        "unit_field": RESAMPLING_UNIT_FIELD,
                        "sample_size": 500,
                        "selected_unit_count": 500,
                        "seed": 42,
                        "dataset": "clevrer",
                        "canonical_split": "validation",
                        "population_record_count": 70862,
                        "population_unit_count": 5000,
                        "population_unit_set_sha256": (
                            resampling_unit_set_sha256(population_ids)
                        ),
                        "selected_record_count": 7000,
                        "population_units": [
                            {
                                "resampling_unit_id": unit_id,
                                "record_count": 15 if index < 862 else 14,
                            }
                            for index, unit_id in enumerate(population_ids)
                        ],
                    },
                },
            )
            lock_path = root / "lock.json"
            write_json(
                lock_path,
                {
                    "audit": {
                        "adapter_reports": [
                            {
                                "adapter_run_id": run_id,
                                "report_sha256": sha256_file(report_path),
                            }
                        ]
                    }
                },
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            output = root / "protocol.yaml"
            authenticated = {
                "datasets": {"clevrer": 7000},
                "records": 7000,
                "adapter_runs": [{"adapter_run_id": run_id}],
                "data_release_sha256": release_id,
            }
            with (
                patch(
                    "information_upper_bound.pilot_protocol.validate_data_lock",
                    return_value=authenticated,
                ),
                patch(
                    "information_upper_bound.pilot_protocol.validate_release_coverage",
                    return_value={"valid": True},
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = prepare_protocol_main(
                    [
                        "--manifest",
                        str(manifest),
                        "--adapter-report",
                        str(report_path),
                        "--data-lock",
                        str(lock_path),
                        "--out",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn(release_id, text)
            self.assertIn(run_id, text)
            self.assertNotIn("REPLACE_WITH_CLEVRER_PILOT_DATA_RELEASE_SHA256", text)
            self.assertNotIn("REPLACE_WITH_CLEVRER_PILOT_ADAPTER_RUN_ID", text)
            self.assertIn("REPLACE_WITH_PROJECTOR_CHECKPOINT_SHA256", text)
            protocol, _metadata = load_protocol(output)
            data_protocol = validate_data_protocol(protocol)
            self.assertEqual(data_protocol["data_release_sha256"], release_id)
            self.assertEqual(
                data_protocol["coverage_contract"]["datasets"]["clevrer"][
                    "required_adapter_run_ids"
                ],
                [run_id],
            )

            tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
            tampered_report["resampling_unit_selection"]["population_units"][-1][
                "resampling_unit_id"
            ] = "clevrer:scene:99999"
            write_json(report_path, tampered_report)
            tampered_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            tampered_lock["audit"]["adapter_reports"][0]["report_sha256"] = sha256_file(
                report_path
            )
            write_json(lock_path, tampered_lock)
            with (
                patch(
                    "information_upper_bound.pilot_protocol.validate_data_lock",
                    return_value=authenticated,
                ),
                self.assertRaisesRegex(ValueError, "official scene-ID universe"),
            ):
                prepare_protocol_main(
                    [
                        "--manifest",
                        str(manifest),
                        "--adapter-report",
                        str(report_path),
                        "--data-lock",
                        str(lock_path),
                        "--out",
                        str(root / "tampered-protocol.yaml"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
