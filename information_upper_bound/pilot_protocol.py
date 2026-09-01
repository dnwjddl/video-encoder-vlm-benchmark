from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .clevrer_pilot_contract import validate_clevrer_selection_report
from .data_lock import validate_data_lock
from .io import read_jsonl, sha256_file
from .protocol import (
    validate_data_protocol,
    validate_frozen_model_protocol,
    validate_release_coverage,
)


DEFAULT_TEMPLATE = Path(__file__).with_name("configs") / "clevrer_pilot_protocol.yaml"
DEFAULT_CONDITIONS = (
    Path(__file__).with_name("configs") / "clevrer_core_conditions.yaml"
)
RELEASE_PLACEHOLDER = "REPLACE_WITH_CLEVRER_PILOT_DATA_RELEASE_SHA256"
RUN_PLACEHOLDER = "REPLACE_WITH_CLEVRER_PILOT_ADAPTER_RUN_ID"


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _template_protocol(path: Path) -> tuple[str, Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"could not read pilot protocol template {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("pilot protocol template must contain a YAML mapping")
    if text.count(RELEASE_PLACEHOLDER) != 1 or text.count(RUN_PLACEHOLDER) != 1:
        raise ValueError(
            "pilot protocol template must contain each pilot data placeholder exactly once"
        )
    return text, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate a sampled CLEVRER adapter report/data lock and fill the two "
            "data identities in the frozen pilot protocol template."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-report", required=True)
    parser.add_argument("--data-lock", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--conditions-config", default=str(DEFAULT_CONDITIONS))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.out).expanduser().resolve()
    inputs = {
        Path(value).expanduser().resolve()
        for value in (
            args.manifest,
            args.adapter_report,
            args.data_lock,
            args.template,
            args.conditions_config,
        )
    }
    if output in inputs:
        raise ValueError("pilot protocol output must not alias an input")
    if (output.exists() or output.is_symlink()) and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    report_path = Path(args.adapter_report).expanduser().resolve()
    lock_path = Path(args.data_lock).expanduser().resolve()
    report = _load_mapping(report_path, label="adapter report")
    lock = _load_mapping(lock_path, label="data lock")
    authenticated = validate_data_lock(
        lock_path,
        manifest_path=args.manifest,
        verify_sources=False,
        verify_media=False,
    )
    if authenticated.get("datasets") != {"clevrer": authenticated.get("records")}:
        raise ValueError("pilot data lock must contain only CLEVRER records")
    runs = authenticated.get("adapter_runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError("pilot data lock must contain exactly one adapter run")
    run_id = str(runs[0].get("adapter_run_id", ""))
    if report.get("dataset") != "clevrer" or report.get("adapter_run_id") != run_id:
        raise ValueError("adapter report does not match the sole CLEVRER data-lock run")
    if report.get("confirmatory_eligible") is not True:
        raise ValueError("adapter report is not confirmatory-eligible")

    selection = report.get("resampling_unit_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("adapter report is not a deterministic resampling-unit sample")
    validate_clevrer_selection_report(
        selection,
        role="validation",
        locked_record_count=int(authenticated["records"]),
    )

    audit = lock.get("audit")
    audit_reports = audit.get("adapter_reports") if isinstance(audit, Mapping) else None
    if not isinstance(audit_reports, list) or len(audit_reports) != 1:
        raise ValueError("data lock has no unique audited adapter report")
    audited_report = audit_reports[0]
    if (
        not isinstance(audited_report, Mapping)
        or audited_report.get("adapter_run_id") != run_id
        or audited_report.get("report_sha256") != sha256_file(report_path)
    ):
        raise ValueError(
            "current adapter report bytes are not the report audited by the lock"
        )

    template_text, template = _template_protocol(
        Path(args.template).expanduser().resolve()
    )
    data = template.get("data")
    pilot = template.get("pilot")
    if not isinstance(data, Mapping) or not isinstance(pilot, Mapping):
        raise ValueError("pilot protocol template is missing data/pilot sections")
    conditions_sha256 = sha256_file(args.conditions_config)
    if data.get("conditions_sha256") != conditions_sha256:
        raise ValueError(
            "pilot protocol template conditions_sha256 does not match --conditions-config"
        )
    if (
        pilot.get("target_resampling_units") != 500
        or pilot.get("selection_seed") != 42
        or pilot.get("answer_blind") is not True
        or pilot.get("preserve_complete_units") is not True
    ):
        raise ValueError("pilot protocol template has incompatible selection metadata")

    release_sha256 = str(authenticated["data_release_sha256"])
    finalized = template_text.replace(RELEASE_PLACEHOLDER, release_sha256).replace(
        RUN_PLACEHOLDER, run_id
    )
    finalized_protocol = yaml.safe_load(finalized)
    if not isinstance(finalized_protocol, Mapping):
        raise ValueError("finalized pilot protocol is not a YAML mapping")
    data_protocol = validate_data_protocol(finalized_protocol)
    validate_frozen_model_protocol(finalized_protocol)
    coverage = validate_release_coverage(
        data_protocol,
        authenticated,
        read_jsonl(args.manifest),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(finalized, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "data_release_sha256": release_sha256,
                "adapter_run_id": run_id,
                "conditions_sha256": conditions_sha256,
                "selected_resampling_units": 500,
                "coverage_valid": coverage["valid"],
                "projector_section_finalized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
