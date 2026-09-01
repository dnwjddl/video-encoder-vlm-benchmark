from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from information_upper_bound.io import sha256_file, write_json, write_jsonl
from information_upper_bound.validate import validate_manifest

from .common import AdapterError
from .registry import available_adapters, build_adapter


ADAPTER_REPORT_SCHEMA_VERSION = "information_upper_bound.adapter_report.v1"
ANNOTATION_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".pq"}
MVP_CATEGORIES = {
    "human_object_interactions",
    "robot_object_interactions",
    "intuitive_physics",
    "temporal_reasoning",
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_set_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Content-address an adapter output independent of JSONL row order/path."""

    leaves: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("id", "")).strip()
        if not record_id or record_id in seen:
            raise AdapterError(
                f"adapter output contains an empty/duplicate id while building its report: {record_id!r}"
            )
        seen.add(record_id)
        leaves.append({"id": record_id, "sha256": _canonical_sha256(row)})
    leaves.sort(key=lambda value: value["id"])
    return _canonical_sha256(
        {
            "schema_version": "information_upper_bound.adapter_record_set.v1",
            "records": leaves,
        }
    )


def _reject_output_aliases(
    args: argparse.Namespace, output: Path, report: Path
) -> None:
    named: dict[str, Path] = {
        "output": output.expanduser().resolve(),
        "report_output": report.expanduser().resolve(),
        "annotations": Path(args.annotations).expanduser().resolve(),
    }
    for name in (
        "exclusions",
        "meta_info",
        "grounding",
        "video_map",
        "answers",
        "cut_frame_mapping",
        "scene_annotations",
    ):
        value = getattr(args, name)
        if value not in (None, ""):
            named[name] = Path(str(value)).expanduser().resolve()
    reverse: dict[Path, str] = {}
    for name, path in named.items():
        previous = reverse.get(path)
        if previous is not None:
            raise AdapterError(
                f"path collision: {previous} and {name} resolve to the same path: {path}"
            )
        reverse[path] = name


def _declared_source_artifacts(
    args: argparse.Namespace,
    *,
    discovered_files: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Enumerate every declared annotation/sidecar byte with portable names."""

    declared: list[tuple[str, Path]] = [("annotations", Path(args.annotations))]
    for role in (
        "exclusions",
        "meta_info",
        "grounding",
        "video_map",
        "answers",
        "cut_frame_mapping",
        "scene_annotations",
    ):
        value = getattr(args, role)
        if value not in (None, ""):
            declared.append((role, Path(str(value))))

    artifacts: list[dict[str, Any]] = []
    covered: set[Path] = set()
    checksums: dict[str, str] = {}
    for role, raw_root in declared:
        root = raw_root.expanduser().resolve()
        if root.is_file():
            candidates = [(root.name, root)]
        elif root.is_dir():
            candidates = [
                (path.relative_to(root).as_posix(), path)
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.casefold() in ANNOTATION_SUFFIXES
            ]
            if not candidates:
                raise AdapterError(
                    f"declared annotation directory contains no supported files: {root}"
                )
        else:
            raise AdapterError(f"declared annotation source does not exist: {root}")
        for relative_path, path in candidates:
            digest = sha256_file(path)
            size = int(path.stat().st_size)
            covered.add(path)
            checksums[str(path)] = digest
            artifacts.append(
                {
                    "role": role,
                    "relative_path": relative_path,
                    "sha256": digest,
                    "size_bytes": size,
                }
            )

    # Provenance may name a derived/consumed sidecar not exposed as a dedicated
    # CLI argument. It remains portable by using its basename rather than its
    # local mount root.
    for path in sorted({value.resolve() for value in discovered_files} - covered):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        checksums[str(path)] = digest
        artifacts.append(
            {
                "role": "provenance_discovered",
                "relative_path": path.name,
                "sha256": digest,
                "size_bytes": int(path.stat().st_size),
            }
        )
    artifacts.sort(
        key=lambda item: (
            str(item["role"]),
            str(item["relative_path"]),
            str(item["sha256"]),
        )
    )
    return artifacts, dict(sorted(checksums.items()))


def _coverage_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    question_family_counts: Counter[str] = Counter()
    evidence_records = 0
    safe_oracle_records = 0
    for row in rows:
        diagnostic = dict(row.get("diagnostic") or {})
        split_counts[str(diagnostic.get("split", ""))] += 1
        family_counts[str(diagnostic.get("information_family", ""))] += 1
        question_family_counts[str(diagnostic.get("question_family", ""))] += 1
        if diagnostic.get("evidence_spans"):
            evidence_records += 1
        oracles = diagnostic.get("oracles")
        if isinstance(oracles, dict):
            facts = [
                fact
                for value in oracles.values()
                if isinstance(value, list)
                for fact in value
                if isinstance(fact, dict)
            ]
            if any(fact.get("access") == "safe_visual_gt" for fact in facts):
                safe_oracle_records += 1
    return {
        "schema_version": "information_upper_bound.adapter_coverage.v1",
        "split_counts": dict(sorted(split_counts.items())),
        "information_family_counts": dict(sorted(family_counts.items())),
        "question_family_counts": dict(sorted(question_family_counts.items())),
        "records_with_evidence": evidence_records,
        "records_with_safe_oracles": safe_oracle_records,
    }


def _confirmatory_eligibility_issues(args: argparse.Namespace) -> list[str]:
    issues: list[str] = []
    if args.allow_missing_media:
        issues.append("allow_missing_media")
    if args.allow_missing_grounding:
        issues.append("allow_missing_grounding")
    if args.allow_missing_cut_mapping:
        issues.append("allow_missing_cut_mapping")
    if args.allow_uncut_cup_games:
        issues.append("allow_uncut_cup_games")
    if args.limit is not None:
        issues.append("limited")
    required_sidecars = {
        "tempcompass": (("meta_info", args.meta_info),),
        "perception_test": (("cut_frame_mapping", args.cut_frame_mapping),),
        "next_gqa": (("grounding", args.grounding), ("video_map", args.video_map)),
        "clevrer": (("scene_annotations", args.scene_annotations),),
        "egoschema": (("answers", args.answers),),
    }
    for role, value in required_sidecars.get(args.dataset, ()):
        if value in (None, ""):
            issues.append(f"missing_required_sidecar:{role}")
    if args.dataset == "tvbench":
        if Path(args.annotations).expanduser().is_file():
            issues.append("tvbench_single_task_file")
        if args.task or args.tasks:
            issues.append("tvbench_task_subset")
    if args.dataset == "mvp" and args.category not in MVP_CATEGORIES:
        issues.append("mvp_category_must_be_explicit_official_partition")
    return sorted(set(issues))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an already-downloaded official video benchmark release into the strict "
            "information-upper-bound manifest. This command never downloads data."
        )
    )
    parser.add_argument("--dataset", required=True, choices=available_adapters())
    parser.add_argument(
        "--annotations", required=True, help="Official annotation file or release root."
    )
    parser.add_argument(
        "--media-root",
        required=True,
        help="Root containing the already-downloaded videos.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument(
        "--report-output", help="Build report JSON; defaults to <output>.report.json."
    )
    parser.add_argument(
        "--exclusions",
        help=(
            "Audited JSON/JSONL exclusions with dataset, source_id, and nonempty reason. "
            "source_id normally matches emitted provenance.source_id. For a malformed source ID, "
            "stable 0-based fallbacks are: task:row:N (TVBench), row:N (NExT-GQA/EgoSchema/MVP), "
            "video:mc_question:row:N (Perception Test), and "
            "row:SCENE:question:QUESTION:choice:CHOICE (CLEVRER)."
        ),
    )
    parser.add_argument(
        "--split", default="eval", help="Canonical analysis split name."
    )
    parser.add_argument(
        "--allow-missing-media",
        action="store_true",
        help="Annotation-audit mode only; records retain deterministic unresolved paths.",
    )
    parser.add_argument("--meta-info", help="TempCompass meta_info.json.")
    parser.add_argument("--grounding", help="NExT-GQA gsub_<split>.json.")
    parser.add_argument("--video-map", help="NExT-GQA map_vid_vidorID.json.")
    parser.add_argument("--answers", help="EgoSchema subset_answers.json.")
    parser.add_argument(
        "--cut-frame-mapping", help="Perception Test official cut-frame JSON."
    )
    parser.add_argument(
        "--scene-annotations",
        help="CLEVRER processed_proposals directory or one exact sim_XXXXX.json file.",
    )
    parser.add_argument(
        "--source-split", help="Official source split, distinct from canonical --split."
    )
    parser.add_argument(
        "--task", help="Single TVBench task when --annotations points to one JSON file."
    )
    parser.add_argument(
        "--tasks", help="Comma-separated TVBench tasks for a release-root build."
    )
    parser.add_argument(
        "--category", help="MVP config/category for JSON, CSV, or parquet input."
    )
    parser.add_argument("--include-track-geometry", action="store_true")
    parser.add_argument("--include-audio-oracles", action="store_true")
    parser.add_argument("--allow-missing-grounding", action="store_true")
    parser.add_argument(
        "--allow-missing-cut-mapping",
        action="store_true",
        help=(
            "Allow Perception Test train/valid MCQ without its official cut map. "
            "This can leak end-of-video answers and is disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-uncut-cup-games",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Infrastructure check only; never use a limited test as a result.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _options(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    mapping = {
        "meta_info": "meta_info_path",
        "grounding": "grounding_path",
        "video_map": "video_map_path",
        "answers": "answers_path",
        "cut_frame_mapping": "cut_frame_mapping_path",
        "scene_annotations": "scene_annotations_path",
        "source_split": "source_split",
        "task": "task",
        "category": "category",
        "exclusions": "exclusions_path",
    }
    for source, target in mapping.items():
        value = getattr(args, source)
        if value not in (None, ""):
            values[target] = value
    if args.tasks:
        values["tasks"] = [
            value.strip() for value in args.tasks.split(",") if value.strip()
        ]
    values.update(
        {
            "include_track_geometry": args.include_track_geometry,
            "include_audio_oracles": args.include_audio_oracles,
            "allow_missing_grounding": args.allow_missing_grounding,
            "allow_missing_cut_mapping": args.allow_missing_cut_mapping,
            "allow_uncut_cup_games": args.allow_uncut_cup_games,
        }
    )
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    report_path = (
        Path(args.report_output)
        if args.report_output
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )
    try:
        _reject_output_aliases(args, output_path, report_path)
    except AdapterError as exc:
        raise SystemExit(f"adapter error: {exc}") from exc
    if not args.dry_run:
        existing = [path for path in (output_path, report_path) if path.exists()]
        if existing and not args.overwrite:
            raise SystemExit(
                "adapter error: output exists; pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    try:
        adapter = build_adapter(
            args.dataset,
            args.annotations,
            args.media_root,
            split=args.split,
            require_media=not args.allow_missing_media,
            **_options(args),
        )
        rows = adapter.load()
        if args.limit is not None:
            if args.limit <= 0:
                raise AdapterError("--limit must be positive")
            rows = rows[: args.limit]
        report = validate_manifest(
            rows,
            require_media=not args.allow_missing_media,
            strict_diagnostic=True,
        )
        errors = [
            issue for issue in report.get("issues", []) if issue.get("level") == "error"
        ]
        if errors:
            preview = "; ".join(
                f"{issue.get('record_id') or '<manifest>'}:{issue.get('path')}: {issue.get('message')}"
                for issue in errors[:10]
            )
            raise AdapterError(
                f"manifest validation failed with {len(errors)} errors: {preview}"
            )
        source_paths: set[Path] = set()
        for row in rows:
            provenance = (row.get("diagnostic") or {}).get("provenance") or {}
            for key, value in provenance.items():
                if not (key.endswith("_file") or key.endswith("_path")) or value in (
                    None,
                    "",
                ):
                    continue
                candidate = Path(str(value)).expanduser().resolve()
                if candidate.is_file():
                    source_paths.add(candidate)
        source_artifacts, source_checksums = _declared_source_artifacts(
            args,
            discovered_files=sorted(source_paths, key=str),
        )
        adapter_options = {
            "include_track_geometry": bool(args.include_track_geometry),
            "include_audio_oracles": bool(args.include_audio_oracles),
            "source_split": args.source_split,
            "task": args.task,
            "tasks": (
                [value.strip() for value in args.tasks.split(",") if value.strip()]
                if args.tasks
                else None
            ),
            "category": args.category,
        }
        adapter_run_id = "adapter-run::" + _canonical_sha256(
            {
                "schema_version": "information_upper_bound.adapter_run.v1",
                "dataset": args.dataset,
                "canonical_split": args.split,
                "adapter_options": adapter_options,
                "source_artifacts": source_artifacts,
                "record_ids": sorted(str(row["id"]) for row in rows),
            }
        )
        rows = [
            {
                **row,
                "diagnostic": {
                    **dict(row.get("diagnostic") or {}),
                    "adapter_run_id": adapter_run_id,
                },
            }
            for row in rows
        ]
        report = validate_manifest(
            rows,
            require_media=not args.allow_missing_media,
            strict_diagnostic=True,
        )
        errors = [
            issue for issue in report.get("issues", []) if issue.get("level") == "error"
        ]
        if errors:
            raise AdapterError(
                f"manifest validation failed after adapter-run binding with {len(errors)} errors"
            )
        debug_options = {
            "allow_missing_media": bool(args.allow_missing_media),
            "allow_missing_grounding": bool(args.allow_missing_grounding),
            "allow_missing_cut_mapping": bool(args.allow_missing_cut_mapping),
            "allow_uncut_cup_games": bool(args.allow_uncut_cup_games),
            "limited": args.limit is not None,
        }
        confirmatory_eligibility_issues = _confirmatory_eligibility_issues(args)
        build_report = {
            "schema_version": ADAPTER_REPORT_SCHEMA_VERSION,
            "dataset": args.dataset,
            "adapter_run_id": adapter_run_id,
            "records": len(rows),
            "limited": args.limit is not None,
            "limit": args.limit,
            "canonical_split": args.split,
            "require_media": not args.allow_missing_media,
            "confirmatory_eligible": not confirmatory_eligibility_issues,
            "confirmatory_eligibility_issues": confirmatory_eligibility_issues,
            "debug_options": debug_options,
            "adapter_options": adapter_options,
            "manifest_record_set_sha256": _record_set_sha256(rows),
            "source_artifacts": source_artifacts,
            "source_artifact_root_sha256": _canonical_sha256(source_artifacts),
            "source_roles": sorted({str(item["role"]) for item in source_artifacts}),
            "coverage": _coverage_summary(rows),
            "source_checksums_sha256": source_checksums,
            "exclusions": adapter.exclusion_report,
            "validation": report,
        }
        if not args.dry_run:
            write_jsonl(output_path, rows)
            build_report["output_manifest_sha256"] = sha256_file(output_path)
            write_json(report_path, build_report)
    except AdapterError as exc:
        raise SystemExit(f"adapter error: {exc}") from exc
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "adapter_run_id": adapter_run_id,
                "records": len(rows),
                "output": None if args.dry_run else str(output_path.resolve()),
                "report_output": None if args.dry_run else str(report_path.resolve()),
                "dry_run": args.dry_run,
                "confirmatory_eligible": build_report["confirmatory_eligible"],
                "source_artifact_root_sha256": build_report[
                    "source_artifact_root_sha256"
                ],
                "exclusions": adapter.exclusion_report,
                "validation": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
