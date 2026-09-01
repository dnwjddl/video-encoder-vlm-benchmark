from __future__ import annotations

import ast
import copy
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from information_upper_bound.schema import (
    InformationFamily,
    PairRole,
    SCHEMA_VERSION,
    has_errors,
    normalize_answer,
    option_label,
    validate_record,
)


class AdapterError(ValueError):
    """Raised when an official annotation cannot be mapped unambiguously."""


@dataclass(frozen=True)
class ExclusionEntry:
    dataset: str
    source_id: str
    reason: str


class ExclusionManifest:
    """Strict, auditable exclusions for one adapter invocation.

    A manifest may contain entries for several datasets.  Matching is always
    performed on the composite ``(dataset, source_id)`` key, and only entries
    for the active adapter are required to be consumed by that adapter.
    """

    def __init__(self, path: str | Path | None, *, dataset: str) -> None:
        self.dataset = require_text(dataset, path="exclusion dataset")
        self.path = (
            Path(path).expanduser().resolve() if path not in (None, "") else None
        )
        self.sha256: str | None = None
        self._total_entries = 0
        self._entries_by_dataset: dict[str, int] = {}
        self._selected: dict[str, ExclusionEntry] = {}
        self._applied: dict[str, dict[str, str]] = {}
        if self.path is None:
            return
        if not self.path.is_file():
            raise AdapterError(
                f"exclusion manifest does not exist or is not a file: {self.path}"
            )
        self.sha256 = _sha256_file(self.path)
        entries = _load_exclusion_entries(self.path)
        self._total_entries = len(entries)
        seen: set[tuple[str, str]] = set()
        for index, raw_entry in enumerate(entries):
            entry_path = f"{self.path}:exclusions[{index}]"
            entry = require_mapping(raw_entry, path=entry_path)
            unknown = set(entry) - {"dataset", "source_id", "reason"}
            if unknown:
                raise AdapterError(
                    f"{entry_path}: unknown fields {sorted(unknown)}; expected dataset, source_id, reason"
                )
            parsed = ExclusionEntry(
                dataset=require_text(
                    entry.get("dataset"), path=f"{entry_path}.dataset"
                ),
                source_id=require_text(
                    entry.get("source_id"), path=f"{entry_path}.source_id"
                ),
                reason=require_text(entry.get("reason"), path=f"{entry_path}.reason"),
            )
            key = (parsed.dataset, parsed.source_id)
            if key in seen:
                raise AdapterError(
                    "duplicate exclusion for composite key "
                    f"dataset={parsed.dataset!r}, source_id={parsed.source_id!r}"
                )
            seen.add(key)
            self._entries_by_dataset[parsed.dataset] = (
                self._entries_by_dataset.get(parsed.dataset, 0) + 1
            )
            if parsed.dataset == self.dataset:
                self._selected[parsed.source_id] = parsed

    @property
    def configured(self) -> bool:
        return self.path is not None

    def contains(self, source_id: str) -> bool:
        """Non-consuming lookup for preliminary pairing/grouping passes."""

        return str(source_id) in self._selected

    def consume(self, source_id: str, *, raw_location: str) -> bool:
        """Consume one matching entry before parsing the excluded raw row."""

        clean_source_id = require_text(source_id, path="exclusion lookup source_id")
        entry = self._selected.get(clean_source_id)
        if entry is None:
            return False
        if clean_source_id in self._applied:
            previous = self._applied[clean_source_id]["raw_location"]
            raise AdapterError(
                "ambiguous exclusion matched multiple raw rows for "
                f"dataset={self.dataset!r}, source_id={clean_source_id!r}: "
                f"{previous!r} and {raw_location!r}"
            )
        self._applied[clean_source_id] = {
            "dataset": entry.dataset,
            "source_id": entry.source_id,
            "reason": entry.reason,
            "raw_location": require_text(raw_location, path="exclusion raw_location"),
        }
        return True

    def finalize(self) -> dict[str, Any]:
        unused = sorted(set(self._selected) - set(self._applied))
        if unused:
            raise AdapterError(
                f"unused exclusions for dataset {self.dataset!r}: {unused[:20]}"
            )
        return self.report()

    def report(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "dataset": self.dataset,
            "file": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "manifest_entries": self._total_entries,
            "entries_by_dataset": dict(sorted(self._entries_by_dataset.items())),
            "selected_entries": len(self._selected),
            "applied": [self._applied[key] for key in sorted(self._applied)],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exclusion_entries(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        entries: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise AdapterError(
                            f"{path}:{line_number}: invalid exclusion JSON: {exc}"
                        ) from exc
                    entries.append(require_mapping(value, path=f"{path}:{line_number}"))
        except OSError as exc:
            raise AdapterError(
                f"failed to read exclusion manifest {path}: {exc}"
            ) from exc
        return entries
    value = load_json(path)
    if isinstance(value, Mapping):
        unknown = set(value) - {"exclusions"}
        if unknown or "exclusions" not in value:
            raise AdapterError(
                f"{path}: exclusion JSON object must contain only an 'exclusions' list"
            )
        value = value["exclusions"]
    rows = require_list(value, path=str(path))
    return [
        require_mapping(row, path=f"{path}[{index}]") for index, row in enumerate(rows)
    ]


@dataclass(frozen=True)
class AdapterConfig:
    annotation_path: Path
    media_root: Path
    split: str = "eval"
    require_media: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        annotation_path: str | Path,
        media_root: str | Path,
        *,
        split: str = "eval",
        require_media: bool = True,
        options: Mapping[str, Any] | None = None,
    ) -> "AdapterConfig":
        annotation = Path(annotation_path).expanduser().resolve()
        root = Path(media_root).expanduser().resolve()
        if not annotation.exists():
            raise AdapterError(f"annotation path does not exist: {annotation}")
        if require_media and not root.is_dir():
            raise AdapterError(
                f"media root does not exist or is not a directory: {root}"
            )
        if not str(split).strip():
            raise AdapterError("split must be a non-empty string")
        return cls(
            annotation_path=annotation,
            media_root=root,
            split=str(split).strip(),
            require_media=bool(require_media),
            options=dict(options or {}),
        )


ORACLE_LIST_FIELDS = (
    "static_facts",
    "unordered_events",
    "ordered_events",
    "temporal_relations",
    "state_changes",
    "relations",
    "intermediate",
)


def empty_oracles() -> dict[str, Any]:
    return {
        **{field: [] for field in ORACLE_LIST_FIELDS},
        "operator": None,
        "answer_derived": False,
    }


def merge_oracles(value: Mapping[str, Any] | None) -> dict[str, Any]:
    out = empty_oracles()
    if value:
        unknown = set(value) - set(out)
        if unknown:
            raise AdapterError(f"unknown oracle fields: {sorted(unknown)}")
        out.update(value)
    if out.get("answer_derived") is not False:
        raise AdapterError("adapter oracles must not be derived from the gold answer")

    def normalize_fact(field: str, fact: Any) -> dict[str, Any]:
        if isinstance(fact, Mapping):
            normalized = dict(fact)
        elif isinstance(fact, str) and fact.strip():
            normalized = {"text": fact.strip()}
        else:
            raise AdapterError(
                f"oracle field {field!r} facts must be non-empty strings or objects"
            )
        normalized.setdefault(
            "access", "operator_only" if field == "operator" else "safe_visual_gt"
        )
        normalized.setdefault("source", "official_adapter_annotation")
        normalized.setdefault("lineage", "official_adapter")
        return normalized

    for key in ORACLE_LIST_FIELDS:
        if not isinstance(out[key], list):
            raise AdapterError(f"oracle field {key!r} must be a list")
        out[key] = [normalize_fact(key, fact) for fact in out[key]]
    operator = out.get("operator")
    if operator not in (None, "", []):
        operator_values = operator if isinstance(operator, list) else [operator]
        out["operator"] = [normalize_fact("operator", fact) for fact in operator_values]
    else:
        out["operator"] = None
    return out


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"failed to read JSON {source}: {exc}") from exc


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AdapterError(
                        f"{source}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AdapterError(f"{source}:{line_number}: expected an object")
                rows.append(row)
        return rows
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except (
            ImportError
        ) as exc:  # pragma: no cover - dependency is present in production setup
            raise AdapterError(
                "parquet input requires pandas and a parquet engine such as pyarrow"
            ) from exc
        try:
            return [
                dict(row) for row in pd.read_parquet(source).to_dict(orient="records")
            ]
        except Exception as exc:
            raise AdapterError(f"failed to read parquet {source}: {exc}") from exc
    value = load_json(source)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AdapterError(f"{source}: expected a JSON list of objects")
    return [dict(row) for row in value]


def require_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterError(f"{path}: expected an object, got {type(value).__name__}")
    return dict(value)


def require_list(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise AdapterError(f"{path}: expected a list, got {type(value).__name__}")
    return value


def require_text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{path}: expected a non-empty string")
    return value.strip()


def optional_source_id_component(value: Any) -> str | None:
    """Read a scalar raw ID without invoking strict row parsing."""

    if not isinstance(value, bool) and isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            return text
    return None


def source_id_component(value: Any, *, fallback: str) -> str:
    """Return a raw ID without invoking strict row parsing, or a stable row key."""

    return optional_source_id_component(value) or require_text(
        fallback,
        path="fallback source_id",
    )


def parse_candidates(value: Any, *, path: str) -> list[str]:
    raw = value
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise AdapterError(
                f"{path}: candidate string is not a Python list literal"
            ) from exc
    values = require_list(raw, path=path)
    choices = [
        require_text(item, path=f"{path}[{index}]") for index, item in enumerate(values)
    ]
    if not 2 <= len(choices) <= 26:
        raise AdapterError(f"{path}: expected 2..26 choices, got {len(choices)}")
    normalized = [" ".join(choice.casefold().split()) for choice in choices]
    if len(set(normalized)) != len(normalized):
        raise AdapterError(f"{path}: duplicate normalized choices")
    return choices


def normalize_integer_answer(
    value: Any, choices: Sequence[str], *, base: int, path: str
) -> str:
    """Normalize a declared zero- or one-based source label."""

    if base not in {0, 1}:
        raise AdapterError(f"{path}: unsupported integer answer base {base}")
    if isinstance(value, bool):
        raise AdapterError(f"{path}: boolean is not an integer answer label")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        value = int(value.strip())
    if not isinstance(value, int):
        raise AdapterError(f"{path}: expected a base-{base} integer answer label")
    index = value - base
    if not 0 <= index < len(choices):
        raise AdapterError(
            f"{path}: base-{base} answer {value} is outside the {len(choices)} choices"
        )
    return option_label(index)


def normalize_text_answer(value: Any, choices: Sequence[str], *, path: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        raise AdapterError(
            f"{path}: integer answer is ambiguous; this adapter expects answer text"
        )
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{path}: expected non-empty answer text")
    normalized = " ".join(value.casefold().split())
    matches = [
        option_label(index)
        for index, choice in enumerate(choices)
        if " ".join(str(choice).casefold().split()) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AdapterError(f"{path}: answer text matches multiple duplicate choices")
    raise AdapterError(f"{path}: answer text does not exactly match a candidate")


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_media(
    media_root: str | Path,
    relative_candidates: Iterable[str | Path],
    *,
    require: bool = True,
    search_basename: str | None = None,
) -> str:
    """Resolve exactly one media file; ambiguity is always an error."""

    root = Path(media_root).expanduser().resolve()
    attempted: list[Path] = []
    found: dict[str, Path] = {}
    for raw in relative_candidates:
        if raw in (None, ""):
            continue
        value = Path(str(raw)).expanduser()
        candidate = value.resolve() if value.is_absolute() else (root / value).resolve()
        if not _inside(root, candidate):
            raise AdapterError(
                f"media candidate escapes media root {root}: {candidate}"
            )
        attempted.append(candidate)
        if candidate.is_file():
            found[str(candidate)] = candidate
    if not found and search_basename:
        basename = Path(search_basename).name
        if basename != search_basename:
            raise AdapterError(f"invalid basename search request: {search_basename}")
        for candidate in root.rglob(basename):
            if candidate.is_file():
                resolved = candidate.resolve()
                if not _inside(root, resolved):
                    raise AdapterError(
                        f"media basename search escapes media root {root}: {resolved}"
                    )
                found[str(resolved)] = resolved
    if len(found) > 1:
        raise AdapterError(
            "ambiguous media resolution; matched multiple files: "
            + ", ".join(sorted(found))
        )
    if found:
        return str(next(iter(found.values())))
    if require:
        preview = ", ".join(str(path) for path in attempted[:8])
        raise AdapterError(
            f"media file not found; attempted: {preview or '<no candidates>'}"
        )
    if attempted:
        return str(attempted[0])
    raise AdapterError("no media candidates were provided")


def stable_id(dataset: str, *parts: Any) -> str:
    raw = json.dumps([dataset, *parts], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{dataset}:{digest}"


def make_record(
    *,
    record_id: str,
    source: str,
    benchmark: str,
    task: str,
    media_path: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    dataset: str,
    split: str,
    information_family: str,
    question_family: str,
    reasoning_depth: int,
    resampling_unit_id: str,
    pair_id: str | None,
    pair_role: str,
    evidence_spans: Sequence[Mapping[str, Any]] | None,
    oracles: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    extra_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean_question = require_text(question, path="question")
    clean_choices = parse_candidates(list(choices), path="choices")
    try:
        clean_answer = normalize_answer(answer, clean_choices)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"answer: {exc}") from exc
    if information_family not in {value.value for value in InformationFamily}:
        raise AdapterError(f"invalid information family: {information_family}")
    if pair_role not in {value.value for value in PairRole}:
        raise AdapterError(f"invalid pair role: {pair_role}")
    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "dataset": require_text(dataset, path="diagnostic.dataset"),
        "split": require_text(split, path="diagnostic.split"),
        "information_family": information_family,
        "question_family": require_text(
            question_family, path="diagnostic.question_family"
        ),
        "reasoning_depth": int(reasoning_depth),
        "resampling_unit_id": require_text(
            resampling_unit_id, path="diagnostic.resampling_unit_id"
        ),
        "pair_id": pair_id or f"standalone:{record_id}",
        "pair_role": pair_role,
        "evidence_spans": [dict(span) for span in (evidence_spans or [])],
        "oracles": merge_oracles(oracles),
        "provenance": dict(provenance),
    }
    if extra_diagnostic:
        overlap = set(diagnostic).intersection(extra_diagnostic)
        if overlap:
            raise AdapterError(
                f"extra diagnostic fields overwrite canonical keys: {sorted(overlap)}"
            )
        diagnostic.update(extra_diagnostic)
    if not diagnostic["provenance"].get("source_id"):
        raise AdapterError("provenance.source_id is required")
    record = {
        "id": require_text(record_id, path="id"),
        "source": require_text(source, path="source"),
        "benchmark": require_text(benchmark, path="benchmark"),
        "task": require_text(task, path="task"),
        "media_type": "video",
        "media_path": require_text(media_path, path="media_path"),
        "question": clean_question,
        "choices": clean_choices,
        "answer": clean_answer,
        "diagnostic": diagnostic,
    }
    issues = validate_record(record, require_media=False, strict_diagnostic=True)
    if has_errors(issues):
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise AdapterError(f"record {record_id} failed schema validation: {details}")
    return record


class BaseAdapter:
    name = "base"

    def __init__(
        self,
        annotation_path: str | Path,
        media_root: str | Path,
        *,
        split: str = "eval",
        require_media: bool = True,
        **options: Any,
    ) -> None:
        self.config = AdapterConfig.create(
            annotation_path,
            media_root,
            split=split,
            require_media=require_media,
            options=options,
        )
        self._exclusions = ExclusionManifest(
            options.get("exclusions_path"),
            dataset=self.name,
        )
        self._exclusion_report = self._exclusions.report()

    def is_excluded(self, source_id: str) -> bool:
        """Check an exclusion during a non-emitting preliminary pass."""

        return self._exclusions.contains(source_id)

    def skip_excluded(self, source_id: str, *, raw_location: str) -> bool:
        """Apply a matching exclusion before strict parsing of a raw row."""

        return self._exclusions.consume(source_id, raw_location=raw_location)

    @property
    def exclusion_report(self) -> dict[str, Any]:
        return copy.deepcopy(self._exclusion_report)

    def iter_records(self) -> Iterable[dict[str, Any]]:
        raise NotImplementedError

    def load(self) -> list[dict[str, Any]]:
        rows = list(self.iter_records())
        self._exclusion_report = self._exclusions.finalize()
        if not rows:
            raise AdapterError(f"{self.name}: annotation produced no evaluable records")
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            raise AdapterError(f"{self.name}: duplicate record ids: {duplicates[:10]}")
        if self._exclusions.configured:
            for row in rows:
                provenance = row["diagnostic"]["provenance"]
                provenance["exclusion_manifest"] = copy.deepcopy(self._exclusion_report)
        return rows
