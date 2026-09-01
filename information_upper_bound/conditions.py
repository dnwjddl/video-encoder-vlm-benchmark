from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from .schema import answer_text, normalize_answer, option_label, parse_evidence_spans


INPUT_CHANNELS = {
    "question_only",
    "visual",
    "text_oracle",
    "embedding_oracle",
    "visual_plus_text",
}
VISUAL_VIEWS = {
    "none",
    "full",
    "single",
    "reverse",
    "shuffle",
    "evidence_only",
    "evidence_present",
    "evidence_removed",
    "random_position_mask",
    "random_matched",
}
SAFE_ORACLE_ACCESS = {"safe_visual_gt", "operator_only"}
SAFE_ORACLE_LINEAGE = {
    "official_adapter",
    "audited_human_annotation",
    "audited_simulator_gt",
}
ALLOWED_CLUE_FIELDS = {
    "static_facts",
    "unordered_events",
    "ordered_events",
    "temporal_relations",
    "state_changes",
    "relations",
    "operator",
    "intermediate",
}
DEFAULT_CONDITION_PATH = Path(__file__).with_name("configs") / "conditions.yaml"


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    input_channel: str
    visual_view: str
    clue_fields: tuple[str, ...] = ()
    always_include_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    required_any_fields: tuple[str, ...] = ()
    sham_fields: tuple[str, ...] = ()
    matched_event_view: str | None = None
    requires_matched_events: bool = False
    doses: tuple[int | str, ...] = ("all",)
    requires_evidence: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConditionSpec":
        name = str(value.get("name", "")).strip()
        input_channel = str(value.get("input_channel", "")).strip()
        visual_view = str(value.get("visual_view", "none")).strip()
        clue_fields = tuple(str(item) for item in value.get("clue_fields", []) or [])
        always_include_fields = tuple(
            str(item) for item in value.get("always_include_fields", []) or []
        )
        required_fields = tuple(
            str(item) for item in value.get("required_fields", []) or []
        )
        required_any_fields = tuple(
            str(item) for item in value.get("required_any_fields", []) or []
        )
        sham_fields = tuple(str(item) for item in value.get("sham_fields", []) or [])
        matched_event_view = value.get("matched_event_view")
        if matched_event_view is not None:
            matched_event_view = str(matched_event_view).strip().casefold()
        raw_doses = value.get("doses", ["all"])
        if not isinstance(raw_doses, list) or not raw_doses:
            raise ValueError(f"condition {name!r}: doses must be a non-empty list")
        doses: list[int | str] = []
        for dose in raw_doses:
            if dose == "all":
                doses.append("all")
            elif isinstance(dose, int) and not isinstance(dose, bool) and dose >= 0:
                doses.append(dose)
            else:
                raise ValueError(f"condition {name!r}: invalid dose {dose!r}")
        if len(doses) != len(set(doses)):
            raise ValueError(f"condition {name!r}: doses must be unique")
        spec = cls(
            name=name,
            input_channel=input_channel,
            visual_view=visual_view,
            clue_fields=clue_fields,
            always_include_fields=always_include_fields,
            required_fields=required_fields,
            required_any_fields=required_any_fields,
            sham_fields=sham_fields,
            matched_event_view=matched_event_view,
            requires_matched_events=bool(value.get("requires_matched_events", False)),
            doses=tuple(doses),
            requires_evidence=bool(value.get("requires_evidence", False)),
            description=str(value.get("description", "")),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.name:
            raise ValueError("condition name cannot be empty")
        if self.input_channel not in INPUT_CHANNELS:
            raise ValueError(
                f"condition {self.name!r}: invalid input_channel {self.input_channel!r}"
            )
        if self.visual_view not in VISUAL_VIEWS:
            raise ValueError(
                f"condition {self.name!r}: invalid visual_view {self.visual_view!r}"
            )
        needs_visual = self.input_channel in {"visual", "visual_plus_text"}
        if needs_visual != (self.visual_view != "none"):
            raise ValueError(
                f"condition {self.name!r}: visual channel/view mismatch "
                f"({self.input_channel}, {self.visual_view})"
            )
        needs_clue = self.input_channel in {
            "text_oracle",
            "embedding_oracle",
            "visual_plus_text",
        }
        if needs_clue and not self.clue_fields:
            raise ValueError(
                f"condition {self.name!r}: oracle channel requires clue_fields"
            )
        if not needs_clue and self.clue_fields:
            raise ValueError(
                f"condition {self.name!r}: non-oracle channel cannot use clue_fields"
            )
        declared = set(self.clue_fields)
        unknown_clue_fields = sorted(declared - ALLOWED_CLUE_FIELDS)
        if unknown_clue_fields:
            raise ValueError(
                f"condition {self.name!r}: unsupported clue fields {unknown_clue_fields}"
            )
        constrained = (
            set(self.always_include_fields)
            | set(self.required_fields)
            | set(self.required_any_fields)
        )
        if not constrained.issubset(declared):
            raise ValueError(
                f"condition {self.name!r}: required/always fields must also appear in clue_fields"
            )
        if not set(self.sham_fields).issubset(declared):
            raise ValueError(
                f"condition {self.name!r}: sham_fields must also appear in clue_fields"
            )
        unsupported_shams = sorted(set(self.sham_fields) - {"operator"})
        if unsupported_shams:
            raise ValueError(
                f"condition {self.name!r}: unsupported sham fields {unsupported_shams}"
            )
        if self.matched_event_view not in {None, "atomic", "ordered", "timestamp_sham"}:
            raise ValueError(
                f"condition {self.name!r}: matched_event_view must be "
                "'atomic', 'ordered', or 'timestamp_sham'"
            )
        if self.requires_matched_events and self.matched_event_view is None:
            raise ValueError(
                f"condition {self.name!r}: requires_matched_events needs matched_event_view"
            )
        if self.matched_event_view is not None:
            required_event_field = (
                "unordered_events"
                if self.matched_event_view == "atomic"
                else "ordered_events"
            )
            if required_event_field not in declared:
                raise ValueError(
                    f"condition {self.name!r}: matched_event_view={self.matched_event_view!r} "
                    f"requires {required_event_field!r} in clue_fields"
                )


def load_condition_config(
    path: str | Path,
) -> tuple[list[ConditionSpec], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("condition config must contain a YAML object")
    values = raw.get("conditions")
    if not isinstance(values, list) or not values:
        raise ValueError("condition config must contain a non-empty 'conditions' list")
    specs = [ConditionSpec.from_dict(value) for value in values]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    options = dict(raw.get("options") or {})
    return specs, options


def _canonical_hash(value: Any, *, length: int = 20) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def trial_content_payload(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical scoring- and analysis-relevant content bound to a trial ID."""

    raw_diagnostic = trial.get("diagnostic") or {}
    diagnostic = {
        key: raw_diagnostic.get(key)
        for key in (
            "schema_version",
            "dataset",
            "split",
            "information_family",
            "question_family",
            "reasoning_depth",
            "resampling_unit_id",
            "pair_id",
            "pair_role",
            "independent_unit_id",
            "official_candidate_id",
            "official_candidate_count",
            "evidence_spans",
            "media_clip",
        )
        if key in raw_diagnostic
    }
    raw_visual_spec = trial.get("visual_spec")
    visual_spec = (
        {
            key: value
            for key, value in raw_visual_spec.items()
            if key not in {"media_path"}
        }
        if isinstance(raw_visual_spec, Mapping)
        else raw_visual_spec
    )
    return {
        "schema": "information_upper_bound.trial_content.v3",
        "data_release_sha256": trial.get("data_release_sha256"),
        "trial_build_attestation": trial.get("trial_build_attestation"),
        "base_id": str(trial.get("base_id", "")),
        "media_type": str(trial.get("media_type", "video")),
        "question": str(trial.get("question", "")),
        "choices": list(trial.get("choices") or []),
        "answer": trial.get("answer"),
        "answer_text": trial.get("answer_text"),
        "clue_text": trial.get("clue_text", ""),
        "visual_id": trial.get("visual_id"),
        "visual_spec": visual_spec,
        "condition": trial.get("condition"),
        "diagnostic": diagnostic,
    }


def trial_content_sha256(trial: Mapping[str, Any]) -> str:
    return _canonical_hash(trial_content_payload(trial), length=64)


def _format_time(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _fact_text(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if not isinstance(value, Mapping):
        return str(value)

    if field == "operator":
        if value.get("text"):
            return " ".join(str(value["text"]).split())

        def program_steps(raw: Any) -> str | None:
            if not isinstance(raw, list) or not raw:
                return None
            steps = [
                (
                    " ".join(step.replace("_", " ").split())
                    if isinstance(step, str)
                    else json.dumps(step, ensure_ascii=False, sort_keys=True)
                )
                for step in raw
            ]
            return " -> ".join(steps)

        candidate_program = program_steps(value.get("choice_program"))
        question_program = program_steps(value.get("question_program"))
        program_parts = []
        if candidate_program:
            program_parts.append(f"candidate program: {candidate_program}")
        if question_program:
            program_parts.append(f"question program: {question_program}")
        if program_parts:
            composition = str(value.get("composition", "")).strip()
            if composition:
                program_parts.append(
                    "composition rule: " + " ".join(composition.split())
                )
            return "; ".join(program_parts)

        semantic_payload = {
            str(key): item
            for key, item in value.items()
            if str(key) not in {"access", "source", "lineage"}
        }
        return json.dumps(semantic_payload, ensure_ascii=False, sort_keys=True)

    if field in {"unordered_events", "ordered_events"}:
        if value.get("text"):
            statement = " ".join(str(value["text"]).split())
        else:
            pieces = [
                str(value.get("subject", "")).strip(),
                str(value.get("predicate", value.get("action", ""))).strip(),
                str(value.get("object", "")).strip(),
            ]
            statement = " ".join(piece for piece in pieces if piece)
        start = _format_time(value.get("start_sec", value.get("start")))
        end = _format_time(value.get("end_sec", value.get("end")))
        frame = value.get("frame", value.get("frame_index"))
        if field == "ordered_events":
            if start is not None:
                interval = f"t={start}s" if end is None else f"t={start}-{end}s"
                statement = f"[{interval}] {statement}"
            elif frame is not None:
                statement = f"[frame={frame}] {statement}"
        return statement or json.dumps(value, ensure_ascii=False, sort_keys=True)

    if value.get("text"):
        return " ".join(str(value["text"]).split())

    left = value.get("left", value.get("source", value.get("subject")))
    relation = value.get("relation", value.get("predicate", value.get("type")))
    right = value.get("right", value.get("target", value.get("object")))
    pieces = [
        str(piece).strip()
        for piece in (left, relation, right)
        if piece not in (None, "")
    ]
    return " ".join(pieces) or json.dumps(value, ensure_ascii=False, sort_keys=True)


def oracle_facts(
    record: Mapping[str, Any], fields: Sequence[str]
) -> list[tuple[str, str]]:
    diagnostic = record.get("diagnostic") or {}
    oracles = diagnostic.get("oracles") or {}
    facts: list[tuple[str, str]] = []
    for oracle_field in fields:
        raw = oracles.get(oracle_field)
        if raw in (None, "", []):
            continue
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, Mapping):
                access = str(value.get("access", "")).strip().casefold()
                lineage = str(value.get("lineage", "")).strip().casefold()
                source = str(value.get("source", "")).strip()
                if (
                    access not in SAFE_ORACLE_ACCESS
                    or lineage not in SAFE_ORACLE_LINEAGE
                    or not source
                ):
                    # Answer-equivalent annotations are useful for dataset
                    # bookkeeping, but only explicitly safe lineages may enter
                    # a clue intervention.
                    continue
            else:
                # Strict manifests use fact objects with explicit access/source
                # lineage. Bare strings remain invalid even if a caller skipped
                # schema validation.
                continue
            text = _fact_text(value, field=oracle_field).strip()
            if text:
                facts.append((oracle_field, text))
    return facts


_EVENT_TIMING_KEYS = {
    "start",
    "end",
    "start_sec",
    "end_sec",
    "frame",
    "frame_index",
    "timestamp",
    "timestamps",
    "time",
    "unit",
}


def _safe_oracle_values(record: Mapping[str, Any], field: str) -> list[Any]:
    diagnostic = record.get("diagnostic") or {}
    oracles = diagnostic.get("oracles") or {}
    raw = oracles.get(field)
    if raw in (None, "", []):
        return []
    values = raw if isinstance(raw, list) else [raw]
    safe: list[Any] = []
    for value in values:
        if isinstance(value, Mapping):
            access = str(value.get("access", "")).strip().casefold()
            lineage = str(value.get("lineage", "")).strip().casefold()
            source = str(value.get("source", "")).strip()
            if (
                access not in SAFE_ORACLE_ACCESS
                or lineage not in SAFE_ORACLE_LINEAGE
                or not source
            ):
                continue
        else:
            continue
        safe.append(value)
    return safe


def _event_fact_id(value: Any, *, field: str, index: int) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field}[{index}] cannot enter a matched atomic/ordered contrast: "
            "an object with a shared fact_id or event_id is required"
        )
    raw_id = value.get("fact_id", value.get("event_id"))
    if raw_id in (None, ""):
        raise ValueError(
            f"{field}[{index}] cannot enter a matched atomic/ordered contrast: "
            "shared fact_id/event_id is missing"
        )
    return str(raw_id)


def _event_semantic_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only presentation/timing fields before matching event semantics."""

    return {
        str(key): item
        for key, item in value.items()
        if str(key) not in _EVENT_TIMING_KEYS
    }


def _event_time_key(value: Mapping[str, Any], fallback: str) -> tuple[int, float, str]:
    for key in ("start_sec", "start", "frame", "frame_index", "timestamp", "time"):
        raw = value.get(key)
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        return (0, numeric, fallback)
    return (1, 0.0, fallback)


def _timestamp_sham_text(value: Mapping[str, Any], atomic_text: str) -> str:
    """Render timing-shaped but temporally uninformative event metadata.

    The fixed neutral values never depend on the answer, question, choices, or
    true timestamp magnitude/order. Event semantics remain identical to the
    atomic/ordered conditions.
    """

    start = _format_time(value.get("start_sec", value.get("start")))
    end = _format_time(value.get("end_sec", value.get("end")))
    frame = value.get("frame", value.get("frame_index"))
    if start is not None:
        prefix = "[t=000.000s]" if end is None else "[t=000.000-000.000s]"
        return f"{prefix} {atomic_text}"
    if frame is not None:
        return f"[frame=000000] {atomic_text}"
    return atomic_text


def _format_sham_text(text: str) -> str:
    """Erase lexical content while preserving exact character/punctuation layout."""

    return "".join(
        "0" if character.isdigit() else "x" if character.isalpha() else character
        for character in text
    )


def matched_event_facts(
    record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair atomic and timed renderings by an explicit answer-independent event ID.

    The selected event set is ranked by an opaque hash, never by timestamp. This
    makes a finite dose contain exactly the same semantic events in the atomic
    and ordered conditions without leaking temporal order through selection.
    """

    raw_by_field = {
        field: _safe_oracle_values(record, field)
        for field in ("unordered_events", "ordered_events")
    }
    if not raw_by_field["unordered_events"] or not raw_by_field["ordered_events"]:
        return [], {
            "shared_event_count": 0,
            "atomic_event_count": len(raw_by_field["unordered_events"]),
            "ordered_event_count": len(raw_by_field["ordered_events"]),
            "unmatched_atomic_event_ids": [],
            "unmatched_ordered_event_ids": [],
        }

    by_field: dict[str, dict[str, Mapping[str, Any]]] = {}
    for oracle_field in ("unordered_events", "ordered_events"):
        indexed: dict[str, Mapping[str, Any]] = {}
        for index, value in enumerate(raw_by_field[oracle_field]):
            fact_id = _event_fact_id(value, field=oracle_field, index=index)
            if fact_id in indexed:
                raise ValueError(
                    f"diagnostic.oracles.{oracle_field} has duplicate event ID {fact_id!r}"
                )
            assert isinstance(value, Mapping)
            indexed[fact_id] = value
        by_field[oracle_field] = indexed

    unordered = by_field["unordered_events"]
    ordered = by_field["ordered_events"]
    shared_ids = sorted(set(unordered) & set(ordered))
    facts: list[dict[str, Any]] = []
    record_id = str(record.get("id", ""))
    for fact_id in shared_ids:
        atomic_value = unordered[fact_id]
        ordered_value = ordered[fact_id]
        atomic_semantic = _event_semantic_payload(atomic_value)
        ordered_semantic = _event_semantic_payload(ordered_value)
        if atomic_semantic != ordered_semantic:
            raise ValueError(
                "matched oracle event changes non-temporal semantics for "
                f"record={record_id!r}, event_id={fact_id!r}"
            )
        semantic_sha256 = _canonical_hash(atomic_semantic, length=64)
        selection_rank = _canonical_hash(
            {"record_id": record_id, "event_id": fact_id}, length=64
        )
        facts.append(
            {
                "fact_id": fact_id,
                "semantic_sha256": semantic_sha256,
                "selection_rank": selection_rank,
                "atomic_text": _fact_text(atomic_value, field="unordered_events"),
                "ordered_text": _fact_text(ordered_value, field="ordered_events"),
                "timestamp_sham_text": _timestamp_sham_text(
                    ordered_value,
                    _fact_text(atomic_value, field="unordered_events"),
                ),
                "ordered_sort_key": _event_time_key(ordered_value, selection_rank),
            }
        )
    facts.sort(key=lambda value: (value["selection_rank"], value["fact_id"]))
    audit = {
        "shared_event_count": len(shared_ids),
        "atomic_event_count": len(unordered),
        "ordered_event_count": len(ordered),
        "unmatched_atomic_event_ids": sorted(set(unordered) - set(ordered)),
        "unmatched_ordered_event_ids": sorted(set(ordered) - set(unordered)),
    }
    return facts, audit


def render_clue_details(
    record: Mapping[str, Any],
    fields: Sequence[str],
    dose: int | str,
    *,
    always_include_fields: Sequence[str] = (),
    matched_event_view: str | None = None,
    sham_fields: Sequence[str] = (),
) -> tuple[str, int, dict[str, Any]]:
    if matched_event_view is None:
        always = set(always_include_fields)
        facts = oracle_facts(record, [field for field in fields if field not in always])
        mandatory = oracle_facts(record, always_include_fields)
        selected = facts if dose == "all" else facts[: int(dose)]
        selected = selected + mandatory
        selected_fact_ids: list[str] = []
        semantic_hashes: list[str] = []
        matched_audit: dict[str, Any] = {}
    else:
        if matched_event_view not in {"atomic", "ordered", "timestamp_sham"}:
            raise ValueError(
                "matched_event_view must be 'atomic', 'ordered', or 'timestamp_sham'"
            )
        matched, matched_audit = matched_event_facts(record)
        selected_matched = matched if dose == "all" else matched[: int(dose)]
        rendered = (
            sorted(
                selected_matched,
                key=lambda value: (value["ordered_sort_key"], value["selection_rank"]),
            )
            if matched_event_view == "ordered"
            else selected_matched
        )
        field = (
            "unordered_events" if matched_event_view == "atomic" else "ordered_events"
        )
        text_key = {
            "atomic": "atomic_text",
            "ordered": "ordered_text",
            "timestamp_sham": "timestamp_sham_text",
        }[matched_event_view]
        selected = [(field, str(value[text_key])) for value in rendered]
        mandatory = oracle_facts(record, always_include_fields)
        selected.extend(mandatory)
        selected_fact_ids = [str(value["fact_id"]) for value in selected_matched]
        semantic_hashes = [str(value["semantic_sha256"]) for value in selected_matched]

    sham_set = set(sham_fields)
    sham_character_counts: list[dict[str, Any]] = []
    transformed: list[tuple[str, str]] = []
    for field, text in selected:
        if field in sham_set:
            sham_text = _format_sham_text(text)
            sham_character_counts.append(
                {
                    "field": field,
                    "source_characters": len(text),
                    "sham_characters": len(sham_text),
                }
            )
            transformed.append((field, sham_text))
        else:
            transformed.append((field, text))
    selected = transformed

    audit = {
        "render_mode": matched_event_view or "raw_fields",
        "selected_fact_ids": selected_fact_ids,
        "selected_fact_semantic_sha256": semantic_hashes,
        "mandatory_fields": list(always_include_fields),
        "sham_fields": list(sham_fields),
        "sham_character_counts": sham_character_counts,
        **matched_audit,
    }
    if not selected:
        return "", 0, audit
    labels = {
        "static_facts": "Visual fact",
        "unordered_events": "Observed event",
        "ordered_events": "Timed event",
        "temporal_relations": "Temporal relation",
        "state_changes": "State change",
        "relations": "Relation",
        "operator": "Reasoning operator",
        "intermediate": "Intermediate fact",
        "rationale": "Rationale",
    }
    lines = [f"- {labels.get(field, field)}: {text}" for field, text in selected]
    return (
        "Evidence supplied by the diagnostic oracle:\n" + "\n".join(lines),
        len(selected),
        audit,
    )


def render_clue(
    record: Mapping[str, Any],
    fields: Sequence[str],
    dose: int | str,
    *,
    always_include_fields: Sequence[str] = (),
    matched_event_view: str | None = None,
    sham_fields: Sequence[str] = (),
) -> tuple[str, int]:
    clue, effective_dose, _audit = render_clue_details(
        record,
        fields,
        dose,
        always_include_fields=always_include_fields,
        matched_event_view=matched_event_view,
        sham_fields=sham_fields,
    )
    return clue, effective_dose


def _permutations(
    count: int, replicates: int, *, seed: int, base_id: str
) -> list[tuple[int, ...]]:
    identity = tuple(range(count))
    if replicates <= 1:
        return [identity]
    values: list[tuple[int, ...]] = []
    # Cyclic shifts give exact answer-position balance when replicates is a
    # multiple of the number of choices.
    for shift in range(min(replicates, count)):
        values.append(tuple((index + shift) % count for index in range(count)))
    if len(values) >= replicates:
        return values[:replicates]

    rng = random.Random(f"{seed}:{base_id}:option-permutation")
    seen = set(values)
    max_unique = 1
    for value in range(2, count + 1):
        max_unique *= value
    while len(values) < min(replicates, max_unique):
        candidate = list(identity)
        rng.shuffle(candidate)
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            values.append(key)
    if len(values) < replicates:
        values.extend(
            values[index % len(values)] for index in range(replicates - len(values))
        )
    return values


def _apply_permutation(
    record: Mapping[str, Any], permutation: Sequence[int]
) -> tuple[list[str], str]:
    choices = [str(value) for value in record.get("choices") or []]
    gold_label = normalize_answer(record.get("answer"), choices)
    gold_index = ord(gold_label) - ord("A")
    # permutation[new_position] = old_position
    permuted = [choices[old_index] for old_index in permutation]
    new_gold_index = list(permutation).index(gold_index)
    return permuted, option_label(new_gold_index)


def _visual_spec(
    record: Mapping[str, Any], spec: ConditionSpec, *, seed: int
) -> dict[str, Any] | None:
    if spec.visual_view == "none":
        return None
    evidence_views = {
        "evidence_only",
        "evidence_present",
        "evidence_removed",
        "random_position_mask",
        "random_matched",
    }
    spans = (
        [span.to_dict() for span in parse_evidence_spans(record)]
        if spec.visual_view in evidence_views
        else []
    )
    diagnostic = record.get("diagnostic") or {}
    media_clip = (
        diagnostic.get("media_clip") if isinstance(diagnostic, Mapping) else None
    )
    return {
        "view": spec.visual_view,
        "media_path": str(record["media_path"]),
        "media_type": str(record.get("media_type", "video")),
        "evidence_spans": spans,
        "clip": dict(media_clip) if isinstance(media_clip, Mapping) else None,
        "seed": int(seed),
    }


def _visual_source_identity(record: Mapping[str, Any]) -> dict[str, str]:
    diagnostic = record.get("diagnostic") or {}
    provenance = diagnostic.get("provenance") or {}
    for key in (
        "source_video_id",
        "raw_video_id",
        "video_id",
        "video_key",
        "scene_index",
        "raw_video",
        "q_uid",
        "source_id",
    ):
        value = provenance.get(key) if isinstance(provenance, Mapping) else None
        if value not in (None, ""):
            return {
                "dataset": str(
                    diagnostic.get("dataset", record.get("benchmark", "unknown"))
                ),
                "field": key,
                "value": str(value),
            }
    return {
        "dataset": str(diagnostic.get("dataset", record.get("benchmark", "unknown"))),
        "field": "base_id",
        "value": str(record.get("id", "")),
    }


def _normalize_option_permutations(option_permutations: int | str) -> tuple[bool, int]:
    all_option_positions = str(option_permutations).strip().casefold() in {
        "all",
        "all_positions",
    }
    if all_option_positions:
        return True, 0
    if isinstance(option_permutations, bool):
        raise ValueError("option_permutations must be >= 1 or 'all'")
    try:
        fixed_permutations = int(option_permutations)
    except (TypeError, ValueError) as exc:
        raise ValueError("option_permutations must be >= 1 or 'all'") from exc
    if fixed_permutations < 1:
        raise ValueError("option_permutations must be >= 1 or 'all'")
    return False, fixed_permutations


@dataclass
class TrialBuildState:
    seed: int
    option_permutations: int | str
    conditions: tuple[str, ...]
    base_records: int = 0
    trials: int = 0
    skipped_count: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    skipped_reasons: Counter[str] = field(default_factory=Counter)
    unique_visual_inputs: set[str] = field(default_factory=set)
    skipped_preview_limit: int = 100

    def add_skip(self, *, base_id: str, condition: str, reason: str) -> None:
        self.skipped_count += 1
        self.skipped_reasons[reason] += 1
        if len(self.skipped) < self.skipped_preview_limit:
            self.skipped.append(
                {"base_id": base_id, "condition": condition, "reason": reason}
            )

    def report(self) -> dict[str, Any]:
        return {
            "base_records": self.base_records,
            "trials": self.trials,
            "unique_visual_inputs": len(self.unique_visual_inputs),
            "skipped_count": self.skipped_count,
            "skipped_reasons": dict(sorted(self.skipped_reasons.items())),
            "skipped": list(self.skipped),
            "skipped_preview_truncated": self.skipped_count > len(self.skipped),
            "seed": self.seed,
            "option_permutations": self.option_permutations,
            "conditions": list(self.conditions),
        }


def stream_trials(
    records: Iterable[Mapping[str, Any]],
    specs: Sequence[ConditionSpec],
    *,
    seed: int = 42,
    option_permutations: int | str = 1,
) -> tuple[Iterator[dict[str, Any]], TrialBuildState]:
    """Return a streaming trial iterator plus a mutable build-report state."""

    all_option_positions, fixed_permutations = _normalize_option_permutations(
        option_permutations
    )
    state = TrialBuildState(
        seed=int(seed),
        option_permutations="all" if all_option_positions else fixed_permutations,
        conditions=tuple(spec.name for spec in specs),
    )

    def generate() -> Iterator[dict[str, Any]]:
        for raw_record in records:
            record = dict(raw_record)
            state.base_records += 1
            base_id = str(record["id"])
            choices = [str(value) for value in record.get("choices") or []]
            replicate_count = (
                len(choices) if all_option_positions else fixed_permutations
            )
            permutations = _permutations(
                len(choices), replicate_count, seed=seed, base_id=base_id
            )
            evidence_spans = parse_evidence_spans(record)

            for spec in specs:
                if spec.requires_evidence and not evidence_spans:
                    state.add_skip(
                        base_id=base_id,
                        condition=spec.name,
                        reason="no_evidence_span",
                    )
                    continue
                available_fields = {
                    name for name in spec.clue_fields if oracle_facts(record, [name])
                }
                missing_required = sorted(set(spec.required_fields) - available_fields)
                if missing_required:
                    state.add_skip(
                        base_id=base_id,
                        condition=spec.name,
                        reason="missing_required_oracle_fields:"
                        + ",".join(missing_required),
                    )
                    continue
                if spec.required_any_fields and not (
                    set(spec.required_any_fields) & available_fields
                ):
                    state.add_skip(
                        base_id=base_id,
                        condition=spec.name,
                        reason="missing_all_required_any_oracle_fields",
                    )
                    continue
                if spec.requires_matched_events:
                    matched, _matched_audit = matched_event_facts(record)
                    if not matched:
                        state.add_skip(
                            base_id=base_id,
                            condition=spec.name,
                            reason="no_matched_atomic_ordered_events",
                        )
                        continue
                for dose in spec.doses:
                    clue_text, effective_dose, clue_audit = render_clue_details(
                        record,
                        spec.clue_fields,
                        dose,
                        always_include_fields=spec.always_include_fields,
                        matched_event_view=spec.matched_event_view,
                        sham_fields=spec.sham_fields,
                    )
                    if (
                        spec.input_channel
                        in {"text_oracle", "embedding_oracle", "visual_plus_text"}
                        and not clue_text
                    ):
                        state.add_skip(
                            base_id=base_id,
                            condition=spec.name,
                            reason="no_oracle_fact",
                        )
                        continue
                    visual_spec = _visual_spec(record, spec, seed=seed)
                    visual_id = (
                        "visual::"
                        + _canonical_hash(
                            {
                                "source": _visual_source_identity(record),
                                "view_spec": {
                                    key: value
                                    for key, value in visual_spec.items()
                                    if key != "media_path"
                                },
                            }
                        )
                        if visual_spec is not None
                        else None
                    )
                    if visual_id is not None:
                        state.unique_visual_inputs.add(visual_id)
                    for permutation_index, permutation in enumerate(permutations):
                        permuted_choices, permuted_answer = _apply_permutation(
                            record, permutation
                        )
                        condition = {
                            "name": spec.name,
                            "description": spec.description,
                            "input_channel": spec.input_channel,
                            "visual_view": spec.visual_view,
                            "clue_fields": list(spec.clue_fields),
                            "always_include_fields": list(spec.always_include_fields),
                            "required_fields": list(spec.required_fields),
                            "required_any_fields": list(spec.required_any_fields),
                            "sham_fields": list(spec.sham_fields),
                            "matched_event_view": spec.matched_event_view,
                            "requires_matched_events": spec.requires_matched_events,
                            "clue_audit": clue_audit,
                            "requested_dose": dose,
                            "effective_dose": effective_dose,
                            "permutation_index": permutation_index,
                            "permutation": list(permutation),
                            "seed": seed,
                        }
                        diagnostic = json.loads(
                            json.dumps(
                                record.get("diagnostic") or {}, ensure_ascii=False
                            )
                        )
                        semantic_answer = answer_text(record)
                        trial_body = {
                            **record,
                            "base_id": base_id,
                            "visual_id": visual_id,
                            "choices": permuted_choices,
                            "answer": permuted_answer,
                            "answer_text": semantic_answer,
                            "clue_text": clue_text,
                            "visual_spec": visual_spec,
                            "condition": condition,
                            "diagnostic": diagnostic,
                        }
                        content_sha256 = trial_content_sha256(trial_body)
                        trial_id = f"trial::{content_sha256}"
                        state.trials += 1
                        yield {
                            **trial_body,
                            "id": trial_id,
                            "trial_id": trial_id,
                            "trial_content_sha256": content_sha256,
                        }

    return generate(), state


def build_trials(
    records: Iterable[Mapping[str, Any]],
    specs: Sequence[ConditionSpec],
    *,
    seed: int = 42,
    option_permutations: int | str = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize trials for programmatic use; the CLI uses ``stream_trials``."""

    iterator, state = stream_trials(
        records,
        specs,
        seed=seed,
        option_permutations=option_permutations,
    )
    trials = list(iterator)
    return trials, state.report()
