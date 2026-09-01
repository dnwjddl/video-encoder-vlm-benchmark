from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from information_upper_bound.schema import InformationFamily, PairRole

from .common import (
    AdapterError,
    BaseAdapter,
    empty_oracles,
    load_json,
    load_json_rows,
    make_record,
    normalize_text_answer,
    optional_source_id_component,
    parse_candidates,
    require_list,
    require_mapping,
    require_text,
    resolve_media,
    stable_id,
)


def _family(question_type: str) -> tuple[str, int]:
    prefix = question_type.strip().upper()[:1]
    if prefix == "C":
        return InformationFamily.CAUSAL_COMPOSITIONAL.value, 2
    if prefix == "T":
        return InformationFamily.TEMPORAL_ORDER.value, 2
    if prefix == "D":
        return InformationFamily.STATIC.value, 0
    raise AdapterError(f"unknown NExT-GQA question type {question_type!r}")


def _float(value: Any, *, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{path}: expected a numeric value") from exc
    if not math.isfinite(number):
        raise AdapterError(f"{path}: value must be finite")
    return number


def _spans(
    grounding: Mapping[str, Any],
    *,
    video_id: str,
    qid: str,
    path: str,
) -> list[dict[str, Any]]:
    if video_id not in grounding:
        raise AdapterError(f"{path}: grounding has no entry for video {video_id}")
    video = require_mapping(grounding[video_id], path=f"{path}.{video_id}")
    locations = require_mapping(
        video.get("location"), path=f"{path}.{video_id}.location"
    )
    if qid not in locations:
        raise AdapterError(f"{path}.{video_id}.location: no evidence for qid {qid}")
    duration = _float(video.get("duration"), path=f"{path}.{video_id}.duration")
    if duration <= 0:
        raise AdapterError(f"{path}.{video_id}.duration: must be positive")
    raw_spans = require_list(locations[qid], path=f"{path}.{video_id}.location.{qid}")
    out = []
    for index, raw_span in enumerate(raw_spans):
        span = require_list(raw_span, path=f"{path}.{video_id}.location.{qid}[{index}]")
        if len(span) != 2:
            raise AdapterError(
                f"{path}.{video_id}.location.{qid}[{index}]: expected [start,end]"
            )
        start = _float(span[0], path=f"{path}.{video_id}.location.{qid}[{index}][0]")
        end = _float(span[1], path=f"{path}.{video_id}.location.{qid}[{index}][1]")
        # The official release contains a small number of -0.1/-0.2 starts
        # caused by decimal timestamp rounding, and its integer ``duration``
        # can be up to 0.5 s shorter than a released span endpoint.  Preserve
        # the endpoint and only clamp the physically impossible near-zero
        # start, retaining the raw value on the span for auditability.  Larger
        # discrepancies still indicate a malformed sidecar.
        if start < -1.0 or end <= max(start, 0.0) or end > duration + 1.0:
            raise AdapterError(
                f"{path}.{video_id}.location.{qid}[{index}]: interval [{start},{end}] "
                f"is inconsistent with reported duration {duration}"
            )
        span_record: dict[str, Any] = {
            "start": max(start, 0.0),
            "end": end,
            "unit": "seconds",
            "role": "necessary",
            "event_id": f"{video_id}:{qid}:evidence:{index}",
        }
        if start < 0:
            span_record.update(
                {
                    "source_start": start,
                    "normalization": "clamped_near_zero_rounding_to_video_start",
                }
            )
        out.append(span_record)
    if not out:
        raise AdapterError(f"{path}.{video_id}.location.{qid}: evidence list is empty")
    return out


class NExTGQAAdapter(BaseAdapter):
    """Adapter for official NExT-GQA CSV plus gsub temporal grounding files."""

    name = "next_gqa"

    def _paths(self) -> tuple[Path, Path | None, Path | None, str]:
        annotation = self.config.annotation_path
        if annotation.is_dir():
            source_split = str(self.config.options.get("source_split") or "").strip()
            if source_split not in {"val", "test", "train"}:
                raise AdapterError(
                    "NExT-GQA directory input requires source_split=val|test|train"
                )
            csv_path = annotation / f"{source_split}.csv"
        else:
            csv_path = annotation
            source_split = str(self.config.options.get("source_split") or csv_path.stem)
        if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
            raise AdapterError(f"NExT-GQA annotation must be a CSV file: {csv_path}")
        base = csv_path.parent
        grounding_option = self.config.options.get("grounding_path")
        grounding_path = (
            Path(grounding_option).expanduser().resolve()
            if grounding_option
            else base / f"gsub_{source_split}.json"
        )
        allow_missing = bool(self.config.options.get("allow_missing_grounding", False))
        if not grounding_path.is_file():
            if not allow_missing:
                raise AdapterError(
                    f"NExT-GQA grounding file not found: {grounding_path}"
                )
            grounding_path = None
        map_option = self.config.options.get("video_map_path")
        map_path = (
            Path(map_option).expanduser().resolve()
            if map_option
            else base / "map_vid_vidorID.json"
        )
        if not map_path.is_file():
            map_path = None
        return csv_path, grounding_path, map_path, source_split

    def iter_records(self) -> Iterable[dict[str, Any]]:
        csv_path, grounding_path, map_path, source_split = self._paths()
        rows = load_json_rows(csv_path)
        grounding = (
            require_mapping(load_json(grounding_path), path=str(grounding_path))
            if grounding_path
            else {}
        )
        video_map = (
            require_mapping(load_json(map_path), path=str(map_path)) if map_path else {}
        )

        for index, row in enumerate(rows):
            path = f"{csv_path}:{index + 2}"
            fallback_id = f"row:{index}"
            raw_video_id = optional_source_id_component(row.get("video_id"))
            raw_qid = optional_source_id_component(row.get("qid"))
            exclusion_id = (
                f"{raw_video_id}:{raw_qid}"
                if raw_video_id is not None and raw_qid is not None
                else fallback_id
            )
            if self.skip_excluded(exclusion_id, raw_location=path):
                continue
            video_id = require_text(row.get("video_id"), path=f"{path}.video_id")
            qid = require_text(row.get("qid"), path=f"{path}.qid")
            question_type = require_text(row.get("type"), path=f"{path}.type")
            choices = parse_candidates(
                [row.get(f"a{choice}") for choice in range(5)], path=f"{path}.a0..a4"
            )
            answer = normalize_text_answer(
                row.get("answer"), choices, path=f"{path}.answer"
            )
            mapped = str(video_map.get(video_id) or video_id)
            mapped_path = Path(mapped)
            mapped_file = (
                mapped_path if mapped_path.suffix else mapped_path.with_suffix(".mp4")
            )
            media_path = resolve_media(
                self.config.media_root,
                (
                    mapped_file,
                    Path("videos") / mapped_file,
                    f"{video_id}.mp4",
                    Path("videos") / f"{video_id}.mp4",
                ),
                require=self.config.require_media,
                search_basename=f"{video_id}.mp4",
            )
            evidence = (
                _spans(
                    grounding,
                    video_id=video_id,
                    qid=qid,
                    path=str(grounding_path),
                )
                if grounding_path
                else []
            )
            grounding_video = (
                require_mapping(
                    grounding[video_id], path=f"{grounding_path}.{video_id}"
                )
                if grounding_path
                else {}
            )
            information_family, depth = _family(question_type)
            source_id = f"{video_id}:{qid}"
            record_id = stable_id(self.name, source_split, source_id)
            yield make_record(
                record_id=record_id,
                source="NExT-GQA",
                benchmark="next_gqa",
                task="mcq",
                media_path=media_path,
                question=require_text(row.get("question"), path=f"{path}.question"),
                choices=choices,
                answer=answer,
                dataset=self.name,
                split=self.config.split,
                information_family=information_family,
                question_family=f"next_gqa:{question_type}",
                reasoning_depth=depth,
                resampling_unit_id=f"next_gqa:video:{mapped}",
                pair_id=f"standalone:{record_id}",
                pair_role=PairRole.STANDALONE.value,
                evidence_spans=evidence,
                oracles=empty_oracles(),
                provenance={
                    "source_id": source_id,
                    "video_id": video_id,
                    "qid": qid,
                    "question_type": question_type,
                    "annotation_file": str(csv_path),
                    "grounding_file": str(grounding_path) if grounding_path else None,
                    "grounding_duration_seconds": grounding_video.get("duration"),
                    "grounding_fps": grounding_video.get("fps"),
                    "video_map_file": str(map_path) if map_path else None,
                    "mapped_video_id": mapped,
                    "source_split": source_split,
                    "answer_index_base": "text",
                },
            )
