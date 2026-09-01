from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from information_upper_bound.schema import InformationFamily, PairRole

from .common import (
    AdapterError,
    BaseAdapter,
    empty_oracles,
    load_json,
    make_record,
    normalize_integer_answer,
    parse_candidates,
    require_list,
    require_mapping,
    require_text,
    resolve_media,
    source_id_component,
    stable_id,
)


def _finite_number(value: Any, *, path: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{path}: expected a number") from exc
    if not math.isfinite(numeric):
        raise AdapterError(f"{path}: expected a finite number")
    return numeric


def _strictly_increasing(values: Sequence[Any], *, path: str) -> None:
    numeric = [
        _finite_number(value, path=f"{path}[{index}]")
        for index, value in enumerate(values)
    ]
    if any(right <= left for left, right in zip(numeric, numeric[1:])):
        raise AdapterError(f"{path}: values must be strictly increasing")


def _validate_track(track: Mapping[str, Any], *, path: str) -> None:
    require_text(track.get("label"), path=f"{path}.label")
    arrays = {}
    for key in (
        "bounding_boxes",
        "initial_tracking_box",
        "frame_ids",
        "timestamps",
        "is_masked",
    ):
        arrays[key] = require_list(track.get(key), path=f"{path}.{key}")
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
        raise AdapterError(
            f"{path}: track arrays must have the same positive length, got {lengths}"
        )
    _strictly_increasing(arrays["frame_ids"], path=f"{path}.frame_ids")
    _strictly_increasing(arrays["timestamps"], path=f"{path}.timestamps")
    for index, raw_box in enumerate(arrays["bounding_boxes"]):
        box = require_list(raw_box, path=f"{path}.bounding_boxes[{index}]")
        if len(box) != 4:
            raise AdapterError(
                f"{path}.bounding_boxes[{index}]: expected [x1,y1,x2,y2]"
            )
        x1, y1, x2, y2 = [
            _finite_number(value, path=f"{path}.bounding_boxes[{index}]")
            for value in box
        ]
        if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
            raise AdapterError(
                f"{path}.bounding_boxes[{index}]: coordinates must be normalized"
            )


def _validate_segment(
    event: Mapping[str, Any],
    *,
    path: str,
    object_ids: set[int],
) -> tuple[float, float, int, int]:
    require_text(event.get("label"), path=f"{path}.label")
    timestamps = require_list(event.get("timestamps"), path=f"{path}.timestamps")
    frames = require_list(event.get("frame_ids"), path=f"{path}.frame_ids")
    if len(timestamps) != 2 or len(frames) != 2:
        raise AdapterError(f"{path}: timestamps and frame_ids must be [start, end]")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in frames):
        raise AdapterError(f"{path}.frame_ids: start and end must be integers")
    start_frame, end_frame = frames
    if start_frame < 0 or end_frame < start_frame:
        raise AdapterError(
            f"{path}.frame_ids: invalid interval [{start_frame}, {end_frame}]"
        )
    start_us = _finite_number(timestamps[0], path=f"{path}.timestamps[0]")
    end_us = _finite_number(timestamps[1], path=f"{path}.timestamps[1]")
    if start_us < 0 or end_us <= start_us:
        raise AdapterError(
            f"{path}.timestamps: invalid interval [{start_us}, {end_us}]"
        )
    parent_objects = require_list(
        event.get("parent_objects", []), path=f"{path}.parent_objects"
    )
    for value in parent_objects:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdapterError(f"{path}.parent_objects: IDs must be integers")
        if value != -1 and value not in object_ids:
            raise AdapterError(f"{path}.parent_objects: unknown object ID {value}")
    return start_us / 1_000_000.0, end_us / 1_000_000.0, start_frame, end_frame


def _information_family(area: str, reasoning: str, tags: Sequence[str]) -> str:
    text = " ".join([area, reasoning, *tags]).casefold()
    if (
        reasoning.casefold() in {"explanatory", "predictive", "counterfactual"}
        or "physics" in text
    ):
        return InformationFamily.CAUSAL_COMPOSITIONAL.value
    if any(
        token in text for token in ("memory", "track", "occlusion", "object permanence")
    ):
        return InformationFamily.BINDING_TRACKING.value
    if any(token in text for token in ("duration", "count", "speed", "frequency")):
        return InformationFamily.METRIC_TEMPORAL.value
    if any(
        token in text for token in ("before", "after", "order", "sequence", "temporal")
    ):
        return InformationFamily.TEMPORAL_ORDER.value
    if any(token in text for token in ("motion", "direction", "action recognition")):
        return InformationFamily.LOCAL_MOTION.value
    return InformationFamily.STATIC.value


def _reasoning_depth(reasoning: str) -> int:
    return {
        "descriptive": 0,
        "explanatory": 2,
        "predictive": 2,
        "counterfactual": 3,
    }.get(reasoning.casefold(), 1)


def _build_oracles(
    video: Mapping[str, Any],
    *,
    video_id: str,
    include_track_geometry: bool,
    include_audio: bool,
    cut_frame: int | None,
) -> dict[str, Any]:
    oracles = empty_oracles()
    raw_tracks = require_list(
        video.get("object_tracking", []), path=f"{video_id}.object_tracking"
    )
    object_ids: set[int] = set()
    tracks: list[dict[str, Any]] = []
    for index, raw_track in enumerate(raw_tracks):
        path = f"{video_id}.object_tracking[{index}]"
        track = require_mapping(raw_track, path=path)
        track_id = track.get("id")
        if (
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id in object_ids
        ):
            raise AdapterError(f"{path}.id: expected a unique integer")
        object_ids.add(track_id)
        _validate_track(track, path=path)
        retained_indices = [
            sample_index
            for sample_index, frame_id in enumerate(track["frame_ids"])
            if cut_frame is None or frame_id < cut_frame
        ]
        # An object whose first annotation is after the visible clip boundary
        # must not leak into either the static-fact or persistent-track oracle.
        if not retained_indices:
            continue
        label = require_text(track.get("label"), path=f"{path}.label")
        oracles["static_facts"].append(
            {
                "entity_id": track_id,
                "label": label,
                "text": f"Entity {track_id} is a {label}.",
            }
        )
        relation: dict[str, Any] = {
            "entity_id": track_id,
            "text": f"Entity {track_id} ({label}) is one persistent object track.",
            "frame_ids": [
                track["frame_ids"][sample_index] for sample_index in retained_indices
            ],
        }
        if include_track_geometry:
            relation["timestamps_us"] = [
                track["timestamps"][sample_index] for sample_index in retained_indices
            ]
            relation["bounding_boxes"] = [
                track["bounding_boxes"][sample_index]
                for sample_index in retained_indices
            ]
            relation["is_masked"] = [
                track["is_masked"][sample_index] for sample_index in retained_indices
            ]
        tracks.append(relation)
    oracles["relations"] = tracks

    raw_actions = require_list(
        video.get("action_localisation", []), path=f"{video_id}.action_localisation"
    )
    ordered: list[dict[str, Any]] = []
    for index, raw_event in enumerate(raw_actions):
        path = f"{video_id}.action_localisation[{index}]"
        event = require_mapping(raw_event, path=path)
        start, end, _, end_frame = _validate_segment(
            event, path=path, object_ids=object_ids
        )
        if cut_frame is not None and end_frame >= cut_frame:
            # The official visible range is [0, cut_frame). Crossing,
            # boundary-ending, and post-cut events are hidden; retaining even
            # the label would reveal information unavailable in the clip.
            continue
        label = str(event["label"]).strip()
        raw_event_id = event.get("id")
        event_id = (
            str(raw_event_id)
            if raw_event_id not in (None, "")
            else stable_id("perception_action", video_id, index)
        )
        base = {
            "event_id": event_id,
            "label": label,
            "parent_objects": list(event.get("parent_objects") or []),
            "text": label,
        }
        oracles["unordered_events"].append(base)
        ordered.append({**base, "start": start, "end": end, "unit": "seconds"})
    oracles["ordered_events"] = sorted(
        ordered, key=lambda value: (value["start"], value["end"])
    )

    raw_sounds = require_list(
        video.get("sound_localisation", []), path=f"{video_id}.sound_localisation"
    )
    for index, raw_event in enumerate(raw_sounds):
        event = require_mapping(
            raw_event, path=f"{video_id}.sound_localisation[{index}]"
        )
        _, _, _, end_frame = _validate_segment(
            event,
            path=f"{video_id}.sound_localisation[{index}]",
            object_ids=object_ids,
        )
        if include_audio and (cut_frame is None or end_frame < cut_frame):
            oracles["unordered_events"].append(
                {
                    "event_id": event.get("id"),
                    "label": event.get("label"),
                    "modality": "audio",
                    "text": str(event.get("label")),
                }
            )
    return oracles


class PerceptionTestAdapter(BaseAdapter):
    """MCQ adapter with validated object/action annotations from Perception Test."""

    name = "perception_test"

    def iter_records(self) -> Iterable[dict[str, Any]]:
        source = require_mapping(
            load_json(self.config.annotation_path),
            path=str(self.config.annotation_path),
        )
        cut_path = self.config.options.get("cut_frame_mapping_path")
        cuts = (
            require_mapping(load_json(cut_path), path=str(cut_path)) if cut_path else {}
        )
        include_tracks = bool(self.config.options.get("include_track_geometry", False))
        include_audio = bool(self.config.options.get("include_audio_oracles", False))
        allow_missing_cut_mapping = bool(
            self.config.options.get("allow_missing_cut_mapping", False)
            or self.config.options.get("allow_uncut_cup_games", False)
        )

        for video_id in sorted(source):
            video = require_mapping(source[video_id], path=video_id)
            metadata = require_mapping(
                video.get("metadata"), path=f"{video_id}.metadata"
            )
            if str(metadata.get("video_id")) != video_id:
                raise AdapterError(
                    f"{video_id}.metadata.video_id does not match top-level key"
                )
            _finite_number(
                metadata.get("frame_rate"), path=f"{video_id}.metadata.frame_rate"
            )
            num_frames = metadata.get("num_frames")
            if (
                isinstance(num_frames, bool)
                or not isinstance(num_frames, int)
                or num_frames <= 0
            ):
                raise AdapterError(
                    f"{video_id}.metadata.num_frames must be a positive integer"
                )
            split = str(metadata.get("split") or self.config.split)
            media_path = resolve_media(
                self.config.media_root,
                (
                    f"{video_id}.mp4",
                    Path(split) / f"{video_id}.mp4",
                    Path(f"{split}_videos") / f"{video_id}.mp4",
                    Path("videos") / split / f"{video_id}.mp4",
                ),
                require=self.config.require_media,
                search_basename=f"{video_id}.mp4",
            )
            extra_diagnostic: dict[str, Any] = {}
            questions = require_list(
                video.get("mc_question", []), path=f"{video_id}.mc_question"
            )
            raw_cut: int | None = None
            if questions and split in {"train", "valid", "validation"}:
                if video_id not in cuts and not allow_missing_cut_mapping:
                    raise AdapterError(
                        f"{video_id}: validation/train MCQ requires the official cut_frame_mapping "
                        "to prevent end-of-video answer leakage"
                    )
                if video_id in cuts:
                    raw_cut = cuts[video_id]
                    if isinstance(raw_cut, bool) or not isinstance(raw_cut, int):
                        raise AdapterError(f"{video_id}: invalid cut frame {raw_cut!r}")
                    if raw_cut != -1:
                        if not 0 < raw_cut <= num_frames:
                            raise AdapterError(
                                f"{video_id}: invalid cut frame {raw_cut!r}"
                            )
                        extra_diagnostic["media_clip"] = {
                            "start_frame": 0,
                            "end_frame_exclusive": raw_cut,
                            "expected_total_frames": num_frames,
                            "unit": "frames",
                            "source": "official_cut_frame_mapping",
                        }

            cut_frame = raw_cut if raw_cut not in (None, -1) else None
            oracles = _build_oracles(
                video,
                video_id=video_id,
                include_track_geometry=include_tracks,
                include_audio=include_audio,
                cut_frame=cut_frame,
            )

            for index, raw_question in enumerate(questions):
                path = f"{video_id}.mc_question[{index}]"
                raw_qid = (
                    raw_question.get("id")
                    if isinstance(raw_question, Mapping)
                    else None
                )
                qid_component = source_id_component(raw_qid, fallback=f"row:{index}")
                exclusion_id = f"{video_id}:mc_question:{qid_component}"
                if self.skip_excluded(exclusion_id, raw_location=path):
                    continue
                question = require_mapping(raw_question, path=path)
                qid = question.get("id")
                if isinstance(qid, bool) or not isinstance(qid, int):
                    raise AdapterError(f"{path}.id: expected an integer")
                choices = parse_candidates(
                    question.get("options"), path=f"{path}.options"
                )
                answer = normalize_integer_answer(
                    question.get("answer_id"), choices, base=0, path=f"{path}.answer_id"
                )
                area = require_text(question.get("area"), path=f"{path}.area")
                reasoning = require_text(
                    question.get("reasoning"), path=f"{path}.reasoning"
                )
                tags = [
                    str(value).strip()
                    for value in require_list(
                        question.get("tag", []), path=f"{path}.tag"
                    )
                ]
                source_id = f"{video_id}:mc_question:{qid}"
                record_id = stable_id(self.name, source_id)
                yield make_record(
                    record_id=record_id,
                    source="Perception Test",
                    benchmark="perception_test:mcq",
                    task="mcq",
                    media_path=media_path,
                    question=require_text(
                        question.get("question"), path=f"{path}.question"
                    ),
                    choices=choices,
                    answer=answer,
                    dataset=self.name,
                    split=self.config.split,
                    information_family=_information_family(area, reasoning, tags),
                    question_family=f"perception_test:{area.casefold()}:{reasoning.casefold()}",
                    reasoning_depth=_reasoning_depth(reasoning),
                    resampling_unit_id=f"perception_test:video:{video_id}",
                    pair_id=f"standalone:{record_id}",
                    pair_role=PairRole.STANDALONE.value,
                    evidence_spans=[],
                    oracles=oracles,
                    provenance={
                        "source_id": source_id,
                        "video_id": video_id,
                        "annotation_file": str(self.config.annotation_path),
                        "cut_frame_mapping_file": str(
                            Path(cut_path).expanduser().resolve()
                        )
                        if cut_path
                        else None,
                        "source_cut_frame": raw_cut,
                        "source_split": split,
                        "answer_index_base": 0,
                        "area": area,
                        "reasoning": reasoning,
                        "tags": tags,
                        "timestamps_unit": "microseconds",
                        "track_geometry_included": include_tracks,
                        "audio_oracles_included": include_audio,
                    },
                    extra_diagnostic=extra_diagnostic,
                )
