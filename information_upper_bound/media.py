"""Timestamp-aware, deterministic video view sampling.

This module keeps *selection* separate from *decoding*.  The selection API only
needs video metadata, which makes it straightforward to test and to audit.  The
decoder then requests exactly the selected source-frame indices, preferring
decord and falling back to OpenCV.

For time-aligned operations, average FPS is not treated as a frame clock.  When
decord exposes per-frame timestamp intervals, those decoder timestamps drive
evidence and clip overlap, including for variable-frame-rate media.  Metadata
obtained from a backend that only exposes an average FPS is marked unverified;
second-based evidence/clip sampling then fails explicitly instead of silently
misaligning annotations.  A directly constructed ``VideoMetadata`` remains a
trusted declared-CFR contract for callers that already know their media is CFR.

Frame/span convention
---------------------
An evidence span is half-open, ``[start, end)``.  A source frame is considered
part of a span when its nominal frame interval overlaps that span.  Consequently
even a sub-frame-duration annotation resolves to at least one frame (provided it
overlaps the video).  All public sampling results contain the resolved spans,
indices, and timestamps required to reproduce the view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Any, Mapping, Sequence

from PIL import Image


VALID_VIEWS = (
    "full",
    "single",
    "reverse",
    "shuffle",
    "evidence_only",
    "evidence_present",
    "evidence_removed",
    "random_position_mask",
    "random_matched",
)
EVIDENCE_REQUIRED_VIEWS = frozenset(
    {
        "evidence_only",
        "evidence_present",
        "evidence_removed",
        "random_position_mask",
        "random_matched",
    }
)


class EvidenceSpanError(ValueError):
    """Raised when evidence annotations are missing or malformed."""


class ViewSamplingError(ValueError):
    """Raised when a requested view cannot be constructed without confounding."""


class VideoDecodeError(RuntimeError):
    """Raised when video metadata or requested frames cannot be decoded."""


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceSpanError(f"{name} must be a finite number, got {value!r}.")
    converted = float(value)
    if not math.isfinite(converted):
        raise EvidenceSpanError(f"{name} must be finite, got {value!r}.")
    return converted


def normalize_span_unit(unit: str) -> str:
    value = str(unit).strip().lower().replace("-", "_")
    if value in {"s", "sec", "secs", "second", "seconds"}:
        return "seconds"
    if value in {"norm", "normalized", "relative", "fraction", "fractions"}:
        return "normalized"
    raise EvidenceSpanError(
        f"Unsupported evidence span unit {unit!r}; expected 'seconds' or 'normalized'."
    )


@dataclass(frozen=True)
class EvidenceSpan:
    """Unresolved half-open evidence span in seconds or normalized video time."""

    start: float
    end: float
    unit: str = "seconds"

    def __post_init__(self) -> None:
        start = _finite_number(self.start, name="evidence start")
        end = _finite_number(self.end, name="evidence end")
        unit = normalize_span_unit(self.unit)
        if start < 0:
            raise EvidenceSpanError(f"Evidence start must be >= 0, got {start}.")
        if end <= start:
            raise EvidenceSpanError(
                f"Evidence span must have positive duration, got start={start}, end={end}."
            )
        if unit == "normalized" and end > 1:
            raise EvidenceSpanError(
                f"Normalized evidence values must be within [0, 1], got start={start}, end={end}."
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "unit", unit)

    @classmethod
    def from_value(cls, value: Any, *, default_unit: str = "seconds") -> "EvidenceSpan":
        """Parse a span from a dataclass, mapping, or two-number sequence.

        Supported mapping spellings are ``start``/``end`` (with an optional
        ``unit``), ``start_sec``/``end_sec``, and
        ``start_normalized``/``end_normalized``.
        """

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "start_sec" in value or "end_sec" in value:
                if "start_sec" not in value or "end_sec" not in value:
                    raise EvidenceSpanError(
                        "A seconds span must contain both start_sec and end_sec."
                    )
                return cls(value["start_sec"], value["end_sec"], "seconds")
            normalized_start_keys = ("start_normalized", "start_norm")
            normalized_end_keys = ("end_normalized", "end_norm")
            start_key = next(
                (key for key in normalized_start_keys if key in value), None
            )
            end_key = next((key for key in normalized_end_keys if key in value), None)
            if start_key is not None or end_key is not None:
                if start_key is None or end_key is None:
                    raise EvidenceSpanError(
                        "A normalized span must contain both normalized start and end values."
                    )
                return cls(value[start_key], value[end_key], "normalized")
            if "start" not in value or "end" not in value:
                raise EvidenceSpanError(
                    "Evidence span mapping must contain start/end, start_sec/end_sec, "
                    "or normalized start/end keys."
                )
            return cls(value["start"], value["end"], value.get("unit", default_unit))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise EvidenceSpanError(
                    f"Evidence span sequence must have two values, got {len(value)}."
                )
            return cls(value[0], value[1], default_unit)
        raise EvidenceSpanError(
            f"Cannot parse evidence span from {type(value).__name__}: {value!r}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "unit": self.unit}


@dataclass(frozen=True)
class ResolvedEvidenceSpan:
    """Evidence span resolved to seconds and overlapping source-frame bounds."""

    start: float
    end: float
    unit: str
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame_exclusive: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoMetadata:
    total_frames: int
    fps: float
    duration_sec: float
    backend: str | None = None
    frame_intervals_sec: tuple[tuple[float, float], ...] | None = None
    timestamp_source: str | None = None
    frame_timestamp_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_frames, bool)
            or int(self.total_frames) != self.total_frames
        ):
            raise ValueError(
                f"total_frames must be an integer, got {self.total_frames!r}."
            )
        if int(self.total_frames) <= 0:
            raise ValueError(f"total_frames must be positive, got {self.total_frames}.")
        fps = float(self.fps)
        duration = float(self.duration_sec)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {self.fps!r}.")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"duration_sec must be finite and positive, got {self.duration_sec!r}."
            )
        intervals: tuple[tuple[float, float], ...] | None = None
        timestamp_digest: str | None = None
        if self.frame_intervals_sec is not None:
            if len(self.frame_intervals_sec) != int(self.total_frames):
                raise ValueError(
                    "frame_intervals_sec must contain exactly one interval per frame, "
                    f"got {len(self.frame_intervals_sec)} for {self.total_frames} frames."
                )
            parsed: list[tuple[float, float]] = []
            previous_start = -math.inf
            previous_end = -math.inf
            digest = hashlib.sha256()
            tolerance = max(1e-9, duration * 1e-9)
            for index, value in enumerate(self.frame_intervals_sec):
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ValueError(
                        f"frame interval {index} must be a start/end sequence, got {value!r}."
                    )
                if len(value) != 2:
                    raise ValueError(
                        f"frame interval {index} must contain two values, got {len(value)}."
                    )
                start = float(value[0])
                end = float(value[1])
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError(
                        f"frame interval {index} must be finite, got ({start}, {end})."
                    )
                if start < -tolerance:
                    raise ValueError(
                        f"frame interval {index} starts before media time zero: {start}."
                    )
                start = max(0.0, start)
                if end <= start:
                    raise ValueError(
                        f"frame interval {index} must have positive duration, got ({start}, {end})."
                    )
                if start < previous_start or end < previous_end:
                    raise ValueError(
                        "frame_intervals_sec must be in nondecreasing presentation order; "
                        f"interval {index} is ({start}, {end}) after "
                        f"({previous_start}, {previous_end})."
                    )
                if end > duration + tolerance:
                    raise ValueError(
                        f"frame interval {index} ends at {end}s beyond duration {duration}s."
                    )
                parsed.append((start, end))
                digest.update(struct.pack("!dd", start, end))
                previous_start = start
                previous_end = end
            intervals = tuple(parsed)
            timestamp_digest = digest.hexdigest()
            if (
                self.frame_timestamp_sha256 is not None
                and str(self.frame_timestamp_sha256) != timestamp_digest
            ):
                raise ValueError(
                    "frame_timestamp_sha256 does not match frame_intervals_sec."
                )
        elif self.frame_timestamp_sha256 is not None:
            raise ValueError(
                "frame_timestamp_sha256 requires frame_intervals_sec; a digest cannot "
                "establish a frame clock by itself."
            )

        source = self.timestamp_source
        if source is None:
            if intervals is not None:
                source = "explicit_frame_intervals"
            elif str(self.backend or "").lower() == "opencv":
                source = "opencv_average_fps_unverified"
            else:
                source = "declared_cfr"
        source = str(source).strip()
        if not source:
            raise ValueError(
                "timestamp_source must be a non-empty string when provided."
            )

        if intervals is None and source == "declared_cfr":
            declared_duration = int(self.total_frames) / fps
            tolerance = max(1e-9, declared_duration * 1e-9)
            if abs(duration - declared_duration) > tolerance:
                raise ValueError(
                    "declared_cfr metadata requires duration_sec == total_frames / fps; "
                    f"got duration={duration}, frame-derived duration={declared_duration}."
                )
        object.__setattr__(self, "total_frames", int(self.total_frames))
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "duration_sec", duration)
        object.__setattr__(self, "frame_intervals_sec", intervals)
        object.__setattr__(self, "timestamp_source", source)
        object.__setattr__(self, "frame_timestamp_sha256", timestamp_digest)

    @property
    def time_alignment_guaranteed(self) -> bool:
        """Whether seconds can be mapped to source frames without average-FPS inference."""

        return (
            self.frame_intervals_sec is not None
            or self.timestamp_source == "declared_cfr"
        )

    def frame_start_sec(self, index: int) -> float:
        if self.frame_intervals_sec is not None:
            return self.frame_intervals_sec[index][0]
        return index / self.fps

    def frame_interval_sec(self, index: int) -> tuple[float, float]:
        if self.frame_intervals_sec is not None:
            return self.frame_intervals_sec[index]
        return index / self.fps, (index + 1) / self.fps

    def to_dict(self) -> dict[str, Any]:
        # Do not duplicate an O(number-of-frames) timestamp table in every trial
        # record.  The digest plus selected timestamps keeps provenance compact.
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "backend": self.backend,
            "timestamp_source": self.timestamp_source,
            "time_alignment_guaranteed": self.time_alignment_guaranteed,
            "frame_timestamp_sha256": self.frame_timestamp_sha256,
        }


@dataclass(frozen=True)
class ViewSpec:
    """A reproducible request for one model-ready view of a video."""

    view: str
    num_frames: int
    seed: int = 42
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    clip_start_sec: float = 0.0
    clip_end_sec: float | None = None
    clip_start_frame: int | None = None
    clip_end_frame_exclusive: int | None = None
    clip_expected_total_frames: int | None = None

    def __post_init__(self) -> None:
        view = str(self.view).strip().lower()
        if view not in VALID_VIEWS:
            raise ValueError(
                f"Unknown view {self.view!r}; expected one of {VALID_VIEWS}."
            )
        if isinstance(self.num_frames, bool) or int(self.num_frames) != self.num_frames:
            raise ValueError(f"num_frames must be an integer, got {self.num_frames!r}.")
        if int(self.num_frames) <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError(f"seed must be an integer, got {self.seed!r}.")
        spans = tuple(EvidenceSpan.from_value(span) for span in self.evidence_spans)
        clip_start = _finite_number(self.clip_start_sec, name="clip_start_sec")
        clip_end = (
            _finite_number(self.clip_end_sec, name="clip_end_sec")
            if self.clip_end_sec is not None
            else None
        )
        if clip_start < 0:
            raise ValueError(f"clip_start_sec must be >= 0, got {clip_start}.")
        if clip_end is not None and clip_end <= clip_start:
            raise ValueError(
                f"clip_end_sec must be greater than clip_start_sec, got {clip_end} <= {clip_start}."
            )
        frame_values = (
            self.clip_start_frame,
            self.clip_end_frame_exclusive,
            self.clip_expected_total_frames,
        )
        uses_frame_clip = any(value is not None for value in frame_values)
        if uses_frame_clip:
            if clip_start != 0.0 or clip_end is not None:
                raise ValueError(
                    "second-based and frame-based clip bounds are mutually exclusive."
                )
            if self.clip_end_frame_exclusive is None:
                raise ValueError("frame-based clip requires clip_end_frame_exclusive.")
            start_frame = 0 if self.clip_start_frame is None else self.clip_start_frame
            end_frame = self.clip_end_frame_exclusive
            expected_total = self.clip_expected_total_frames
            for name, value in (
                ("clip_start_frame", start_frame),
                ("clip_end_frame_exclusive", end_frame),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer, got {value!r}.")
            if start_frame < 0 or end_frame <= start_frame:
                raise ValueError(
                    "frame clip requires 0 <= start_frame < end_frame_exclusive."
                )
            if expected_total is not None:
                if isinstance(expected_total, bool) or not isinstance(
                    expected_total, int
                ):
                    raise ValueError("clip_expected_total_frames must be an integer.")
                if expected_total < end_frame:
                    raise ValueError(
                        "clip_expected_total_frames must be >= clip_end_frame_exclusive."
                    )
            object.__setattr__(self, "clip_start_frame", int(start_frame))
            object.__setattr__(self, "clip_end_frame_exclusive", int(end_frame))
            object.__setattr__(
                self,
                "clip_expected_total_frames",
                int(expected_total) if expected_total is not None else None,
            )
        if view in EVIDENCE_REQUIRED_VIEWS and not spans:
            raise EvidenceSpanError(
                f"View {view!r} requires at least one evidence span."
            )
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "num_frames", int(self.num_frames))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "evidence_spans", spans)
        object.__setattr__(self, "clip_start_sec", clip_start)
        object.__setattr__(self, "clip_end_sec", clip_end)

    @classmethod
    def create(
        cls,
        *,
        view: str,
        num_frames: int,
        seed: int = 42,
        evidence_spans: Sequence[Any] | None = None,
        default_evidence_unit: str = "seconds",
        clip: Mapping[str, Any] | None = None,
    ) -> "ViewSpec":
        parsed = tuple(
            EvidenceSpan.from_value(span, default_unit=default_evidence_unit)
            for span in (evidence_spans or ())
        )
        clip_value = dict(clip or {})
        unit = str(clip_value.get("unit", "seconds"))
        if unit not in {"seconds", "frames"}:
            raise ValueError(
                "media clip bounds require unit='seconds' or unit='frames'."
            )
        if unit == "frames":
            end_frame = clip_value.get("end_frame_exclusive", clip_value.get("end"))
            if end_frame is None:
                raise ValueError("frame clip requires end_frame_exclusive.")
            return cls(
                view=view,
                num_frames=num_frames,
                seed=seed,
                evidence_spans=parsed,
                clip_start_frame=int(
                    clip_value.get("start_frame", clip_value.get("start", 0))
                ),
                clip_end_frame_exclusive=int(end_frame),
                clip_expected_total_frames=(
                    int(clip_value["expected_total_frames"])
                    if clip_value.get("expected_total_frames") is not None
                    else None
                ),
            )
        return cls(
            view=view,
            num_frames=num_frames,
            seed=seed,
            evidence_spans=parsed,
            clip_start_sec=float(clip_value.get("start", 0.0)),
            clip_end_sec=(
                float(clip_value["end"]) if clip_value.get("end") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "num_frames": self.num_frames,
            "seed": self.seed,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "clip_start_sec": self.clip_start_sec,
            "clip_end_sec": self.clip_end_sec,
            "clip_start_frame": self.clip_start_frame,
            "clip_end_frame_exclusive": self.clip_end_frame_exclusive,
            "clip_expected_total_frames": self.clip_expected_total_frames,
        }


@dataclass(frozen=True)
class FrameSelection:
    """Selected source frames plus complete sampling provenance."""

    visual_id: str
    view_spec: ViewSpec
    video: VideoMetadata
    indices: tuple[int, ...]
    timestamps_sec: tuple[float, ...]
    resolved_evidence_spans: tuple[ResolvedEvidenceSpan, ...]
    evidence_source_frame_count: int
    candidate_source_frame_count: int
    evidence_input_positions: tuple[int, ...] = ()
    masked_input_positions: tuple[int, ...] = ()
    mask_strategy: str | None = None
    random_match_strategy: str | None = None
    random_match_target_duration_sec: float | None = None
    random_match_actual_duration_sec: float | None = None
    random_match_error_sec: float | None = None
    random_match_spans_sec: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_id": self.visual_id,
            "view_spec": self.view_spec.to_dict(),
            "video": self.video.to_dict(),
            "selected_indices": list(self.indices),
            "selected_timestamps_sec": list(self.timestamps_sec),
            "resolved_evidence_spans": [
                span.to_dict() for span in self.resolved_evidence_spans
            ],
            "evidence_source_frame_count": self.evidence_source_frame_count,
            "candidate_source_frame_count": self.candidate_source_frame_count,
            "evidence_input_positions": list(self.evidence_input_positions),
            "masked_input_positions": list(self.masked_input_positions),
            "mask_target_count": len(self.masked_input_positions),
            "mask_evidence_overlap_count": len(
                set(self.masked_input_positions) & set(self.evidence_input_positions)
            ),
            "mask_non_evidence_count": len(
                set(self.masked_input_positions) - set(self.evidence_input_positions)
            ),
            "mask_strategy": self.mask_strategy,
            "random_match_strategy": self.random_match_strategy,
            "random_match_target_duration_sec": self.random_match_target_duration_sec,
            "random_match_actual_duration_sec": self.random_match_actual_duration_sec,
            "random_match_error_sec": self.random_match_error_sec,
            "random_match_spans_sec": [
                list(span) for span in self.random_match_spans_sec
            ],
        }


@dataclass
class DecodedVideoView:
    frames: list[Image.Image]
    selection: FrameSelection


def validate_evidence_spans(
    spans: Sequence[Any] | None,
    *,
    duration_sec: float,
    total_frames: int,
    fps: float,
    default_unit: str = "seconds",
    frame_intervals_sec: Sequence[Sequence[float]] | None = None,
    timestamp_source: str | None = None,
) -> tuple[ResolvedEvidenceSpan, ...]:
    """Validate and resolve evidence spans against one video's frame clock.

    ``frame_intervals_sec`` should contain decoder-provided ``[start, end)``
    intervals in presentation order.  When it is absent, this public helper
    treats the supplied ``fps``/``duration_sec`` tuple as a declared-CFR
    contract unless an explicitly unverified ``timestamp_source`` is supplied.
    """

    metadata = VideoMetadata(
        total_frames=total_frames,
        fps=fps,
        duration_sec=duration_sec,
        frame_intervals_sec=(
            tuple((float(value[0]), float(value[1])) for value in frame_intervals_sec)
            if frame_intervals_sec is not None
            else None
        ),
        timestamp_source=timestamp_source,
    )
    return _resolve_evidence_spans(
        spans,
        metadata=metadata,
        default_unit=default_unit,
    )


def _first_frame_ending_after(
    intervals: Sequence[tuple[float, float]], value: float
) -> int:
    low = 0
    high = len(intervals)
    while low < high:
        middle = (low + high) // 2
        if intervals[middle][1] <= value:
            low = middle + 1
        else:
            high = middle
    return low


def _first_frame_starting_at_or_after(
    intervals: Sequence[tuple[float, float]], value: float
) -> int:
    low = 0
    high = len(intervals)
    while low < high:
        middle = (low + high) // 2
        if intervals[middle][0] < value:
            low = middle + 1
        else:
            high = middle
    return low


def _time_overlap_bounds(
    metadata: VideoMetadata,
    *,
    start_sec: float,
    end_sec: float,
) -> tuple[int, int] | None:
    """Find the contiguous frame-index bounds overlapping ``[start, end)``."""

    if metadata.frame_intervals_sec is None:
        start_frame = max(
            0,
            min(metadata.total_frames - 1, math.floor(start_sec * metadata.fps)),
        )
        end_frame = max(
            start_frame + 1,
            min(metadata.total_frames, math.ceil(end_sec * metadata.fps - 1e-12)),
        )
        return start_frame, end_frame

    intervals = metadata.frame_intervals_sec
    start_frame = _first_frame_ending_after(intervals, start_sec)
    end_frame = _first_frame_starting_at_or_after(intervals, end_sec)
    if start_frame >= end_frame:
        return None
    # Defensive checks make the half-open contract explicit even if tolerance
    # accepted almost-equal timestamp values during metadata validation.
    if intervals[start_frame][0] >= end_sec or intervals[end_frame - 1][1] <= start_sec:
        return None
    return start_frame, end_frame


def _resolve_evidence_spans(
    spans: Sequence[Any] | None,
    *,
    metadata: VideoMetadata,
    default_unit: str = "seconds",
) -> tuple[ResolvedEvidenceSpan, ...]:
    parsed = [
        EvidenceSpan.from_value(value, default_unit=default_unit)
        for value in (spans or ())
    ]
    if parsed and not metadata.time_alignment_guaranteed:
        raise ViewSamplingError(
            "Evidence alignment requires decoder-provided frame timestamps or an explicit "
            "declared-CFR frame clock; average FPS metadata is not sufficient. "
            f"timestamp_source={metadata.timestamp_source!r}."
        )
    resolved: list[ResolvedEvidenceSpan] = []
    for span in parsed:
        if span.unit == "normalized":
            start_sec = span.start * metadata.duration_sec
            end_sec = span.end * metadata.duration_sec
        else:
            start_sec = span.start
            end_sec = span.end
            if end_sec > metadata.duration_sec + 1e-9:
                raise EvidenceSpanError(
                    f"Seconds evidence span [{span.start}, {span.end}) exceeds video duration "
                    f"{metadata.duration_sec:.9g}s."
                )
        bounds = _time_overlap_bounds(
            metadata,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if bounds is None:
            raise EvidenceSpanError(
                f"Evidence span [{start_sec}, {end_sec})s does not overlap any decoded "
                "frame timestamp interval."
            )
        start_frame, end_frame_exclusive = bounds
        resolved.append(
            ResolvedEvidenceSpan(
                start=span.start,
                end=span.end,
                unit=span.unit,
                start_sec=start_sec,
                end_sec=end_sec,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
            )
        )
    return tuple(resolved)


def _time_overlapping_indices(
    metadata: VideoMetadata,
    *,
    start_sec: float,
    end_sec: float,
    context: str,
) -> list[int]:
    """Return frames whose half-open intervals overlap a requested time range."""

    if not metadata.time_alignment_guaranteed:
        raise ViewSamplingError(
            f"{context} requires decoder-provided frame timestamps or an explicit "
            "declared-CFR frame clock; average FPS metadata cannot guarantee alignment. "
            f"timestamp_source={metadata.timestamp_source!r}."
        )
    bounds = _time_overlap_bounds(metadata, start_sec=start_sec, end_sec=end_sec)
    if bounds is None:
        raise ViewSamplingError(
            f"{context} [{start_sec}, {end_sec})s does not overlap any decoded frame interval."
        )
    return list(range(bounds[0], bounds[1]))


def _uniform_select(candidates: Sequence[int], count: int) -> list[int]:
    if not candidates:
        raise ViewSamplingError("Cannot sample from an empty source-frame set.")
    if count == 1:
        return [int(candidates[len(candidates) // 2])]
    last = len(candidates) - 1
    positions = [round(step * last / (count - 1)) for step in range(count)]
    return [int(candidates[position]) for position in positions]


def _evidence_stratified_grid(
    all_indices: Sequence[int],
    evidence: Sequence[int],
    complement: Sequence[int],
    count: int,
) -> list[int]:
    """Select one shared grid for evidence-present and evidence-masked views."""

    if not evidence:
        raise ViewSamplingError(
            "evidence-stratified sampling resolved to no evidence frames."
        )
    evidence_fraction = len(evidence) / len(all_indices)
    evidence_slots = max(1, round(count * evidence_fraction))
    if complement and count > 1:
        evidence_slots = min(evidence_slots, count - 1)
    evidence_slots = min(evidence_slots, count)
    complement_slots = count - evidence_slots
    selected = _uniform_select(evidence, evidence_slots)
    if complement_slots:
        selected.extend(_uniform_select(complement, complement_slots))
    return sorted(selected)


def _stable_rng(seed: int, *, visual_id: str, view: str) -> random.Random:
    payload = json.dumps(
        {"seed": int(seed), "view": view, "visual_id": str(visual_id)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))


def _evidence_indices(
    resolved: Sequence[ResolvedEvidenceSpan], total_frames: int
) -> list[int]:
    evidence: set[int] = set()
    for span in resolved:
        evidence.update(range(span.start_frame, span.end_frame_exclusive))
    return sorted(index for index in evidence if 0 <= index < total_frames)


def _merge_time_intervals(
    intervals: Sequence[tuple[float, float]], *, tolerance: float = 1e-9
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _duration_matched_control(
    metadata: VideoMetadata,
    complement: Sequence[int],
    resolved: Sequence[ResolvedEvidenceSpan],
    *,
    clip_start_sec: float,
    clip_end_sec: float,
    rng: random.Random,
) -> tuple[list[int], tuple[tuple[float, float], ...], float, float, str]:
    """Select non-evidence time support matching the union of evidence durations."""

    evidence_union = _merge_time_intervals(
        [(span.start_sec, span.end_sec) for span in resolved]
    )
    target_duration = sum(end - start for start, end in evidence_union)
    if target_duration <= 0:
        raise ViewSamplingError("random_matched evidence has zero temporal duration.")

    available = _merge_time_intervals(
        [
            (max(start, clip_start_sec), min(end, clip_end_sec))
            for index in complement
            for start, end in [metadata.frame_interval_sec(index)]
            if min(end, clip_end_sec) > max(start, clip_start_sec)
        ]
    )
    available_duration = sum(end - start for start, end in available)
    tolerance = max(1e-9, metadata.duration_sec * 1e-9)
    if available_duration + tolerance < target_duration:
        raise ViewSamplingError(
            "random_matched lacks non-evidence timestamp coverage: "
            f"target={target_duration:.9g}s, available={available_duration:.9g}s."
        )

    contiguous = [
        interval
        for interval in available
        if interval[1] - interval[0] + tolerance >= target_duration
    ]
    selected_windows: list[tuple[float, float]] = []
    if contiguous:
        segment_start, segment_end = contiguous[rng.randrange(len(contiguous))]
        slack = max(0.0, segment_end - segment_start - target_duration)
        window_start = segment_start + (
            rng.random() * slack if slack > tolerance else 0.0
        )
        selected_windows.append((window_start, window_start + target_duration))
        strategy = "contiguous_non_evidence_duration_window"
    else:
        shuffled = list(available)
        rng.shuffle(shuffled)
        remaining = target_duration
        for segment_start, segment_end in shuffled:
            if remaining <= tolerance:
                break
            segment_duration = segment_end - segment_start
            take = min(segment_duration, remaining)
            slack = max(0.0, segment_duration - take)
            window_start = segment_start + (
                rng.random() * slack if slack > tolerance else 0.0
            )
            selected_windows.append((window_start, window_start + take))
            remaining -= take
        if remaining > tolerance:
            raise ViewSamplingError(
                "random_matched could not assemble the requested non-evidence duration; "
                f"unmatched={remaining:.9g}s."
            )
        strategy = "multi_span_non_evidence_duration_match"

    complement_set = set(complement)
    candidates: set[int] = set()
    for window_start, window_end in selected_windows:
        bounds = _time_overlap_bounds(
            metadata, start_sec=window_start, end_sec=window_end
        )
        if bounds is None:
            continue
        candidates.update(
            index for index in range(bounds[0], bounds[1]) if index in complement_set
        )
    if not candidates:
        raise ViewSamplingError(
            "random_matched duration window did not retain any non-evidence source frames."
        )
    actual_duration = sum(end - start for start, end in selected_windows)
    return (
        sorted(candidates),
        tuple(sorted(selected_windows)),
        target_duration,
        actual_duration,
        strategy,
    )


def sample_frame_indices(
    metadata: VideoMetadata,
    spec: ViewSpec,
    *,
    visual_id: str,
) -> FrameSelection:
    """Build deterministic source-frame indices for a diagnostic view.

    ``shuffle`` permutes the uniformly sampled model input frames.  The
    ``evidence_present``, ``evidence_removed``, and ``random_position_mask``
    share one stratified source-frame grid. The latter two use exactly the same
    mask count; the random control prefers non-evidence input positions.
    In contrast,
    ``random_matched`` keeps chronological order while selecting a non-evidence
    control region with exactly the same number of source frames as the union of
    the annotated evidence.
    """

    resolved = _resolve_evidence_spans(
        spec.evidence_spans,
        metadata=metadata,
    )
    uses_frame_clip = spec.clip_end_frame_exclusive is not None
    if uses_frame_clip:
        if (
            spec.clip_expected_total_frames is not None
            and metadata.total_frames != spec.clip_expected_total_frames
        ):
            raise ViewSamplingError(
                "decoder frame count does not match the official frame clip contract: "
                f"decoded={metadata.total_frames}, expected={spec.clip_expected_total_frames}."
            )
        assert spec.clip_start_frame is not None
        assert spec.clip_end_frame_exclusive is not None
        if spec.clip_end_frame_exclusive > metadata.total_frames:
            raise ViewSamplingError(
                f"frame clip end {spec.clip_end_frame_exclusive} exceeds decoded frame count "
                f"{metadata.total_frames}."
            )
        all_indices = list(range(spec.clip_start_frame, spec.clip_end_frame_exclusive))
        clip_start_sec = metadata.frame_interval_sec(spec.clip_start_frame)[0]
        clip_end_sec = metadata.frame_interval_sec(spec.clip_end_frame_exclusive - 1)[1]
    else:
        clip_start_sec = spec.clip_start_sec
        clip_end_sec = (
            spec.clip_end_sec
            if spec.clip_end_sec is not None
            else metadata.duration_sec
        )
    if clip_start_sec >= metadata.duration_sec:
        raise ViewSamplingError(
            f"clip start {clip_start_sec}s is outside video duration {metadata.duration_sec}s."
        )
    if clip_end_sec > metadata.duration_sec + 1e-9:
        raise ViewSamplingError(
            f"clip end {clip_end_sec}s exceeds video duration {metadata.duration_sec}s."
        )
    uses_explicit_clip = (
        uses_frame_clip or spec.clip_start_sec != 0.0 or spec.clip_end_sec is not None
    )
    if uses_explicit_clip and not uses_frame_clip:
        all_indices = _time_overlapping_indices(
            metadata,
            start_sec=spec.clip_start_sec,
            end_sec=clip_end_sec,
            context="media clip alignment",
        )
    elif not uses_frame_clip:
        all_indices = list(range(metadata.total_frames))
    evidence = _evidence_indices(resolved, metadata.total_frames)
    tolerance = max(1e-9, metadata.duration_sec * 1e-9)
    evidence_outside_clip = any(
        span.start_sec < clip_start_sec - tolerance
        or span.end_sec > clip_end_sec + tolerance
        for span in resolved
    )
    if evidence_outside_clip:
        raise ViewSamplingError(
            "evidence spans extend outside the declared media clip; fix the annotation or clip bounds."
        )
    evidence_set = set(evidence)
    complement = [index for index in all_indices if index not in evidence_set]
    random_match_strategy: str | None = None
    random_match_spans: tuple[tuple[float, float], ...] = ()
    random_match_target_duration: float | None = None
    random_match_actual_duration: float | None = None
    masked_input_positions: tuple[int, ...] = ()
    evidence_input_positions: tuple[int, ...] = ()
    mask_strategy: str | None = None

    if spec.view == "full":
        candidates = all_indices
        selected = _uniform_select(candidates, spec.num_frames)
    elif spec.view == "single":
        candidates = [all_indices[len(all_indices) // 2]]
        selected = candidates * spec.num_frames
    elif spec.view == "reverse":
        candidates = all_indices
        selected = list(reversed(_uniform_select(candidates, spec.num_frames)))
    elif spec.view == "shuffle":
        candidates = all_indices
        selected = _uniform_select(candidates, spec.num_frames)
        _stable_rng(spec.seed, visual_id=visual_id, view=spec.view).shuffle(selected)
    elif spec.view == "evidence_only":
        if not evidence:
            raise ViewSamplingError("evidence_only resolved to no source frames.")
        candidates = evidence
        selected = _uniform_select(candidates, spec.num_frames)
    elif spec.view in {
        "evidence_present",
        "evidence_removed",
        "random_position_mask",
    }:
        candidates = all_indices
        selected = _evidence_stratified_grid(
            all_indices,
            evidence,
            complement,
            spec.num_frames,
        )
        evidence_input_positions = tuple(
            position for position, index in enumerate(selected) if index in evidence_set
        )
        if not evidence_input_positions:
            raise ViewSamplingError(
                f"{spec.view} shared grid contains no evidence input position."
            )
        if spec.view == "evidence_removed":
            masked_input_positions = evidence_input_positions
            mask_strategy = "solid_midgray_rgb_on_shared_evidence_grid"
        elif spec.view == "random_position_mask":
            target_count = len(evidence_input_positions)
            evidence_position_set = set(evidence_input_positions)
            non_evidence_positions = [
                position
                for position in range(len(selected))
                if position not in evidence_position_set
            ]
            rng = _stable_rng(
                spec.seed, visual_id=visual_id, view="random_position_mask"
            )
            rng.shuffle(non_evidence_positions)
            chosen = non_evidence_positions[:target_count]
            if len(chosen) < target_count:
                fallback = list(evidence_input_positions)
                rng.shuffle(fallback)
                chosen.extend(fallback[: target_count - len(chosen)])
            masked_input_positions = tuple(sorted(chosen))
            mask_strategy = (
                "solid_midgray_rgb_random_same_count_prefer_non_evidence_on_shared_grid"
            )
    elif spec.view == "random_matched":
        if not evidence:
            raise ViewSamplingError(
                "random_matched resolved to no evidence source frames."
            )
        rng = _stable_rng(spec.seed, visual_id=visual_id, view=spec.view)
        (
            candidates,
            random_match_spans,
            random_match_target_duration,
            random_match_actual_duration,
            random_match_strategy,
        ) = _duration_matched_control(
            metadata,
            complement,
            resolved,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            rng=rng,
        )
        selected = _uniform_select(candidates, spec.num_frames)
    else:  # ViewSpec validates this, but keep this branch defensive.
        raise ValueError(f"Unsupported view: {spec.view}")

    timestamps = tuple(metadata.frame_start_sec(index) for index in selected)
    return FrameSelection(
        visual_id=str(visual_id),
        view_spec=spec,
        video=metadata,
        indices=tuple(selected),
        timestamps_sec=timestamps,
        resolved_evidence_spans=resolved,
        evidence_source_frame_count=len(evidence),
        candidate_source_frame_count=len(candidates),
        evidence_input_positions=evidence_input_positions,
        masked_input_positions=masked_input_positions,
        mask_strategy=mask_strategy,
        random_match_strategy=random_match_strategy,
        random_match_target_duration_sec=random_match_target_duration,
        random_match_actual_duration_sec=random_match_actual_duration,
        random_match_error_sec=(
            abs(random_match_actual_duration - random_match_target_duration)
            if random_match_actual_duration is not None
            and random_match_target_duration is not None
            else None
        ),
        random_match_spans_sec=random_match_spans,
    )


def _validate_decode_indices(indices: Sequence[int], *, total_frames: int) -> list[int]:
    validated: list[int] = []
    if not indices:
        raise VideoDecodeError("At least one frame index is required for decoding.")
    for index in indices:
        if isinstance(index, bool) or int(index) != index:
            raise VideoDecodeError(f"Frame index must be an integer, got {index!r}.")
        converted = int(index)
        if converted < 0 or converted >= total_frames:
            raise VideoDecodeError(
                f"Frame index {converted} is outside [0, {total_frames})."
            )
        validated.append(converted)
    return validated


def _probe_decord(path: Path) -> VideoMetadata:
    import decord

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    total_frames = len(reader)
    fps = float(reader.get_avg_fps())
    if total_frames <= 0:
        raise VideoDecodeError(
            f"decord returned invalid metadata: total_frames={total_frames}, fps={fps}."
        )

    intervals: tuple[tuple[float, float], ...] | None = None
    try:
        raw_timestamps = reader.get_frame_timestamp(list(range(total_frames)))
        values = (
            raw_timestamps.asnumpy().tolist()
            if hasattr(raw_timestamps, "asnumpy")
            else raw_timestamps.tolist()
            if hasattr(raw_timestamps, "tolist")
            else list(raw_timestamps)
        )
        if len(values) != total_frames:
            raise ValueError(
                f"expected {total_frames} frame timestamp rows, got {len(values)}."
            )
        raw_intervals = tuple((float(row[0]), float(row[1])) for row in values)
        # Dataset annotations use time relative to the first presented frame.
        # Preserve all VFR gaps/durations while removing a container PTS origin.
        origin = raw_intervals[0][0]
        intervals = tuple(
            (start - origin, end - origin) for start, end in raw_intervals
        )
    except Exception:
        # Older/limited decord builds may not expose timestamp intervals.  Keep
        # index-only views available, but label the clock unverified so any
        # evidence or explicit clip alignment is rejected by sample_frame_indices.
        intervals = None

    if intervals is not None:
        duration_sec = max(end for _, end in intervals)
        if not math.isfinite(fps) or fps <= 0:
            fps = total_frames / duration_sec
        timestamp_source = "decord_frame_timestamp"
    else:
        if not math.isfinite(fps) or fps <= 0:
            raise VideoDecodeError(
                f"decord returned invalid metadata: total_frames={total_frames}, fps={fps}."
            )
        duration_sec = total_frames / fps
        timestamp_source = "decord_average_fps_unverified"
    return VideoMetadata(
        total_frames=total_frames,
        fps=fps,
        duration_sec=duration_sec,
        backend="decord",
        frame_intervals_sec=intervals,
        timestamp_source=timestamp_source,
    )


def _probe_opencv(path: Path) -> VideoMetadata:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoDecodeError(f"OpenCV could not open video: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
            raise VideoDecodeError(
                f"OpenCV returned invalid metadata: total_frames={total_frames}, fps={fps}."
            )
        return VideoMetadata(
            total_frames=total_frames,
            fps=fps,
            duration_sec=total_frames / fps,
            backend="opencv",
            timestamp_source="opencv_average_fps_unverified",
        )
    finally:
        cap.release()


def _backend_order(backend: str) -> tuple[str, ...]:
    normalized = str(backend).strip().lower()
    if normalized == "auto":
        return ("decord", "opencv")
    if normalized in {"decord", "opencv"}:
        return (normalized,)
    raise ValueError("backend must be 'auto', 'decord', or 'opencv'.")


def probe_video(path: str | Path, *, backend: str = "auto") -> VideoMetadata:
    """Probe metadata with decord first and OpenCV fallback."""

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    errors: list[str] = []
    for candidate in _backend_order(backend):
        try:
            return (
                _probe_decord(video_path)
                if candidate == "decord"
                else _probe_opencv(video_path)
            )
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise VideoDecodeError(
        f"Could not probe video {video_path} with requested backends. "
        + " | ".join(errors)
    )


def _decode_decord(
    path: Path, indices: Sequence[int], metadata: VideoMetadata
) -> list[Image.Image]:
    import decord

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    validated = _validate_decode_indices(indices, total_frames=len(reader))
    unique_indices = sorted(set(validated))
    batch = reader.get_batch(unique_indices).asnumpy()
    decoded = {
        index: Image.fromarray(frame).convert("RGB")
        for index, frame in zip(unique_indices, batch)
    }
    return [decoded[index].copy() for index in validated]


def _decode_opencv(
    path: Path, indices: Sequence[int], metadata: VideoMetadata
) -> list[Image.Image]:
    import cv2

    validated = _validate_decode_indices(indices, total_frames=metadata.total_frames)
    unique_indices = sorted(set(validated))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError(f"OpenCV could not open video: {path}")
    decoded: dict[int, Image.Image] = {}
    try:
        for index in unique_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoDecodeError(
                    f"OpenCV could not decode frame {index} from {path}."
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[index] = Image.fromarray(rgb).convert("RGB")
    finally:
        cap.release()
    return [decoded[index].copy() for index in validated]


def decode_selected_frames(
    path: str | Path,
    indices: Sequence[int],
    *,
    metadata: VideoMetadata | None = None,
    backend: str = "auto",
) -> tuple[list[Image.Image], VideoMetadata]:
    """Decode only unique requested indices, restoring duplicates and order afterward."""

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    errors: list[str] = []
    for candidate in _backend_order(backend):
        try:
            current = (
                metadata
                if metadata and metadata.backend == candidate
                else probe_video(video_path, backend=candidate)
            )
            if candidate == "decord":
                frames = _decode_decord(video_path, indices, current)
            else:
                frames = _decode_opencv(video_path, indices, current)
            return frames, current
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise VideoDecodeError(
        f"Could not decode selected frames from {video_path}. " + " | ".join(errors)
    )


def load_video_view(
    path: str | Path,
    spec: ViewSpec,
    *,
    visual_id: str,
    backend: str = "auto",
) -> DecodedVideoView:
    """Probe, select, and decode one view using a consistent backend/metadata pair."""

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    errors: list[str] = []
    for candidate in _backend_order(backend):
        try:
            metadata = probe_video(video_path, backend=candidate)
            selection = sample_frame_indices(metadata, spec, visual_id=visual_id)
            frames, decoded_metadata = decode_selected_frames(
                video_path,
                selection.indices,
                metadata=metadata,
                backend=candidate,
            )
            if decoded_metadata != metadata:
                raise VideoDecodeError(
                    "Probe/decode metadata changed within one backend."
                )
            if selection.masked_input_positions:
                frames = list(frames)
                for position in selection.masked_input_positions:
                    source = frames[position]
                    frames[position] = Image.new(
                        "RGB", source.size, color=(128, 128, 128)
                    )
            return DecodedVideoView(frames=frames, selection=selection)
        except (EvidenceSpanError, ViewSamplingError):
            # Annotation and experimental-design errors must never be hidden by a
            # decoder fallback.
            raise
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise VideoDecodeError(
        f"Could not load sampled view from {video_path}. " + " | ".join(errors)
    )
