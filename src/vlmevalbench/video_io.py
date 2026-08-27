from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _linspace_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample frames from an empty video.")
    if count <= 1:
        return np.array([length // 2], dtype=np.int64)
    return np.linspace(0, length - 1, count).round().astype(np.int64)


def load_image_repeated(path: str | Path, count: int) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    return [image.copy() for _ in range(count)]


def load_video_frames(path: str | Path, count: int) -> list[Image.Image]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Media file does not exist: {path}")

    try:
        import decord

        reader = decord.VideoReader(str(path))
        indices = _linspace_indices(len(reader), count)
        batch = reader.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except Exception:
        pass

    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(int(i) for i in _linspace_indices(total, count))
    frames: list[Image.Image] = []
    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        frame_idx += 1
        ok, frame = cap.read()
    cap.release()

    if not frames:
        raise ValueError(f"Could not decode frames from {path}")
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames[:count]


def load_media_frames(path: str | Path, media_type: str, count: int) -> list[Image.Image]:
    if media_type == "image":
        return load_image_repeated(path, count)
    return load_video_frames(path, count)
