#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from vlmevalbench.configs import resolve_encoder
from vlmevalbench.encoders import FrozenEncoder
from vlmevalbench.utils import get_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load one configured encoder and optionally run a synthetic forward pass.")
    parser.add_argument("--encoder", required=True, help="Encoder name from configs/encoders.yaml.")
    parser.add_argument("--encoder-config", default="configs/encoders.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--forward", action="store_true", help="Run one in-memory synthetic encode pass.")
    return parser.parse_args()


def make_dummy_frames(count: int, size: int = 448) -> list[Image.Image]:
    frames: list[Image.Image] = []
    yy, xx = np.mgrid[0:size, 0:size]
    for idx in range(count):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[..., 0] = (xx + idx * 11) % 256
        arr[..., 1] = (yy + idx * 17) % 256
        arr[..., 2] = ((xx // 2 + yy // 3 + idx * 23) % 256).astype(np.uint8)
        frames.append(Image.fromarray(arr, mode="RGB"))
    return frames


def main() -> None:
    args = parse_args()
    cfg = resolve_encoder(args.encoder, args.encoder_config)

    print(f"encoder={cfg.name}")
    print(f"model_id={cfg.model_id}")
    print(f"family={cfg.family}")
    print(f"processor={cfg.processor}")
    print(f"feature_key={cfg.feature_key}")
    print(f"num_frames={cfg.num_frames}")

    encoder = FrozenEncoder(cfg, device=args.device, dtype=get_dtype(args.dtype))
    print(f"OK loaded model_type={type(encoder.model).__name__} device={encoder.device}")

    if args.forward:
        frames = make_dummy_frames(cfg.num_frames)
        features = encoder.encode_frames(frames)
        if not torch.isfinite(features).all():
            raise RuntimeError("Encoder produced non-finite features.")
        print(f"OK forward shape={tuple(features.shape)} dtype={features.dtype}")


if __name__ == "__main__":
    main()
