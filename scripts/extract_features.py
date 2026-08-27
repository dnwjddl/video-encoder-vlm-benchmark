#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from vlmevalbench.configs import resolve_encoder
from vlmevalbench.data import read_jsonl, safe_id, write_jsonl
from vlmevalbench.encoders import FrozenEncoder
from vlmevalbench.utils import get_dtype, save_json, set_seed
from vlmevalbench.video_io import load_media_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen visual encoder features for a manifest.")
    parser.add_argument("--manifest", required=True, help="Unified JSONL manifest.")
    parser.add_argument("--encoder", required=True, help="Encoder name from configs/encoders.yaml.")
    parser.add_argument("--encoder-config", default="configs/encoders.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default=None, help="Override model_id from encoder config.")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing-media", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = resolve_encoder(
        args.encoder,
        args.encoder_config,
        overrides={
            "model_id": args.model_id,
            "num_frames": args.num_frames,
            "max_tokens": args.max_tokens,
        },
    )

    out_dir = Path(args.out_dir)
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]

    encoder = FrozenEncoder(cfg, device=args.device, dtype=get_dtype(args.dtype))
    index_records = []
    skipped = 0

    for record in tqdm(records, desc=f"extract:{args.encoder}"):
        record_id = str(record["id"])
        feature_path = feature_dir / f"{safe_id(record_id)}.pt"
        if args.skip_existing and feature_path.exists():
            loaded = torch.load(feature_path, map_location="cpu")
            shape = list(loaded["features"].shape)
            index_records.append({"id": record_id, "feature_path": str(feature_path), "shape": shape})
            continue

        media_path = record.get("media_path")
        if not media_path or not Path(media_path).exists():
            if args.allow_missing_media:
                skipped += 1
                continue
            raise FileNotFoundError(
                f"Missing media for record id={record_id}: {media_path}. "
                "Set media_path correctly or pass --allow-missing-media."
            )

        frames = load_media_frames(media_path, record.get("media_type", "video"), cfg.num_frames)
        features = encoder.encode_frames(frames)
        torch.save(
            {
                "id": record_id,
                "encoder": args.encoder,
                "model_id": cfg.model_id,
                "features": features,
            },
            feature_path,
        )
        index_records.append(
            {
                "id": record_id,
                "feature_path": str(feature_path),
                "shape": list(features.shape),
            }
        )

    write_jsonl(out_dir / "index.jsonl", index_records)
    save_json(
        out_dir / "metadata.json",
        {
            "encoder": args.encoder,
            "encoder_config": encoder.metadata(),
            "manifest": str(args.manifest),
            "num_records": len(index_records),
            "skipped": skipped,
        },
    )
    print(f"Wrote {len(index_records)} feature files to {feature_dir}")
    if skipped:
        print(f"Skipped {skipped} records with missing media.")


if __name__ == "__main__":
    main()
