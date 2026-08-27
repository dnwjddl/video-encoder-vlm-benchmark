#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from vlmevalbench.configs import resolve_encoder
from vlmevalbench.data import normalize_choice_answer, read_jsonl, write_jsonl
from vlmevalbench.utils import get_dtype, set_seed
from vlmevalbench.video_io import load_media_frames


PERTURBATIONS = ("original", "reverse", "shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot MCQ perturbation evaluation for text-aligned visual encoders. "
            "Compares original, reversed, and shuffled videos without projector training."
        )
    )
    parser.add_argument("--manifest", required=True, help="MCQ JSONL manifest with choices and answer.")
    parser.add_argument("--encoder", required=True, help="Encoder name from configs/encoders.yaml.")
    parser.add_argument("--encoder-config", default="configs/encoders.yaml")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--choice-template",
        default="Question: {question}\nAnswer: {choice}",
        help="Text template used to embed each option.",
    )
    parser.add_argument("--allow-missing-media", action="store_true")
    parser.add_argument(
        "--strict-media",
        action="store_true",
        help="Raise on unreadable/corrupt media instead of skipping it.",
    )
    return parser.parse_args()


def move_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def pooled_output(output: Any) -> torch.Tensor:
    for key in ("image_embeds", "video_embeds", "pooler_output", "text_embeds", "last_hidden_state"):
        if hasattr(output, key):
            value = getattr(output, key)
            if torch.is_tensor(value):
                return value
        if isinstance(output, dict) and key in output and torch.is_tensor(output[key]):
            return output[key]
    if isinstance(output, tuple):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise RuntimeError("Could not find an embedding tensor in model output.")


class TextAlignedEncoder:
    def __init__(self, model_id: str, trust_remote_code: bool, device: str, dtype: torch.dtype) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        if not hasattr(self.model, "get_text_features"):
            raise RuntimeError(
                f"{model_id} does not expose get_text_features. "
                "Zero-shot MCQ perturbation evaluation requires a text-aligned encoder."
            )

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        batch = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        batch = move_to_device(batch, self.device, self.dtype)
        features = self.model.get_text_features(**batch)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_images(self, frames: list) -> torch.Tensor:
        batch = self.processor(images=frames, return_tensors="pt")
        batch = move_to_device(batch, self.device, self.dtype)
        if hasattr(self.model, "get_image_features"):
            features = self.model.get_image_features(**batch)
        else:
            features = pooled_output(self.model(**batch))
        if features.ndim > 2:
            features = features.mean(dim=1)
        features = F.normalize(features.float(), dim=-1)
        return F.normalize(features.mean(dim=0), dim=-1)

    @torch.no_grad()
    def encode_video(self, frames: list) -> torch.Tensor:
        if hasattr(self.model, "get_video_features"):
            try:
                batch = self.processor(videos=[frames], return_tensors="pt")
            except Exception:
                batch = self.processor(frames, return_tensors="pt")
            batch = move_to_device(batch, self.device, self.dtype)
            features = self.model.get_video_features(**batch)
            if features.ndim > 2:
                features = features.mean(dim=1)
            return F.normalize(features.float().squeeze(0), dim=-1)
        return self.encode_images(frames)


def perturb_frames(frames: list, mode: str, rng: random.Random) -> list:
    if mode == "original":
        return list(frames)
    if mode == "reverse":
        return list(reversed(frames))
    if mode == "shuffle":
        shuffled = list(frames)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown perturbation: {mode}")


def predict(
    *,
    encoder: TextAlignedEncoder,
    frames: list,
    question: str,
    choices: list[str],
    choice_template: str,
    media_type: str,
) -> tuple[str, dict[str, float]]:
    texts = [choice_template.format(question=question, choice=choice) for choice in choices]
    text_features = encoder.encode_texts(texts)
    if media_type == "video":
        visual_feature = encoder.encode_video(frames)
    else:
        visual_feature = encoder.encode_images(frames)
    scores_tensor = visual_feature @ text_features.T
    scores = {
        chr(ord("A") + idx): float(scores_tensor[idx].detach().cpu().item())
        for idx in range(len(choices))
    }
    pred = max(scores, key=scores.get)
    return pred, scores


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(str(row.get("benchmark", "unknown")), []).append(row)

    summary = []
    for group, items in sorted(groups.items()):
        n = len(items)
        original_correct = sum(bool(r["correct_original"]) for r in items)
        reverse_correct = sum(bool(r["correct_reverse"]) for r in items)
        shuffle_correct = sum(bool(r["correct_shuffle"]) for r in items)
        robust_correct = sum(
            bool(r["correct_original"] and r["correct_reverse"] and r["correct_shuffle"])
            for r in items
        )
        temporal_reverse = sum(bool(r["correct_original"] and not r["correct_reverse"]) for r in items)
        temporal_shuffle = sum(bool(r["correct_original"] and not r["correct_shuffle"]) for r in items)
        temporal_any = sum(
            bool(r["correct_original"] and (not r["correct_reverse"] or not r["correct_shuffle"]))
            for r in items
        )
        perturbation_helped = sum(
            bool((not r["correct_original"]) and (r["correct_reverse"] or r["correct_shuffle"]))
            for r in items
        )
        pred_changed_reverse = sum(r["prediction_original"] != r["prediction_reverse"] for r in items)
        pred_changed_shuffle = sum(r["prediction_original"] != r["prediction_shuffle"] for r in items)

        denom_orig = max(original_correct, 1)
        summary.append(
            {
                "group": group,
                "num_examples": n,
                "original_correct": original_correct,
                "reverse_correct": reverse_correct,
                "shuffle_correct": shuffle_correct,
                "robust_correct_all": robust_correct,
                "temporal_sensitive_reverse": temporal_reverse,
                "temporal_sensitive_shuffle": temporal_shuffle,
                "temporal_sensitive_any": temporal_any,
                "perturbation_helped": perturbation_helped,
                "prediction_changed_reverse": pred_changed_reverse,
                "prediction_changed_shuffle": pred_changed_shuffle,
                "original_accuracy": original_correct / max(n, 1),
                "reverse_accuracy": reverse_correct / max(n, 1),
                "shuffle_accuracy": shuffle_correct / max(n, 1),
                "robust_rate_among_original_correct": robust_correct / denom_orig,
                "temporal_sensitive_rate_among_original_correct": temporal_any / denom_orig,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = resolve_encoder(
        args.encoder,
        args.encoder_config,
        overrides={"model_id": args.model_id, "num_frames": args.num_frames},
    )
    records = [r for r in read_jsonl(args.manifest) if r.get("choices")]
    if args.limit:
        records = records[: args.limit]

    encoder = TextAlignedEncoder(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
        device=args.device,
        dtype=get_dtype(args.dtype),
    )
    rng = random.Random(args.seed)
    rows = []
    skipped = 0
    skipped_bad_media = 0

    for record in tqdm(records, desc=f"zeroshot_perturb:{args.encoder}"):
        media_path = record.get("media_path")
        if not media_path or not Path(media_path).exists():
            if args.allow_missing_media:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing media for id={record.get('id')}: {media_path}")

        choices = [str(choice) for choice in record["choices"]]
        gold = normalize_choice_answer(record.get("answer"), choices)
        try:
            frames = load_media_frames(media_path, record.get("media_type", "video"), cfg.num_frames)
        except Exception as exc:
            if args.strict_media:
                raise
            skipped_bad_media += 1
            print(
                f"Warning: skipping unreadable media id={record.get('id')} "
                f"path={media_path}: {type(exc).__name__}: {exc}"
            )
            continue

        preds = {}
        scores = {}
        for mode in PERTURBATIONS:
            mode_frames = perturb_frames(frames, mode, rng)
            pred, mode_scores = predict(
                encoder=encoder,
                frames=mode_frames,
                question=str(record.get("question", "")),
                choices=choices,
                choice_template=args.choice_template,
                media_type=record.get("media_type", "video"),
            )
            preds[mode] = pred
            scores[mode] = mode_scores

        row = {
            "id": record["id"],
            "source": record.get("source", "unknown"),
            "benchmark": record.get("benchmark", record.get("source", "unknown")),
            "encoder": args.encoder,
            "answer": gold,
            "prediction_original": preds["original"],
            "prediction_reverse": preds["reverse"],
            "prediction_shuffle": preds["shuffle"],
            "correct_original": preds["original"] == gold,
            "correct_reverse": preds["reverse"] == gold,
            "correct_shuffle": preds["shuffle"] == gold,
            "temporal_sensitive_reverse": preds["original"] == gold and preds["reverse"] != gold,
            "temporal_sensitive_shuffle": preds["original"] == gold and preds["shuffle"] != gold,
            "temporal_sensitive_any": preds["original"] == gold
            and (preds["reverse"] != gold or preds["shuffle"] != gold),
            "scores": scores,
        }
        rows.append(row)

    write_jsonl(args.out_jsonl, rows)
    summary = summarize(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["group"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote per-example perturbation predictions to {args.out_jsonl}")
    print(f"Wrote perturbation summary to {args.out_csv}")
    if skipped:
        print(f"Skipped {skipped} rows with missing media.")
    if skipped_bad_media:
        print(f"Skipped {skipped_bad_media} rows with unreadable/corrupt media.")


if __name__ == "__main__":
    main()
