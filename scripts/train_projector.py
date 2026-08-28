#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlmevalbench.projector import MLPProjector
from vlmevalbench.training import (
    FeatureTextDataset,
    build_inputs_embeds_and_labels,
    collate_feature_text,
)
from vlmevalbench.utils import get_dtype, save_json, set_seed


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train projector only with frozen visual features and frozen LLM.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--llm-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--projector-depth", type=int, default=2)
    parser.add_argument("--projector-hidden-dim", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    return parser.parse_args()


def infer_feature_dim(feature_index: str) -> int:
    import json

    with Path(feature_index).open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    shape = first["shape"]
    return int(shape[-1])


def save_checkpoint(out_dir: Path, projector: torch.nn.Module, step: int, metadata: dict) -> None:
    ckpt_dir = out_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = projector.module.state_dict() if hasattr(projector, "module") else projector.state_dict()
    torch.save(state, ckpt_dir / "projector.pt")
    save_json(ckpt_dir / "metadata.json", metadata | {"step": step})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    dtype = get_dtype(args.dtype)

    local_files_only = env_flag("VLMEB_LOCAL_FILES_ONLY")
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_id,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    llm.eval()
    llm.requires_grad_(False)

    input_dim = infer_feature_dim(args.feature_index)
    output_dim = int(llm.config.hidden_size)
    projector = MLPProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.projector_hidden_dim,
        depth=args.projector_depth,
    )

    dataset = FeatureTextDataset(args.manifest, args.feature_index)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_feature_text,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    projector, llm, optimizer, dataloader = accelerator.prepare(projector, llm, optimizer, dataloader)

    metadata = {
        "encoder_name": args.encoder_name,
        "llm_id": args.llm_id,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "projector_depth": args.projector_depth,
        "projector_hidden_dim": args.projector_hidden_dim,
        "manifest": args.manifest,
        "feature_index": args.feature_index,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
    }
    save_json(out_dir / "metadata.json", metadata)

    global_step = 0
    projector.train()
    for epoch in range(args.epochs):
        progress = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch + 1}")
        for batch in progress:
            with accelerator.accumulate(projector):
                features = batch["features"].to(accelerator.device, dtype=torch.float32)
                feature_mask = batch["feature_mask"].to(accelerator.device)
                visual_embeds = projector(features)
                inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
                    tokenizer=tokenizer,
                    llm=llm,
                    visual_embeds=visual_embeds,
                    feature_mask=feature_mask,
                    prefixes=batch["prefixes"],
                    suffixes=batch["suffixes"],
                    answers=batch["answers"],
                    max_length=args.max_length,
                )
                outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")
            if (
                accelerator.is_main_process
                and args.save_every_steps > 0
                and global_step % args.save_every_steps == 0
            ):
                save_checkpoint(out_dir, accelerator.unwrap_model(projector), global_step, metadata)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(out_dir, accelerator.unwrap_model(projector), global_step, metadata)
        print(f"Saved final projector checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
