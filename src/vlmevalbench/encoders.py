from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPImageProcessor,
    VideoMAEImageProcessor,
)

from vlmevalbench.configs import EncoderConfig
from vlmevalbench.projector import adaptive_token_pool


def _maybe_get_video_processor():
    try:
        from transformers import AutoVideoProcessor

        return AutoVideoProcessor
    except Exception:
        return None


def _move_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def _extract_tokens(outputs: Any, feature_key: str) -> torch.Tensor:
    if feature_key != "auto":
        if hasattr(outputs, feature_key):
            value = getattr(outputs, feature_key)
            if torch.is_tensor(value):
                return value
        if isinstance(outputs, dict) and feature_key in outputs:
            return outputs[feature_key]

    candidates = [
        "last_hidden_state",
        "image_embeds",
        "video_embeds",
        "pooler_output",
        "logits",
    ]
    for key in candidates:
        if hasattr(outputs, key):
            value = getattr(outputs, key)
            if torch.is_tensor(value):
                return value
        if isinstance(outputs, dict) and key in outputs and torch.is_tensor(outputs[key]):
            return outputs[key]

    if isinstance(outputs, tuple):
        for value in outputs:
            if torch.is_tensor(value):
                return value
            if isinstance(value, (list, tuple)):
                for inner in value:
                    if torch.is_tensor(inner):
                        return inner

    raise RuntimeError(
        "Could not locate tensor features in encoder output. "
        f"Output type={type(outputs)}, feature_key={feature_key}"
    )


def _ensure_token_tensor(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        return features.unsqueeze(1)
    if features.ndim == 3:
        return features
    if features.ndim > 3:
        return features.flatten(1, -2)
    raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")


class FrozenEncoder:
    def __init__(
        self,
        cfg: EncoderConfig,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype

        self.processor = self._load_processor(cfg)
        self.model = self._load_model(cfg)
        self.model.eval()
        self.model.requires_grad_(False)

    def _load_processor(self, cfg: EncoderConfig):
        kwargs = {"trust_remote_code": cfg.trust_remote_code}
        if cfg.processor == "clip_image":
            return CLIPImageProcessor.from_pretrained(cfg.model_id, **kwargs)
        if cfg.processor == "videomae":
            return VideoMAEImageProcessor.from_pretrained(cfg.model_id, **kwargs)
        if cfg.processor == "video":
            auto_video_processor = _maybe_get_video_processor()
            if auto_video_processor is not None:
                try:
                    return auto_video_processor.from_pretrained(cfg.model_id, **kwargs)
                except Exception:
                    pass
        try:
            return AutoProcessor.from_pretrained(cfg.model_id, **kwargs)
        except Exception:
            return AutoImageProcessor.from_pretrained(cfg.model_id, **kwargs)

    def _load_model(self, cfg: EncoderConfig):
        kwargs = {
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            model = AutoModel.from_pretrained(cfg.model_id, **kwargs)
        except TypeError:
            config = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
            model = AutoModel.from_pretrained(cfg.model_id, config=config, trust_remote_code=cfg.trust_remote_code)
            model = model.to(dtype=self.dtype)
        return model.to(self.device)

    @torch.no_grad()
    def encode_frames(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.cfg.family == "image":
            return self._encode_as_images(frames)
        return self._encode_as_video(frames)

    def _encode_as_images(self, frames: list[Image.Image]) -> torch.Tensor:
        batch = self.processor(images=frames, return_tensors="pt")
        batch = _move_to_device(batch, self.device, self.dtype)
        outputs = self.model(**batch)
        tokens = _extract_tokens(outputs, self.cfg.feature_key)
        tokens = _ensure_token_tensor(tokens)
        if self.cfg.drop_cls and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        tokens = tokens.flatten(0, 1).unsqueeze(0)
        tokens = adaptive_token_pool(tokens, self.cfg.max_tokens)
        return tokens.squeeze(0).float().cpu()

    def _encode_as_video(self, frames: list[Image.Image]) -> torch.Tensor:
        try:
            batch = self.processor(videos=[frames], return_tensors="pt")
        except Exception:
            batch = self.processor(frames, return_tensors="pt")

        if "pixel_values" in batch and self.cfg.input_layout == "bcthw":
            # VideoMAEImageProcessor usually returns B,T,C,H,W; VideoMAEv2 remote code expects B,C,T,H,W.
            values = batch["pixel_values"]
            if values.ndim == 5 and values.shape[2] in {1, 3}:
                pass
            elif values.ndim == 5:
                batch["pixel_values"] = values.permute(0, 2, 1, 3, 4).contiguous()

        batch = _move_to_device(batch, self.device, self.dtype)
        outputs = self.model(**batch)
        tokens = _extract_tokens(outputs, self.cfg.feature_key)
        tokens = _ensure_token_tensor(tokens)
        tokens = adaptive_token_pool(tokens, self.cfg.max_tokens)
        return tokens.squeeze(0).float().cpu()

    def metadata(self) -> dict[str, Any]:
        return asdict(self.cfg)
