from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import asdict
import importlib.machinery
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
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


def _call_vision_tower(model: torch.nn.Module, batch: dict[str, Any]) -> Any:
    if hasattr(model, "vision_model"):
        vision_model = getattr(model, "vision_model")
        try:
            return vision_model(**batch)
        except TypeError:
            if "pixel_values" in batch:
                return vision_model(pixel_values=batch["pixel_values"])
            raise
    return model(**batch)


def _flash_attn_qkvpacked_fallback(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: Any,
) -> torch.Tensor:
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise RuntimeError(f"Unsupported qkvpacked shape for flash-attn fallback: {tuple(qkv.shape)}")
    batch, seqlen = qkv.shape[:2]
    flattened = qkv.reshape(batch * seqlen, *qkv.shape[2:])
    cu_seqlens = torch.arange(
        0,
        (batch + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=qkv.device,
    )
    return _flash_attn_varlen_qkvpacked_fallback(
        flattened,
        cu_seqlens,
        seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    ).reshape(batch, seqlen, *qkv.shape[3:])


def _flash_attn_varlen_qkvpacked_fallback(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_s: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: Any,
) -> torch.Tensor:
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise RuntimeError(f"Unsupported varlen qkvpacked shape for flash-attn fallback: {tuple(qkv.shape)}")
    if dropout_p:
        raise RuntimeError("flash-attn fallback does not support attention dropout.")

    outputs = []
    starts = cu_seqlens[:-1].tolist()
    ends = cu_seqlens[1:].tolist()
    for start, end in zip(starts, ends):
        segment = qkv[int(start) : int(end)]
        if segment.numel() == 0:
            continue
        q, k, v = segment.unbind(dim=1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scale = softmax_scale or q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.ones(attn.shape[-2:], device=attn.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
        context = torch.matmul(attn.softmax(dim=-1), v)
        outputs.append(context.transpose(0, 1))
    if not outputs:
        return qkv.new_zeros((0, qkv.shape[2], qkv.shape[3]))
    return torch.cat(outputs, dim=0)


def _flash_attn_func_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: Any,
) -> torch.Tensor:
    if dropout_p:
        raise RuntimeError("flash-attn fallback does not support attention dropout.")
    scale = softmax_scale or q.shape[-1] ** -0.5
    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2)
    v_h = v.transpose(1, 2)
    attn = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
    if causal:
        mask = torch.ones(attn.shape[-2:], device=attn.device, dtype=torch.bool).triu(1)
        attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
    return torch.matmul(attn.softmax(dim=-1), v_h).transpose(1, 2)


def _unpad_input_fallback(x: torch.Tensor, key_padding_mask: torch.Tensor):
    batch, seqlen = key_padding_mask.shape
    indices = torch.nonzero(key_padding_mask.reshape(-1), as_tuple=False).flatten()
    x_unpad = x.reshape(batch * seqlen, *x.shape[2:]).index_select(0, indices)
    lengths = key_padding_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=x.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_s = int(lengths.max().item()) if lengths.numel() else 0
    return x_unpad, indices, cu_seqlens, max_s


def _pad_input_fallback(x_unpad: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = x_unpad.new_zeros((batch * seqlen, *x_unpad.shape[1:]))
    output.index_copy_(0, indices, x_unpad)
    return output.reshape(batch, seqlen, *x_unpad.shape[1:])


class _FallbackFusedMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features, bias=bias),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_features, out_features, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FallbackDropoutAddRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.prenorm = prenorm
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, **_: Any):
        residual_out = x if residual is None else residual + x
        if self.residual_in_fp32:
            residual_out = residual_out.float()
        normed = residual_out * torch.rsqrt(residual_out.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = normed.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)
        if self.prenorm:
            return normed, residual_out
        return normed


@contextmanager
def _temporary_flash_attn_stub():
    names = [
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn.bert_padding",
        "flash_attn.modules",
        "flash_attn.modules.mlp",
        "flash_attn.ops",
        "flash_attn.ops.rms_norm",
    ]
    previous = {name: sys.modules.get(name) for name in names}

    package = types.ModuleType("flash_attn")
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)

    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    interface.flash_attn_unpadded_qkvpacked_func = _flash_attn_varlen_qkvpacked_fallback
    interface.flash_attn_varlen_qkvpacked_func = _flash_attn_varlen_qkvpacked_fallback
    interface.flash_attn_qkvpacked_func = _flash_attn_qkvpacked_fallback
    interface.flash_attn_func = _flash_attn_func_fallback

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.pad_input = _pad_input_fallback
    bert_padding.unpad_input = _unpad_input_fallback

    modules = types.ModuleType("flash_attn.modules")
    modules.__path__ = []
    modules.__spec__ = importlib.machinery.ModuleSpec("flash_attn.modules", loader=None, is_package=True)

    mlp = types.ModuleType("flash_attn.modules.mlp")
    mlp.__spec__ = importlib.machinery.ModuleSpec("flash_attn.modules.mlp", loader=None)
    mlp.Mlp = _FallbackFusedMLP
    mlp.FusedMLP = _FallbackFusedMLP

    ops = types.ModuleType("flash_attn.ops")
    ops.__path__ = []
    ops.__spec__ = importlib.machinery.ModuleSpec("flash_attn.ops", loader=None, is_package=True)

    rms_norm = types.ModuleType("flash_attn.ops.rms_norm")
    rms_norm.__spec__ = importlib.machinery.ModuleSpec("flash_attn.ops.rms_norm", loader=None)
    rms_norm.DropoutAddRMSNorm = _FallbackDropoutAddRMSNorm

    package.flash_attn_interface = interface
    package.bert_padding = bert_padding
    package.modules = modules
    package.ops = ops
    modules.mlp = mlp
    ops.rms_norm = rms_norm
    sys.modules.update(
        {
            "flash_attn": package,
            "flash_attn.flash_attn_interface": interface,
            "flash_attn.bert_padding": bert_padding,
            "flash_attn.modules": modules,
            "flash_attn.modules.mlp": mlp,
            "flash_attn.ops": ops,
            "flash_attn.ops.rms_norm": rms_norm,
        }
    )

    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _disable_flash_attn_in_config(config: Any) -> None:
    disabled_keys = {
        "use_flash_attn",
        "use_flash_attention",
        "use_flash_sdp",
        "use_mem_efficient_sdp",
        "use_fused_mlp",
        "use_fused_rmsnorm",
    }
    seen: set[int] = set()

    def visit(value: Any) -> None:
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, Mapping):
            for key in disabled_keys:
                if key in value:
                    value[key] = False
            for child in value.values():
                visit(child)
            return

        for key in disabled_keys:
            if hasattr(value, key):
                try:
                    setattr(value, key, False)
                except Exception:
                    pass

        if hasattr(value, "__dict__"):
            for child in vars(value).values():
                visit(child)

    visit(config)


def _needs_standard_loading(exc: AttributeError) -> bool:
    message = str(exc)
    return "all_tied_weights_keys" in message or "_tied_weights_keys" in message


def _patch_transformers_tied_weights_compat() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def all_tied_weights_keys(self) -> dict[str, str]:
        keys: dict[str, str] = {}
        for source in [type(self), self]:
            for attr_name in ("_tied_weights_keys", "_dynamic_tied_weights_keys"):
                raw_keys = getattr(source, attr_name, None)
                if raw_keys is None:
                    continue
                if isinstance(raw_keys, Mapping):
                    keys.update({str(key): str(value) for key, value in raw_keys.items()})
                elif isinstance(raw_keys, (list, tuple, set)):
                    keys.update({str(key): str(key) for key in raw_keys})
        return keys

    PreTrainedModel.all_tied_weights_keys = property(all_tied_weights_keys)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_pretrained_source(model_id: str) -> str:
    if not _env_flag("VLMEB_LOCAL_FILES_ONLY"):
        return model_id
    if Path(model_id).exists():
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, repo_type="model", local_files_only=True)


def _frames_to_tchw_uint8(frames: list[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
    video = torch.from_numpy(np.stack(arrays, axis=0))
    return video.permute(0, 3, 1, 2).contiguous()


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
        self.pretrained_source = _resolve_pretrained_source(cfg.model_id)

        self.processor = self._load_processor(cfg)
        self.model = self._load_model(cfg)
        self.model.eval()
        self.model.requires_grad_(False)

    def _load_processor(self, cfg: EncoderConfig):
        kwargs = {
            "trust_remote_code": cfg.trust_remote_code,
            "local_files_only": _env_flag("VLMEB_LOCAL_FILES_ONLY"),
        }
        if cfg.processor == "model_transform":
            return None
        if cfg.processor == "clip_image":
            return CLIPImageProcessor.from_pretrained(self.pretrained_source, **kwargs)
        if cfg.processor == "videomae":
            return VideoMAEImageProcessor.from_pretrained(self.pretrained_source, **kwargs)
        if cfg.processor == "video":
            auto_video_processor = _maybe_get_video_processor()
            if auto_video_processor is not None:
                try:
                    return auto_video_processor.from_pretrained(self.pretrained_source, **kwargs)
                except Exception:
                    pass
        try:
            return AutoProcessor.from_pretrained(self.pretrained_source, **kwargs)
        except Exception:
            return AutoImageProcessor.from_pretrained(self.pretrained_source, **kwargs)

    def _load_model(self, cfg: EncoderConfig):
        kwargs = {
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": cfg.low_cpu_mem_usage,
            "local_files_only": _env_flag("VLMEB_LOCAL_FILES_ONLY"),
        }
        if cfg.disable_flash_attn:
            config = AutoConfig.from_pretrained(
                self.pretrained_source,
                trust_remote_code=cfg.trust_remote_code,
                local_files_only=_env_flag("VLMEB_LOCAL_FILES_ONLY"),
            )
            _disable_flash_attn_in_config(config)
            with _temporary_flash_attn_stub():
                model = self._auto_model_from_pretrained(cfg, kwargs, config=config)
            return model.to(self.device)

        model = self._auto_model_from_pretrained(cfg, kwargs)
        return model.to(self.device)

    def _auto_model_from_pretrained(
        self,
        cfg: EncoderConfig,
        kwargs: dict[str, Any],
        *,
        config: Any | None = None,
    ) -> torch.nn.Module:
        call_kwargs = dict(kwargs)
        if config is not None:
            call_kwargs["config"] = config
        _patch_transformers_tied_weights_compat()
        try:
            return AutoModel.from_pretrained(self.pretrained_source, **call_kwargs)
        except AttributeError as exc:
            if call_kwargs.get("low_cpu_mem_usage") and _needs_standard_loading(exc):
                print(
                    f"Warning: {cfg.name} is incompatible with low_cpu_mem_usage=True; "
                    "retrying with low_cpu_mem_usage=False."
                )
                call_kwargs["low_cpu_mem_usage"] = False
                return AutoModel.from_pretrained(self.pretrained_source, **call_kwargs)
            raise
        except TypeError:
            retry_kwargs = dict(call_kwargs)
            retry_kwargs.pop("torch_dtype", None)
            if "config" not in retry_kwargs:
                retry_kwargs["config"] = AutoConfig.from_pretrained(
                    self.pretrained_source,
                    trust_remote_code=cfg.trust_remote_code,
                    local_files_only=_env_flag("VLMEB_LOCAL_FILES_ONLY"),
                )
            model = AutoModel.from_pretrained(self.pretrained_source, **retry_kwargs)
            return model.to(dtype=self.dtype)

    @torch.no_grad()
    def encode_frames(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.cfg.family == "image":
            return self._encode_as_images(frames)
        return self._encode_as_video(frames)

    def _encode_as_images(self, frames: list[Image.Image]) -> torch.Tensor:
        batch = self.processor(images=frames, return_tensors="pt")
        batch = _move_to_device(batch, self.device, self.dtype)
        outputs = _call_vision_tower(self.model, batch)
        tokens = _extract_tokens(outputs, self.cfg.feature_key)
        tokens = _ensure_token_tensor(tokens)
        if self.cfg.drop_cls and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        tokens = tokens.flatten(0, 1).unsqueeze(0)
        tokens = adaptive_token_pool(tokens, self.cfg.max_tokens)
        return tokens.squeeze(0).float().cpu()

    def _encode_as_video(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.processor is None and hasattr(self.model, "transform") and hasattr(self.model, "encode_vision"):
            return self._encode_with_model_transform(frames)

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

    def _encode_with_model_transform(self, frames: list[Image.Image]) -> torch.Tensor:
        video = _frames_to_tchw_uint8(frames)
        transformed = self.model.transform(video).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        outputs = self.model.encode_vision(transformed, test=True)
        tokens = _ensure_token_tensor(outputs)
        tokens = adaptive_token_pool(tokens, self.cfg.max_tokens)
        return tokens.squeeze(0).float().cpu()

    def metadata(self) -> dict[str, Any]:
        return asdict(self.cfg)
