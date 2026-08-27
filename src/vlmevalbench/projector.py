from __future__ import annotations

import torch
from torch import nn


class MLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Projector depth must be >= 1.")
        hidden_dim = hidden_dim or output_dim
        if depth == 1:
            self.net = nn.Linear(input_dim, output_dim, bias=bias)
            return

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim, bias=bias), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def adaptive_token_pool(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if target_tokens <= 0 or tokens.shape[1] <= target_tokens:
        return tokens
    transposed = tokens.transpose(1, 2)
    pooled = torch.nn.functional.adaptive_avg_pool1d(transposed, target_tokens)
    return pooled.transpose(1, 2)
