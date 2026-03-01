from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2d(x: torch.Tensor, pool: str = "avg") -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim != 4:
        raise ValueError(f"Projection expects 2D or 4D tensor, got shape {tuple(x.shape)}")
    if pool == "max":
        pooled = F.adaptive_max_pool2d(x, output_size=(1, 1))
    else:
        pooled = F.adaptive_avg_pool2d(x, output_size=(1, 1))
    return torch.flatten(pooled, 1)


class IdentityProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PoolLinearProjection(nn.Module):
    def __init__(self, out_dim: int, pool: str = "avg") -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.pool = pool
        self.linear: nn.Linear | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_2d = _to_2d(x, pool=self.pool)
        if self.linear is None:
            in_dim = int(input_2d.shape[1])
            self.linear = nn.Linear(in_dim, self.out_dim)
            self.linear.to(device=input_2d.device, dtype=input_2d.dtype)
        return self.linear(input_2d)


class MlpProjection(nn.Module):
    def __init__(
        self,
        *,
        out_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation
        self.dropout = float(dropout)
        self.net: nn.Sequential | None = None

    def _build(self, in_dim: int, x: torch.Tensor) -> None:
        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        activation = activations.get(self.activation)
        if activation is None:
            raise ValueError(f"Unsupported mlp activation: {self.activation}")
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            activation,
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.net.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_2d = _to_2d(x, pool="avg")
        if self.net is None:
            self._build(int(input_2d.shape[1]), input_2d)
        return self.net(input_2d)


def normalize_projection_spec(spec: dict[str, Any]) -> dict[str, Any]:
    projection_type = str(spec.get("type", "")).strip().lower()
    if projection_type == "linear":
        normalized = dict(spec)
        normalized["type"] = "pool_linear"
        normalized["pool"] = "avg"
        return normalized
    return dict(spec)


def build_projection(spec: dict[str, Any]) -> nn.Module:
    normalized = normalize_projection_spec(spec)
    projection_type = str(normalized.get("type", "")).strip().lower()

    if projection_type == "none":
        return IdentityProjection()
    if projection_type == "pool_linear":
        return PoolLinearProjection(out_dim=int(normalized["out_dim"]), pool=str(normalized.get("pool", "avg")))
    if projection_type == "mlp":
        return MlpProjection(
            out_dim=int(normalized["out_dim"]),
            hidden_dim=int(normalized["hidden_dim"]),
            activation=str(normalized.get("activation", "relu")),
            dropout=float(normalized.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported projection type for v0 composer: {projection_type}")
