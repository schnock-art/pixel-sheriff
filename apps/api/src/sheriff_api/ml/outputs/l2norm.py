from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):  # type: ignore[no-untyped-def]
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)
