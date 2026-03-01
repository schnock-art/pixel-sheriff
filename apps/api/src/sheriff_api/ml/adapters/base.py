from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch
    from torch import nn


TaskType = Literal["classification", "detection", "segmentation"]


class FamilyAdapter(ABC):
    def __init__(
        self,
        *,
        task: TaskType,
        family: str,
        backbone_name: str,
        model: "nn.Module",
        supported_taps: list[str],
    ) -> None:
        self.task = task
        self.family = family
        self.backbone_name = backbone_name
        self.model = model
        self.supported_taps = supported_taps

    @abstractmethod
    def forward_primary(self, x: "torch.Tensor") -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resolve_tap(self, tap_name: str, x: "torch.Tensor | None" = None) -> "torch.Tensor":
        raise NotImplementedError

    def get_primary_output_name(self) -> str:
        return "predictions"
