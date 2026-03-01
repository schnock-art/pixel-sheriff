from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sheriff_api.ml.adapters.base import FamilyAdapter
from sheriff_api.ml.outputs.l2norm import L2Norm
from sheriff_api.ml.outputs.projections import build_projection


@dataclass(frozen=True)
class AuxOutputSpec:
    name: str
    tap_name: str
    projection_spec: dict[str, Any]
    normalize: str


class OutputComposer(nn.Module):
    def __init__(
        self,
        *,
        adapter: FamilyAdapter,
        primary_output_name: str,
        output_names: list[str],
        aux_specs: list[AuxOutputSpec],
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.primary_output_name = primary_output_name
        self.output_names = list(output_names)
        self.aux_specs = list(aux_specs)
        self.projections = nn.ModuleDict()
        self.normalizers = nn.ModuleDict()

        for aux in self.aux_specs:
            self.projections[aux.name] = build_projection(aux.projection_spec)
            if aux.normalize == "l2":
                self.normalizers[aux.name] = L2Norm(dim=1)

    def forward_dict(self, x: torch.Tensor) -> dict[str, Any]:
        primary_bundle = self.adapter.forward_primary(x)
        predictions = primary_bundle.get(self.adapter.get_primary_output_name())
        if predictions is None:
            raise ValueError(f"Primary output '{self.adapter.get_primary_output_name()}' missing from adapter response")
        outputs: dict[str, Any] = {self.primary_output_name: predictions}

        for aux in self.aux_specs:
            tap_tensor = self.adapter.resolve_tap(aux.tap_name)
            projected = self.projections[aux.name](tap_tensor)
            if aux.name in self.normalizers:
                projected = self.normalizers[aux.name](projected)
            outputs[aux.name] = projected

        return outputs

    def forward(self, x: torch.Tensor) -> tuple[Any, ...]:
        outputs = self.forward_dict(x)
        missing = [name for name in self.output_names if name not in outputs]
        if missing:
            raise ValueError(f"Composer outputs missing expected names: {missing}")
        return tuple(outputs[name] for name in self.output_names)
