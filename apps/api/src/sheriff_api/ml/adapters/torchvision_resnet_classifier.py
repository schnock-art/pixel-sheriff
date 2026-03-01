from __future__ import annotations

from typing import Any

from sheriff_api.ml.adapters.base import FamilyAdapter
from sheriff_api.ml.metadata.backbones import get_backbone_meta, list_supported_taps, normalize_tap_name
from sheriff_api.ml.taps.manager import TapManager


def _resolve_weights(_backbone_name: str, _pretrained: bool):  # type: ignore[no-untyped-def]
    # v0 keeps model construction deterministic and avoids implicit network downloads.
    return None


class TorchvisionResnetClassifierAdapter(FamilyAdapter):
    def __init__(self, *, backbone_name: str, num_classes: int, pretrained: bool) -> None:
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models

        metadata = get_backbone_meta(backbone_name)
        if metadata.family != "resnet":
            raise ValueError(f"Backbone '{backbone_name}' is not compatible with resnet_classifier")

        ctor = getattr(tv_models, backbone_name, None)
        if ctor is None:
            raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")

        model = ctor(weights=_resolve_weights(backbone_name, pretrained))
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, int(num_classes))

        super().__init__(
            task="classification",
            family="resnet_classifier",
            backbone_name=backbone_name,
            model=model,
            supported_taps=list_supported_taps(backbone_name),
        )

        self._torch = torch
        self._tap_manager = TapManager(model)
        self._tap_manager.register_hook("backbone.c3", "layer2")
        self._tap_manager.register_hook("backbone.c4", "layer3")
        self._tap_manager.register_hook("backbone.c5", "layer4")
        self._tap_manager.register_extractor("backbone.global_pool", self._extract_global_pool)

    def _extract_global_pool(self, _x=None):  # type: ignore[no-untyped-def]
        c5 = self._tap_manager.require("backbone.c5")
        pooled = self.model.avgpool(c5)
        return self._torch.flatten(pooled, 1)

    def forward_primary(self, x):  # type: ignore[no-untyped-def]
        self._tap_manager.clear()
        logits = self.model(x)
        return {self.get_primary_output_name(): logits}

    def resolve_tap(self, tap_name: str, x=None):  # type: ignore[no-untyped-def]
        canonical = normalize_tap_name(tap_name)
        tensor = self._tap_manager.get(canonical)
        if tensor is None and x is not None:
            self.forward_primary(x)
            tensor = self._tap_manager.get(canonical)
        if tensor is None:
            raise ValueError(f"Tap '{tap_name}' could not be resolved by adapter '{self.family}'")
        return tensor


def build_resnet_classifier_adapter(model_config: dict[str, Any]) -> FamilyAdapter:
    architecture = model_config.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("Model config architecture is missing")

    backbone = architecture.get("backbone", {})
    if not isinstance(backbone, dict):
        raise ValueError("Model config architecture.backbone is missing")

    head = architecture.get("head", {})
    if not isinstance(head, dict):
        raise ValueError("Model config architecture.head is missing")

    backbone_name = str(backbone.get("name", "")).strip().lower()
    pretrained = bool(backbone.get("pretrained"))
    num_classes = int(head.get("num_classes", 0))
    if not backbone_name:
        raise ValueError("Backbone name is required")
    if num_classes < 1:
        raise ValueError("head.num_classes must be >= 1")

    return TorchvisionResnetClassifierAdapter(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
