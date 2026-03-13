from __future__ import annotations

from typing import Any

from sheriff_api.ml.adapters.base import FamilyAdapter


class TorchvisionRetinaNetAdapter(FamilyAdapter):
    def __init__(self, *, backbone_name: str, num_classes: int, pretrained: bool) -> None:
        import torchvision.models.detection as tv_detection
        import torchvision.models as tv_models

        if backbone_name != "resnet50":
            raise ValueError("retinanet adapter currently supports only backbone 'resnet50'")

        try:
            model = tv_detection.retinanet_resnet50_fpn(
                weights=None,
                weights_backbone=tv_models.ResNet50_Weights.DEFAULT if pretrained else None,
                num_classes=int(num_classes),
            )
        except Exception as exc:
            if pretrained:
                raise ValueError(f"pretrained_weights_unavailable:retinanet/{backbone_name}:{exc}") from exc
            raise
        super().__init__(
            task="detection",
            family="retinanet",
            backbone_name=backbone_name,
            model=model,
            supported_taps=[],
        )

    def forward_primary(self, x):  # type: ignore[no-untyped-def]
        images = [img for img in x]
        predictions = self.model(images)
        return {self.get_primary_output_name(): predictions}

    def resolve_tap(self, tap_name: str, x=None):  # type: ignore[no-untyped-def]
        raise ValueError(f"Tap '{tap_name}' is not supported by adapter '{self.family}'")


def build_retinanet_adapter(model_config: dict[str, Any]) -> FamilyAdapter:
    architecture = model_config.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("Model config architecture is missing")
    backbone = architecture.get("backbone", {})
    if not isinstance(backbone, dict):
        raise ValueError("Model config architecture.backbone is missing")
    head = architecture.get("head", {})
    if not isinstance(head, dict):
        raise ValueError("Model config architecture.head is missing")

    return TorchvisionRetinaNetAdapter(
        backbone_name=str(backbone.get("name", "")).strip().lower(),
        num_classes=int(head.get("num_classes", 0)),
        pretrained=bool(backbone.get("pretrained")),
    )
