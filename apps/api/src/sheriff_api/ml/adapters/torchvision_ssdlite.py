from __future__ import annotations

from typing import Any

from sheriff_api.ml.adapters.base import FamilyAdapter


class TorchvisionSSDLiteAdapter(FamilyAdapter):
    def __init__(self, *, backbone_name: str, num_classes: int, pretrained: bool) -> None:
        import torchvision.models.detection as tv_detection
        import torchvision.models as tv_models

        if backbone_name != "mobilenet_v3_large":
            raise ValueError("ssdlite320_mobilenet_v3_large adapter currently supports only backbone 'mobilenet_v3_large'")

        try:
            model = tv_detection.ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
                num_classes=int(num_classes) + 1,
            )
        except Exception as exc:
            if pretrained:
                raise ValueError(f"pretrained_weights_unavailable:ssdlite320_mobilenet_v3_large/{backbone_name}:{exc}") from exc
            raise
        super().__init__(
            task="detection",
            family="ssdlite320_mobilenet_v3_large",
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


def build_ssdlite_adapter(model_config: dict[str, Any]) -> FamilyAdapter:
    architecture = model_config.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("Model config architecture is missing")
    backbone = architecture.get("backbone", {})
    if not isinstance(backbone, dict):
        raise ValueError("Model config architecture.backbone is missing")
    head = architecture.get("head", {})
    if not isinstance(head, dict):
        raise ValueError("Model config architecture.head is missing")

    return TorchvisionSSDLiteAdapter(
        backbone_name=str(backbone.get("name", "")).strip().lower(),
        num_classes=int(head.get("num_classes", 0)),
        pretrained=bool(backbone.get("pretrained")),
    )
