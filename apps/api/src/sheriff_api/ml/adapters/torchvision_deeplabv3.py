from __future__ import annotations

from typing import Any

from sheriff_api.ml.adapters.base import FamilyAdapter


class TorchvisionDeepLabV3Adapter(FamilyAdapter):
    def __init__(self, *, backbone_name: str, num_classes: int, _pretrained: bool) -> None:
        import torchvision.models.segmentation as tv_segmentation

        builders = {
            "resnet50": tv_segmentation.deeplabv3_resnet50,
            "resnet101": tv_segmentation.deeplabv3_resnet101,
        }
        builder = builders.get(backbone_name)
        if builder is None:
            raise ValueError("deeplabv3 adapter currently supports backbones: resnet50, resnet101")

        model = builder(weights=None, weights_backbone=None, num_classes=int(num_classes))
        super().__init__(
            task="segmentation",
            family="deeplabv3",
            backbone_name=backbone_name,
            model=model,
            supported_taps=[],
        )

    def forward_primary(self, x):  # type: ignore[no-untyped-def]
        outputs = self.model(x)
        if not isinstance(outputs, dict) or "out" not in outputs:
            raise ValueError("deeplabv3 forward output is missing 'out'")
        return {self.get_primary_output_name(): outputs["out"]}

    def resolve_tap(self, tap_name: str, x=None):  # type: ignore[no-untyped-def]
        raise ValueError(f"Tap '{tap_name}' is not supported by adapter '{self.family}'")


def build_deeplabv3_adapter(model_config: dict[str, Any]) -> FamilyAdapter:
    architecture = model_config.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("Model config architecture is missing")
    backbone = architecture.get("backbone", {})
    if not isinstance(backbone, dict):
        raise ValueError("Model config architecture.backbone is missing")
    head = architecture.get("head", {})
    if not isinstance(head, dict):
        raise ValueError("Model config architecture.head is missing")

    return TorchvisionDeepLabV3Adapter(
        backbone_name=str(backbone.get("name", "")).strip().lower(),
        num_classes=int(head.get("num_classes", 0)),
        _pretrained=bool(backbone.get("pretrained")),
    )
