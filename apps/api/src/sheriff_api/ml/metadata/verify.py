from __future__ import annotations

from typing import Any

from sheriff_api.ml.metadata.backbones import get_backbone_meta


def _issue(*, tap: str, field: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "tap": tap,
        "field": field,
        "expected": expected,
        "actual": actual,
        "message": f"{tap}: expected {field}={expected}, got {actual}",
    }


def verify_backbone_meta(backbone_name: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    try:
        metadata = get_backbone_meta(backbone_name)
    except KeyError as exc:
        return [{"tap": "$", "field": "backbone", "expected": "supported", "actual": backbone_name, "message": str(exc)}]

    if metadata.family != "resnet":
        return [{"tap": "$", "field": "family", "expected": "resnet", "actual": metadata.family, "message": "Only resnet metadata verification is implemented for v0"}]

    try:
        import torch
        import torchvision.models as tv_models
    except Exception as exc:  # pragma: no cover - exercised only in non-ML environments
        return [{"tap": "$", "field": "imports", "expected": "torch+torchvision", "actual": "missing", "message": str(exc)}]

    ctor = getattr(tv_models, backbone_name, None)
    if ctor is None:
        return [{"tap": "$", "field": "constructor", "expected": backbone_name, "actual": "missing", "message": f"torchvision has no backbone '{backbone_name}'"}]

    model = ctor(weights=None).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        stem = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        c2 = model.layer1(stem)
        c3 = model.layer2(c2)
        c4 = model.layer3(c3)
        c5 = model.layer4(c4)
        global_pool = torch.flatten(model.avgpool(c5), 1)

    tap_tensors = {
        "backbone.global_pool": global_pool,
        "backbone.c3": c3,
        "backbone.c4": c4,
        "backbone.c5": c5,
    }
    input_size = x.shape[-1]

    for tap_name, expected in metadata.taps.items():
        tensor = tap_tensors.get(tap_name)
        if tensor is None:
            issues.append(_issue(tap=tap_name, field="exists", expected=True, actual=False))
            continue
        actual_channels = int(tensor.shape[1])
        if actual_channels != expected.channels:
            issues.append(_issue(tap=tap_name, field="channels", expected=expected.channels, actual=actual_channels))
        if expected.stride is not None:
            spatial = int(tensor.shape[-1])
            actual_stride = input_size // spatial if spatial > 0 else None
            if actual_stride != expected.stride:
                issues.append(_issue(tap=tap_name, field="stride", expected=expected.stride, actual=actual_stride))

    return issues
