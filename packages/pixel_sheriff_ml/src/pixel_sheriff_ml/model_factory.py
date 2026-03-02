from __future__ import annotations

from typing import Any


def architecture_family(model_config: dict[str, Any]) -> str:
    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        return "unknown"
    family = architecture.get("family")
    return str(family).strip().lower() if family is not None else "unknown"


def _backbone_name(model_config: dict[str, Any]) -> str:
    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        return "resnet18"
    backbone = architecture.get("backbone")
    if not isinstance(backbone, dict):
        return "resnet18"
    name = backbone.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    return "resnet18"


def _head_num_classes(model_config: dict[str, Any], num_classes_override: int | None = None) -> int:
    if isinstance(num_classes_override, int) and num_classes_override >= 1:
        return num_classes_override
    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        return 0
    head = architecture.get("head")
    if not isinstance(head, dict):
        return 0
    raw_num_classes = head.get("num_classes")
    if isinstance(raw_num_classes, int):
        return raw_num_classes
    return 0


def _backbone_pretrained(model_config: dict[str, Any]) -> bool:
    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        return False
    backbone = architecture.get("backbone")
    if not isinstance(backbone, dict):
        return False
    return bool(backbone.get("pretrained"))


def build_resnet_classifier(model_config: dict[str, Any], *, num_classes_override: int | None = None):
    family = architecture_family(model_config)
    if family != "resnet_classifier":
        raise ValueError("unsupported_family")

    import torch.nn as nn
    import torchvision.models as tv_models

    backbone_name = _backbone_name(model_config)
    ctor = getattr(tv_models, backbone_name, None)
    if ctor is None:
        raise ValueError(f"unsupported_backbone:{backbone_name}")

    num_classes = _head_num_classes(model_config, num_classes_override=num_classes_override)
    if num_classes < 1:
        raise ValueError("invalid_num_classes")

    # Keep deterministic local behavior: no implicit weights download.
    _pretrained = _backbone_pretrained(model_config)
    model = ctor(weights=None)
    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, int(num_classes))
    return model

