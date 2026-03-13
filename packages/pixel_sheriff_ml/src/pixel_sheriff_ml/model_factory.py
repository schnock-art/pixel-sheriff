from __future__ import annotations

from typing import Any


_CLASSIFIER_WEIGHT_ENUMS: dict[str, str] = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
    "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
    "efficientnet_v2_l": "EfficientNet_V2_L_Weights",
}

_CLASSIFIER_FAMILY_BACKBONES: dict[str, set[str]] = {
    "resnet_classifier": {"resnet18", "resnet34", "resnet50", "resnet101"},
    "efficientnet_v2_classifier": {"efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"},
}


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


def _require_supported_classifier_family(model_config: dict[str, Any]) -> str:
    family = architecture_family(model_config)
    if family not in _CLASSIFIER_FAMILY_BACKBONES:
        raise ValueError("unsupported_family")
    return family


def _resolve_classifier_weights(tv_models, backbone_name: str, pretrained: bool):  # type: ignore[no-untyped-def]
    if not pretrained:
        return None
    enum_name = _CLASSIFIER_WEIGHT_ENUMS.get(backbone_name)
    if not enum_name:
        raise ValueError(f"unsupported_backbone:{backbone_name}")
    weights_enum = getattr(tv_models, enum_name, None)
    if weights_enum is None:
        raise ValueError(f"unsupported_backbone:{backbone_name}")
    return weights_enum.DEFAULT


def _replace_classifier_head(model, family: str, num_classes: int, nn):  # type: ignore[no-untyped-def]
    if family == "resnet_classifier":
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, int(num_classes))
        return model

    if family == "efficientnet_v2_classifier":
        classifier = getattr(model, "classifier", None)
        if not isinstance(classifier, nn.Sequential) or len(classifier) < 2 or not isinstance(classifier[1], nn.Linear):
            raise ValueError("unsupported_classifier_head")
        in_features = int(classifier[1].in_features)
        classifier[1] = nn.Linear(in_features, int(num_classes))
        return model

    raise ValueError("unsupported_family")


def build_classifier_model(model_config: dict[str, Any], *, num_classes_override: int | None = None):
    family = _require_supported_classifier_family(model_config)

    import torch.nn as nn
    import torchvision.models as tv_models

    backbone_name = _backbone_name(model_config)
    supported_backbones = _CLASSIFIER_FAMILY_BACKBONES[family]
    if backbone_name not in supported_backbones:
        raise ValueError(f"unsupported_backbone:{backbone_name}")
    ctor = getattr(tv_models, backbone_name, None)
    if ctor is None:
        raise ValueError(f"unsupported_backbone:{backbone_name}")

    num_classes = _head_num_classes(model_config, num_classes_override=num_classes_override)
    if num_classes < 1:
        raise ValueError("invalid_num_classes")

    pretrained = _backbone_pretrained(model_config)
    weights = _resolve_classifier_weights(tv_models, backbone_name, pretrained)
    try:
        model = ctor(weights=weights)
    except Exception as exc:
        if pretrained:
            raise ValueError(f"pretrained_weights_unavailable:{backbone_name}:{exc}") from exc
        raise

    return _replace_classifier_head(model, family, num_classes, nn)


def build_resnet_classifier(model_config: dict[str, Any], *, num_classes_override: int | None = None):
    family = architecture_family(model_config)
    if family != "resnet_classifier":
        raise ValueError("unsupported_family")
    return build_classifier_model(model_config, num_classes_override=num_classes_override)
