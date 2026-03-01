from __future__ import annotations

from typing import Any, Callable

from sheriff_api.ml.adapters.base import FamilyAdapter


AdapterBuilder = Callable[[dict[str, Any]], FamilyAdapter]


def _build_resnet_classifier(model_config: dict[str, Any]) -> FamilyAdapter:
    from sheriff_api.ml.adapters.torchvision_resnet_classifier import build_resnet_classifier_adapter

    return build_resnet_classifier_adapter(model_config)


def _build_retinanet(model_config: dict[str, Any]) -> FamilyAdapter:
    from sheriff_api.ml.adapters.torchvision_retinanet import build_retinanet_adapter

    return build_retinanet_adapter(model_config)


def _build_deeplabv3(model_config: dict[str, Any]) -> FamilyAdapter:
    from sheriff_api.ml.adapters.torchvision_deeplabv3 import build_deeplabv3_adapter

    return build_deeplabv3_adapter(model_config)


FAMILY_REGISTRY: dict[str, AdapterBuilder] = {
    "resnet_classifier": _build_resnet_classifier,
    "retinanet": _build_retinanet,
    "deeplabv3": _build_deeplabv3,
}


def list_registered_families() -> list[str]:
    return sorted(FAMILY_REGISTRY.keys())


def build_family_adapter(model_config: dict[str, Any]) -> FamilyAdapter:
    architecture = model_config.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("Model config architecture is missing")

    family = architecture.get("family")
    if not isinstance(family, str) or not family.strip():
        raise ValueError("Model config architecture.family is required")
    normalized_family = family.strip()

    builder = FAMILY_REGISTRY.get(normalized_family)
    if builder is None:
        supported = ", ".join(list_registered_families())
        raise ValueError(f"Unsupported model family '{normalized_family}'. Supported families: {supported}")
    return builder(model_config)
