from __future__ import annotations

from typing import Any

__all__ = [
    "BACKBONES",
    "BackboneMeta",
    "TapMeta",
    "build_registry_payload",
    "get_backbone_meta",
    "get_tap_meta",
    "list_supported_taps",
    "normalize_tap_name",
    "verify_backbone_meta",
    "write_registry_json",
]


def __getattr__(name: str) -> Any:
    if name in {
        "BACKBONES",
        "BackboneMeta",
        "TapMeta",
        "get_backbone_meta",
        "get_tap_meta",
        "list_supported_taps",
        "normalize_tap_name",
    }:
        from sheriff_api.ml.metadata import backbones as _backbones

        return getattr(_backbones, name)
    if name in {"build_registry_payload", "write_registry_json"}:
        from sheriff_api.ml.metadata import generate_registry_json as _generator

        return getattr(_generator, name)
    if name == "verify_backbone_meta":
        from sheriff_api.ml.metadata.verify import verify_backbone_meta

        return verify_backbone_meta
    raise AttributeError(name)
