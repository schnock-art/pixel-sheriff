from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TapKind = Literal["embedding", "feature_map"]


@dataclass(frozen=True)
class TapMeta:
    kind: TapKind
    channels: int
    stride: int | None = None


@dataclass(frozen=True)
class BackboneMeta:
    name: str
    family: str
    embedding_dim: int
    taps: dict[str, TapMeta]
    default_out_strides: tuple[int, ...]


_TAP_ALIASES: dict[str, str] = {
    "backbone.avgpool": "backbone.global_pool",
}


BACKBONES: dict[str, BackboneMeta] = {
    "resnet18": BackboneMeta(
        name="resnet18",
        family="resnet",
        embedding_dim=512,
        taps={
            "backbone.global_pool": TapMeta(kind="embedding", channels=512, stride=None),
            "backbone.c3": TapMeta(kind="feature_map", channels=128, stride=8),
            "backbone.c4": TapMeta(kind="feature_map", channels=256, stride=16),
            "backbone.c5": TapMeta(kind="feature_map", channels=512, stride=32),
        },
        default_out_strides=(8, 16, 32),
    ),
    "resnet34": BackboneMeta(
        name="resnet34",
        family="resnet",
        embedding_dim=512,
        taps={
            "backbone.global_pool": TapMeta(kind="embedding", channels=512, stride=None),
            "backbone.c3": TapMeta(kind="feature_map", channels=128, stride=8),
            "backbone.c4": TapMeta(kind="feature_map", channels=256, stride=16),
            "backbone.c5": TapMeta(kind="feature_map", channels=512, stride=32),
        },
        default_out_strides=(8, 16, 32),
    ),
    "resnet50": BackboneMeta(
        name="resnet50",
        family="resnet",
        embedding_dim=2048,
        taps={
            "backbone.global_pool": TapMeta(kind="embedding", channels=2048, stride=None),
            "backbone.c3": TapMeta(kind="feature_map", channels=512, stride=8),
            "backbone.c4": TapMeta(kind="feature_map", channels=1024, stride=16),
            "backbone.c5": TapMeta(kind="feature_map", channels=2048, stride=32),
        },
        default_out_strides=(8, 16, 32),
    ),
    "resnet101": BackboneMeta(
        name="resnet101",
        family="resnet",
        embedding_dim=2048,
        taps={
            "backbone.global_pool": TapMeta(kind="embedding", channels=2048, stride=None),
            "backbone.c3": TapMeta(kind="feature_map", channels=512, stride=8),
            "backbone.c4": TapMeta(kind="feature_map", channels=1024, stride=16),
            "backbone.c5": TapMeta(kind="feature_map", channels=2048, stride=32),
        },
        default_out_strides=(8, 16, 32),
    ),
}


def normalize_tap_name(tap_name: str) -> str:
    normalized = tap_name.strip().lower()
    if not normalized:
        raise ValueError("tap_name must be a non-empty string")
    if "." not in normalized:
        normalized = f"backbone.{normalized}"
    return _TAP_ALIASES.get(normalized, normalized)


def get_backbone_meta(name: str) -> BackboneMeta:
    normalized = name.strip().lower()
    if normalized not in BACKBONES:
        raise KeyError(f"Unsupported backbone metadata: {name}")
    return BACKBONES[normalized]


def get_tap_meta(backbone_name: str, tap_name: str) -> TapMeta:
    backbone = get_backbone_meta(backbone_name)
    normalized_tap = normalize_tap_name(tap_name)
    tap_meta = backbone.taps.get(normalized_tap)
    if tap_meta is None:
        raise KeyError(f"Unsupported tap '{tap_name}' for backbone '{backbone_name}'")
    return tap_meta


def list_supported_taps(backbone_name: str) -> list[str]:
    backbone = get_backbone_meta(backbone_name)
    return sorted(backbone.taps.keys())
