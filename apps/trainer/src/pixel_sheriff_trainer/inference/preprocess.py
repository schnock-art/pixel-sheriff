from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalization_from_metadata(metadata: dict[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    preprocess = metadata.get("preprocess")
    if not isinstance(preprocess, dict):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalization = preprocess.get("normalization")
    if not isinstance(normalization, dict):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm_type = str(normalization.get("type") or "imagenet").strip().lower()
    if norm_type == "none":
        return None, None
    if norm_type == "custom":
        mean = normalization.get("mean")
        std = normalization.get("std")
        if isinstance(mean, list) and isinstance(std, list) and len(mean) == 3 and len(std) == 3:
            return [float(value) for value in mean], [float(value) for value in std]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _resize_target_from_metadata(metadata: dict[str, Any]) -> tuple[int, int, str]:
    preprocess = metadata.get("preprocess")
    if not isinstance(preprocess, dict):
        return (224, 224, "stretch")
    resize = preprocess.get("resize")
    if not isinstance(resize, dict):
        return (224, 224, "stretch")
    width = resize.get("width")
    height = resize.get("height")
    resize_policy = str(preprocess.get("resize_policy") or "stretch").strip().lower()
    if resize_policy not in {"stretch", "letterbox"}:
        resize_policy = "stretch"
    if not isinstance(width, int) or width <= 0:
        width = 224
    if not isinstance(height, int) or height <= 0:
        height = 224
    return (width, height, resize_policy)


def _letterbox(image: Image.Image, *, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return image.resize((width, height))
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = image.resize((resized_w, resized_h))
    canvas = Image.new("RGB", (width, height), color=(114, 114, 114))
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def preprocess_asset(asset_path: Path, metadata: dict[str, Any]) -> np.ndarray:
    width, height, resize_policy = _resize_target_from_metadata(metadata)
    mean, std = _normalization_from_metadata(metadata)

    with Image.open(asset_path) as image:
        rgb = image.convert("RGB")
        if resize_policy == "letterbox":
            resized = _letterbox(rgb, width=width, height=height)
        else:
            resized = rgb.resize((width, height))
        array = np.asarray(resized, dtype=np.float32) / 255.0

    chw = np.transpose(array, (2, 0, 1))
    if mean is not None and std is not None:
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        chw = (chw - mean_arr) / std_arr
    return np.expand_dims(chw.astype(np.float32, copy=False), axis=0)
