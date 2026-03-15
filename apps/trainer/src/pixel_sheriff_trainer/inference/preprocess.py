from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessContext:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    resize_policy: str
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int


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


def _resize_with_context(
    image: Image.Image,
    *,
    width: int,
    height: int,
    resize_policy: str,
) -> tuple[Image.Image, PreprocessContext]:
    src_w, src_h = image.size
    if resize_policy == "letterbox":
        if src_w <= 0 or src_h <= 0:
            resized = image.resize((width, height))
            context = PreprocessContext(
                original_width=max(1, src_w),
                original_height=max(1, src_h),
                target_width=width,
                target_height=height,
                resize_policy="letterbox",
                resized_width=width,
                resized_height=height,
                offset_x=0,
                offset_y=0,
            )
            return resized, context
        scale = min(width / src_w, height / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = image.resize((resized_w, resized_h))
        canvas = Image.new("RGB", (width, height), color=(114, 114, 114))
        offset_x = (width - resized_w) // 2
        offset_y = (height - resized_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        context = PreprocessContext(
            original_width=src_w,
            original_height=src_h,
            target_width=width,
            target_height=height,
            resize_policy="letterbox",
            resized_width=resized_w,
            resized_height=resized_h,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        return canvas, context

    resized = image.resize((width, height))
    context = PreprocessContext(
        original_width=max(1, src_w),
        original_height=max(1, src_h),
        target_width=width,
        target_height=height,
        resize_policy="stretch",
        resized_width=width,
        resized_height=height,
        offset_x=0,
        offset_y=0,
    )
    return resized, context


def preprocess_asset_with_context(asset_path: Path, metadata: dict[str, Any]) -> tuple[np.ndarray, PreprocessContext]:
    width, height, resize_policy = _resize_target_from_metadata(metadata)
    mean, std = _normalization_from_metadata(metadata)

    with Image.open(asset_path) as image:
        rgb = image.convert("RGB")
        resized, context = _resize_with_context(rgb, width=width, height=height, resize_policy=resize_policy)
        array = np.asarray(resized, dtype=np.float32) / 255.0

    chw = np.transpose(array, (2, 0, 1))
    if mean is not None and std is not None:
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        chw = (chw - mean_arr) / std_arr
    return np.expand_dims(chw.astype(np.float32, copy=False), axis=0), context


def preprocess_asset(asset_path: Path, metadata: dict[str, Any]) -> np.ndarray:
    tensor, _context = preprocess_asset_with_context(asset_path, metadata)
    return tensor


def remap_bbox_xyxy_to_original_xywh(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    context: PreprocessContext,
) -> list[float] | None:
    if len(bbox_xyxy) != 4:
        return None

    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    if context.resize_policy == "letterbox":
        scale_x = context.resized_width / max(context.original_width, 1)
        scale_y = context.resized_height / max(context.original_height, 1)
        x_min = (x_min - context.offset_x) / max(scale_x, 1e-8)
        x_max = (x_max - context.offset_x) / max(scale_x, 1e-8)
        y_min = (y_min - context.offset_y) / max(scale_y, 1e-8)
        y_max = (y_max - context.offset_y) / max(scale_y, 1e-8)
    else:
        scale_x = context.original_width / max(context.target_width, 1)
        scale_y = context.original_height / max(context.target_height, 1)
        x_min *= scale_x
        x_max *= scale_x
        y_min *= scale_y
        y_max *= scale_y

    x_min = min(max(x_min, 0.0), float(context.original_width))
    y_min = min(max(y_min, 0.0), float(context.original_height))
    x_max = min(max(x_max, 0.0), float(context.original_width))
    y_max = min(max(y_max, 0.0), float(context.original_height))

    if x_max <= x_min or y_max <= y_min:
        return None

    return [x_min, y_min, x_max - x_min, y_max - y_min]
