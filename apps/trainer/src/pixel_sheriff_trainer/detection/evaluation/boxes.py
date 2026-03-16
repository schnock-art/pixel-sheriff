from __future__ import annotations

from typing import Iterable


XYXYBox = tuple[float, float, float, float]

SIZE_BUCKET_ORDER = ("small", "medium", "large")
SIZE_BUCKET_THRESHOLDS = {
    "small": 32.0 * 32.0,
    "medium": 96.0 * 96.0,
}


def bbox_xywh_to_xyxy(bbox: Iterable[float]) -> XYXYBox:
    x, y, width, height = [float(value) for value in bbox]
    return (x, y, x + width, y + height)


def bbox_xyxy_to_xywh(bbox: Iterable[float]) -> XYXYBox:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def normalize_bbox(bbox: Iterable[float], *, box_format: str) -> XYXYBox:
    values = tuple(float(value) for value in bbox)
    if len(values) != 4:
        raise ValueError("bbox_must_have_four_values")
    if box_format == "xyxy":
        x1, y1, x2, y2 = values
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        return (left, top, right, bottom)
    if box_format == "xywh":
        return bbox_xywh_to_xyxy(values)
    raise ValueError(f"unsupported_box_format:{box_format}")


def bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_iou(left: Iterable[float], right: Iterable[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = bbox_area((ix1, iy1, ix2, iy2))
    if intersection <= 0:
        return 0.0
    union = bbox_area((lx1, ly1, lx2, ly2)) + bbox_area((rx1, ry1, rx2, ry2)) - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def size_bucket_for_area(area: float) -> str:
    if area < SIZE_BUCKET_THRESHOLDS["small"]:
        return "small"
    if area < SIZE_BUCKET_THRESHOLDS["medium"]:
        return "medium"
    return "large"
