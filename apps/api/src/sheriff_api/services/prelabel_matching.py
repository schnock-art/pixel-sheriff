from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from sheriff_api.db.models import Asset, Category, PrelabelSession
from sheriff_api.services.prelabel_adapters import DetectionResult


PRELABEL_DEBUG_DETECTIONS_LIMIT = 200

PRELABEL_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"human", "person", "people"}),
    frozenset({"glass", "glasses", "eyeglasses", "eye glasses", "spectacles", "specs", "eyewear"}),
    frozenset({"eye", "eyes", "eyeball", "eyeballs"}),
    frozenset({"mouth", "mouths", "lip", "lips"}),
    frozenset({"head", "face"}),
)
PRELABEL_ALIAS_LOOKUP: dict[str, set[str]] = {}
for _group in PRELABEL_ALIAS_GROUPS:
    for _value in _group:
        PRELABEL_ALIAS_LOOKUP[_value] = set(_group - {_value})


def normalized_debug_detections(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    detections: list[dict[str, Any]] = []
    for row in raw_rows[:PRELABEL_DEBUG_DETECTIONS_LIMIT]:
        if not isinstance(row, dict):
            continue
        asset_id = str(row.get("asset_id") or "").strip()
        label_text = str(row.get("label_text") or "").strip()
        bbox_xyxy = row.get("bbox_xyxy")
        status = str(row.get("status") or "").strip().lower()
        if not asset_id or not label_text:
            continue
        if status not in {"matched", "unmatched", "discarded"}:
            continue
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in bbox_xyxy):
            continue
        asset_frame_index = row.get("asset_frame_index")
        detections.append(
            {
                "asset_id": asset_id,
                "asset_frame_index": int(asset_frame_index) if isinstance(asset_frame_index, int) else None,
                "label_text": label_text,
                "resolved_category_id": str(row.get("resolved_category_id") or "").strip() or None,
                "resolved_category_name": str(row.get("resolved_category_name") or "").strip() or None,
                "confidence": float(row.get("confidence") or 0.0),
                "bbox_xyxy": [float(value) for value in bbox_xyxy],
                "status": status,
            }
        )
    return detections


def append_debug_detection(
    session: PrelabelSession,
    *,
    asset: Asset,
    detection: DetectionResult,
    status: str,
    category: Category | None,
) -> None:
    debug_detections = normalized_debug_detections(session.debug_detections_json)
    debug_detections.append(
        {
            "asset_id": asset.id,
            "asset_frame_index": int(asset.frame_index) if isinstance(asset.frame_index, int) else None,
            "label_text": str(detection.label_text or "").strip(),
            "resolved_category_id": category.id if category is not None else None,
            "resolved_category_name": category.name if category is not None else None,
            "confidence": float(detection.score),
            "bbox_xyxy": [float(value) for value in detection.bbox_xyxy],
            "status": status,
        }
    )
    session.debug_detections_json = debug_detections[-PRELABEL_DEBUG_DETECTIONS_LIMIT:]


def normalize_prelabel_label_key(value: str | None) -> str:
    chunks: list[str] = []
    current: list[str] = []
    for char in str(value or ""):
        if char.isalnum():
            current.append(char.lower())
            continue
        if current:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return " ".join(chunks)


def inflection_alias_keys(normalized: str) -> set[str]:
    keys: set[str] = set()
    if not normalized:
        return keys

    if normalized.endswith("ies") and len(normalized) > 4:
        keys.add(f"{normalized[:-3]}y")
    if normalized.endswith("sses") and len(normalized) > 5:
        keys.add(normalized[:-2])
    elif normalized.endswith(("ches", "shes", "xes", "zes")) and len(normalized) > 4:
        keys.add(normalized[:-2])
    if normalized.endswith("s") and len(normalized) > 3 and not normalized.endswith("ss"):
        keys.add(normalized[:-1])

    if normalized.endswith("y") and len(normalized) > 1 and normalized[-2] not in {"a", "e", "i", "o", "u"}:
        keys.add(f"{normalized[:-1]}ies")
    elif normalized.endswith(("s", "x", "z", "ch", "sh")):
        keys.add(f"{normalized}es")
    else:
        keys.add(f"{normalized}s")
    return {key for key in keys if key and key != normalized}


def category_exact_keys(value: str | None) -> list[str]:
    normalized = normalize_prelabel_label_key(value)
    if not normalized:
        return []
    keys = [normalized]
    collapsed = normalized.replace(" ", "")
    if collapsed and collapsed != normalized:
        keys.append(collapsed)
    return keys


def category_alias_keys(value: str | None) -> set[str]:
    normalized = normalize_prelabel_label_key(value)
    if not normalized:
        return set()

    keys: set[str] = set()
    collapsed = normalized.replace(" ", "")
    if collapsed and collapsed != normalized:
        keys.add(collapsed)
    keys.update(inflection_alias_keys(normalized))
    for alias in PRELABEL_ALIAS_LOOKUP.get(normalized, set()):
        alias_key = normalize_prelabel_label_key(alias)
        if not alias_key:
            continue
        keys.add(alias_key)
        collapsed_alias = alias_key.replace(" ", "")
        if collapsed_alias and collapsed_alias != alias_key:
            keys.add(collapsed_alias)
        keys.update(inflection_alias_keys(alias_key))
    return keys - set(category_exact_keys(value))


def category_match_maps(categories: list[Category]) -> tuple[dict[str, Category], dict[str, Category]]:
    exact_mapping: dict[str, Category] = {}
    alias_candidates: dict[str, dict[str, Category]] = defaultdict(dict)
    for category in categories:
        for key in category_exact_keys(category.name):
            if key not in exact_mapping:
                exact_mapping[key] = category
        for key in category_alias_keys(category.name):
            alias_candidates[key][category.id] = category
    alias_mapping = {
        key: next(iter(categories_by_id.values()))
        for key, categories_by_id in alias_candidates.items()
        if len(categories_by_id) == 1 and key not in exact_mapping
    }
    return exact_mapping, alias_mapping


def match_detection_category(
    *,
    label_text: str | None,
    exact_mapping: dict[str, Category],
    alias_mapping: dict[str, Category],
) -> Category | None:
    for key in category_exact_keys(label_text):
        category = exact_mapping.get(key)
        if category is not None:
            return category
    for key in category_alias_keys(label_text):
        category = alias_mapping.get(key)
        if category is not None:
            return category
    return None


def normalize_xywh_bbox(bbox: list[Any] | None) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in bbox):
        return None
    x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if width <= 0 or height <= 0:
        return None
    return [x, y, width, height]


def bbox_xyxy_to_xywh(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int | None,
    height: int | None,
) -> list[float] | None:
    x1, y1, x2, y2 = bbox_xyxy
    if width is not None and width > 0:
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
    if height is not None and height > 0:
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
    left = min(x1, x2)
    top = min(y1, y2)
    box_width = max(x1, x2) - left
    box_height = max(y1, y2) - top
    if box_width <= 0 or box_height <= 0:
        return None
    return [left, top, box_width, box_height]
