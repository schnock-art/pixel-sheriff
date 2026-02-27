from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from sheriff_api.db.models import TaskType

@dataclass
class PayloadValidationError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None


def _as_payload_dict(payload_json: Any) -> dict[str, Any]:
    if isinstance(payload_json, BaseModel):
        return payload_json.model_dump()
    if isinstance(payload_json, dict):
        return dict(payload_json)
    raise PayloadValidationError(
        code="annotation_payload_invalid",
        message="Annotation payload must be an object",
    )


def _normalize_label_ids(raw_values: Any) -> list[int]:
    if not isinstance(raw_values, list):
        return []

    values: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        if not isinstance(value, int):
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _normalize_image_basis(payload: dict[str, Any], asset_width: int | None, asset_height: int | None) -> dict[str, int] | None:
    basis = payload.get("image_basis")
    if isinstance(basis, dict):
        raw_width = basis.get("width")
        raw_height = basis.get("height")
        if not isinstance(raw_width, int) or raw_width <= 0 or not isinstance(raw_height, int) or raw_height <= 0:
            raise PayloadValidationError(
                code="annotation_geometry_invalid_image_basis",
                message="image_basis must include positive integer width and height",
            )
        return {"width": raw_width, "height": raw_height}

    if isinstance(asset_width, int) and asset_width > 0 and isinstance(asset_height, int) and asset_height > 0:
        return {"width": asset_width, "height": asset_height}
    return None


def _validate_category(category_id: int, allowed_category_ids: set[int], object_id: str | None = None) -> None:
    if category_id not in allowed_category_ids:
        details: dict[str, Any] = {"category_id": category_id}
        if object_id:
            details["object_id"] = object_id
        raise PayloadValidationError(
            code="annotation_category_invalid",
            message="Annotation category is not part of this project",
            details=details,
        )


def _validate_in_basis(x: float, y: float, basis: dict[str, int], object_id: str) -> None:
    if x < 0 or y < 0 or x > basis["width"] or y > basis["height"]:
        raise PayloadValidationError(
            code="annotation_geometry_out_of_bounds",
            message="Geometry coordinates must be within image bounds",
            details={"object_id": object_id, "x": x, "y": y, "image_basis": basis},
        )


def _normalize_geometry_objects(
    raw_objects: Any,
    *,
    allowed_category_ids: set[int],
    image_basis: dict[str, int] | None,
) -> list[dict[str, Any]]:
    if raw_objects is None:
        return []
    if not isinstance(raw_objects, list):
        raise PayloadValidationError(
            code="annotation_geometry_invalid",
            message="objects must be an array",
        )

    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_objects:
        if not isinstance(raw, dict):
            raise PayloadValidationError(
                code="annotation_geometry_invalid",
                message="Each object entry must be an object",
            )

        object_id = raw.get("id")
        kind = raw.get("kind")
        category_id = raw.get("category_id")
        if not isinstance(object_id, str) or object_id.strip() == "":
            raise PayloadValidationError(
                code="annotation_geometry_invalid",
                message="Geometry object id is required",
            )
        if object_id in seen_ids:
            raise PayloadValidationError(
                code="annotation_geometry_duplicate_id",
                message="Geometry object ids must be unique per annotation",
                details={"object_id": object_id},
            )
        seen_ids.add(object_id)

        if not isinstance(category_id, int):
            raise PayloadValidationError(
                code="annotation_geometry_invalid",
                message="Geometry object category_id must be an integer",
                details={"object_id": object_id},
            )
        _validate_category(category_id, allowed_category_ids, object_id)

        if kind == "bbox":
            raw_bbox = raw.get("bbox")
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4 or not all(_valid_number(value) for value in raw_bbox):
                raise PayloadValidationError(
                    code="annotation_bbox_invalid",
                    message="BBox must be [x, y, width, height] numeric values",
                    details={"object_id": object_id},
                )
            x, y, width, height = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise PayloadValidationError(
                    code="annotation_bbox_invalid",
                    message="BBox coordinates must be non-negative and dimensions must be > 0",
                    details={"object_id": object_id, "bbox": raw_bbox},
                )
            if image_basis is not None:
                _validate_in_basis(x, y, image_basis, object_id)
                _validate_in_basis(x + width, y + height, image_basis, object_id)

            objects.append(
                {
                    "id": object_id,
                    "kind": "bbox",
                    "category_id": category_id,
                    "bbox": [x, y, width, height],
                }
            )
            continue

        if kind == "polygon":
            raw_segmentation = raw.get("segmentation")
            if not isinstance(raw_segmentation, list) or len(raw_segmentation) == 0:
                raise PayloadValidationError(
                    code="annotation_polygon_invalid",
                    message="Polygon segmentation must include at least one polygon",
                    details={"object_id": object_id},
                )

            normalized_segments: list[list[float]] = []
            for segment in raw_segmentation:
                if not isinstance(segment, list):
                    raise PayloadValidationError(
                        code="annotation_polygon_invalid",
                        message="Polygon points must be an array",
                        details={"object_id": object_id},
                    )
                if len(segment) < 6 or len(segment) % 2 != 0:
                    raise PayloadValidationError(
                        code="annotation_polygon_invalid",
                        message="Polygon point list must contain an even number of values and at least 3 points",
                        details={"object_id": object_id},
                    )
                if not all(_valid_number(value) for value in segment):
                    raise PayloadValidationError(
                        code="annotation_polygon_invalid",
                        message="Polygon points must be numeric values",
                        details={"object_id": object_id},
                    )

                normalized_segment = [float(value) for value in segment]
                if image_basis is not None:
                    for index in range(0, len(normalized_segment), 2):
                        _validate_in_basis(normalized_segment[index], normalized_segment[index + 1], image_basis, object_id)
                normalized_segments.append(normalized_segment)

            objects.append(
                {
                    "id": object_id,
                    "kind": "polygon",
                    "category_id": category_id,
                    "segmentation": normalized_segments,
                }
            )
            continue

        raise PayloadValidationError(
            code="annotation_geometry_invalid_kind",
            message="Geometry object kind must be 'bbox' or 'polygon'",
            details={"object_id": object_id, "kind": kind},
        )

    return sorted(objects, key=lambda item: str(item.get("id", "")))


def normalize_annotation_payload(
    payload_json: Any,
    *,
    task_type: TaskType,
    allowed_category_ids: set[int],
    asset_width: int | None,
    asset_height: int | None,
) -> dict[str, Any]:
    payload = _as_payload_dict(payload_json)
    image_basis = _normalize_image_basis(payload, asset_width, asset_height)

    classification = payload.get("classification")
    if isinstance(classification, dict):
        category_ids = _normalize_label_ids(classification.get("category_ids"))
        primary_category_id = classification.get("primary_category_id")
        if primary_category_id is not None and not isinstance(primary_category_id, int):
            raise PayloadValidationError(
                code="annotation_payload_invalid",
                message="classification.primary_category_id must be an integer when provided",
            )
    else:
        category_ids = _normalize_label_ids(payload.get("category_ids"))
        category_id = payload.get("category_id")
        if isinstance(category_id, int) and category_id not in category_ids:
            category_ids = [category_id, *category_ids]
        primary_category_id = category_id if isinstance(category_id, int) else None

    for category_id in category_ids:
        _validate_category(category_id, allowed_category_ids)

    category_ids = sorted(dict.fromkeys(category_ids))

    if primary_category_id is None and category_ids:
        primary_category_id = category_ids[0]
    if primary_category_id is not None:
        _validate_category(primary_category_id, allowed_category_ids)
        if primary_category_id not in category_ids:
            category_ids = [primary_category_id, *category_ids]

    objects = _normalize_geometry_objects(payload.get("objects"), allowed_category_ids=allowed_category_ids, image_basis=image_basis)
    if task_type in {TaskType.classification, TaskType.classification_single} and objects:
        raise PayloadValidationError(
            code="annotation_task_mode_mismatch",
            message="Project task mode does not allow geometry objects",
            details={"task_type": task_type.value},
        )
    if task_type == TaskType.bbox and any(object_value.get("kind") != "bbox" for object_value in objects):
        raise PayloadValidationError(
            code="annotation_task_mode_mismatch",
            message="Project task mode only allows bounding box objects",
            details={"task_type": task_type.value},
        )
    if task_type == TaskType.segmentation and any(object_value.get("kind") != "polygon" for object_value in objects):
        raise PayloadValidationError(
            code="annotation_task_mode_mismatch",
            message="Project task mode only allows segmentation polygon objects",
            details={"task_type": task_type.value},
        )
    source = payload.get("source")

    return {
        "version": "2.0",
        "type": "classification",
        "category_id": primary_category_id,
        "category_ids": category_ids,
        "classification": {
            "category_ids": category_ids,
            "primary_category_id": primary_category_id,
        },
        "objects": objects,
        "image_basis": image_basis,
        "coco": {
            "image_id": payload.get("coco", {}).get("image_id") if isinstance(payload.get("coco"), dict) else None,
            "category_id": primary_category_id,
        },
        "source": source if isinstance(source, str) and source.strip() else "web-ui",
    }
