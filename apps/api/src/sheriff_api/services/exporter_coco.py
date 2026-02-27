from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import math
from pathlib import PurePosixPath
import re
from typing import Any, Callable
import uuid
import zipfile

from sheriff_api.db.models import TaskType
from sheriff_api.services.hashing import stable_hash


@dataclass
class ExportValidationError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None


def _err(code: str, message: str, details: dict[str, Any] | None = None) -> ExportValidationError:
    return ExportValidationError(code=code, message=message, details=details)


def _safe_relative_path(value: str, fallback_filename: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("", ".", "..")]
    return "/".join(parts) if parts else fallback_filename


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _numbers(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def _poly_bbox_area(segmentation: list[list[float]]) -> tuple[list[float], float]:
    xs: list[float] = []
    ys: list[float] = []
    total = 0.0
    for points in segmentation:
        if len(points) < 6:
            continue
        px = points[0::2]
        py = points[1::2]
        if not px or not py:
            continue
        xs.extend(px)
        ys.extend(py)
        area = 0.0
        for i in range(len(px)):
            j = (i + 1) % len(px)
            area += px[i] * py[j] - px[j] * py[i]
        total += abs(area) / 2.0
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)], total


def _annotation_classes(payload: dict[str, Any]) -> tuple[list[int], int | None]:
    class_ids: list[int] = []
    primary: int | None = None
    block = payload.get("classification")
    if isinstance(block, dict):
        raw_ids = block.get("category_ids")
        if isinstance(raw_ids, list):
            class_ids = [v for v in raw_ids if isinstance(v, int)]
        raw_primary = block.get("primary_category_id")
        if isinstance(raw_primary, int):
            primary = raw_primary
    if not class_ids:
        raw_ids = payload.get("category_ids")
        if isinstance(raw_ids, list):
            class_ids = [v for v in raw_ids if isinstance(v, int)]
    raw_category = payload.get("category_id")
    if isinstance(raw_category, int):
        if raw_category not in class_ids:
            class_ids = [raw_category, *class_ids]
        if primary is None:
            primary = raw_category
    class_ids = list(dict.fromkeys(class_ids))
    if primary is None and class_ids:
        primary = class_ids[0]
    return class_ids, primary


def _annotation_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("objects")
    if not isinstance(raw, list):
        return []
    return sorted([item for item in raw if isinstance(item, dict)], key=lambda item: str(item.get("id", "")))


def _has_three_unique_points(points: list[float]) -> bool:
    if len(points) < 6 or len(points) % 2 != 0:
        return False
    seen: set[tuple[float, float]] = set()
    for index in range(0, len(points), 2):
        seen.add((points[index], points[index + 1]))
    return len(seen) >= 3


def _status(status: str) -> str:
    if status == "approved":
        return "reviewed"
    if status in {"skipped", "unlabeled"}:
        return "skipped"
    return "labeled"


def _iso_utc(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _obj_uuid(annotation_id: str, raw_id: Any, index: int) -> str:
    value = str(raw_id) if raw_id is not None else f"object-{index}"
    try:
        return str(uuid.UUID(value))
    except (TypeError, ValueError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pixel-sheriff:{annotation_id}:{value}"))


def _task_contract(task_type: TaskType) -> tuple[dict[str, Any], str]:
    if task_type in {TaskType.classification, TaskType.classification_single}:
        return {
            "primary": "classification",
            "enabled": ["classification"],
            "multitask_policy": {
                "allow_image_and_objects": False,
                "image_labels_used_for_training": True,
                "object_labels_used_for_training": False,
            },
        }, "classification"
    if task_type == TaskType.bbox:
        return {
            "primary": "detection",
            "enabled": ["detection"],
            "multitask_policy": {
                "allow_image_and_objects": False,
                "image_labels_used_for_training": False,
                "object_labels_used_for_training": True,
            },
        }, "detection"
    return {
        "primary": "segmentation",
        "enabled": ["segmentation"],
        "multitask_policy": {
            "allow_image_and_objects": False,
            "image_labels_used_for_training": False,
            "object_labels_used_for_training": True,
        },
    }, "segmentation"


def build_export_result(
    *,
    project_id: str,
    project_name: str,
    task_type: TaskType,
    selection_criteria: dict[str, Any],
    categories: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    load_asset_bytes: Callable[[dict[str, Any]], bytes | None],
    tool_version: str = "0.1.0",
) -> tuple[dict[str, Any], dict[str, Any], str, bytes]:
    categories = sorted(categories, key=lambda item: (item.get("display_order", 0), item["id"]))
    assets = sorted(assets, key=lambda item: (str(item.get("relative_path", "")), item["id"]))
    annotations = sorted(annotations, key=lambda item: (item["asset_id"], item["id"]))

    if not categories:
        raise _err("export_classes_empty", "At least one class is required to export a dataset")

    classes = []
    class_order: list[int] = []
    class_names_seen: set[str] = set()
    for cat in categories:
        class_id = cat.get("id")
        if not isinstance(class_id, int):
            raise _err("export_class_invalid", "Class id must be an integer")
        display_name = str(cat.get("name") or "").strip() or f"class_{class_id}"
        normalized_name = _slugify(display_name) or f"class_{class_id}"
        if normalized_name in class_names_seen:
            raise _err(
                "export_class_name_collision",
                "Normalized class names must be unique",
                {"class_id": class_id, "name": normalized_name},
            )
        class_names_seen.add(normalized_name)
        aliases = [display_name] if display_name != normalized_name else []
        classes.append(
            {
                "id": class_id,
                "name": normalized_name,
                "display_name": display_name,
                "supercategory": None,
                "aliases": aliases,
            }
        )
        class_order.append(class_id)

    class_id_set = {item["id"] for item in classes}
    if len(class_order) != len(set(class_order)):
        raise _err("export_class_order_duplicate", "label_schema.class_order must not contain duplicates")
    if set(class_order) != class_id_set:
        raise _err("export_class_order_invalid", "label_schema.class_order must contain exactly exported class ids")
    class_rank = {class_id: i for i, class_id in enumerate(class_order)}

    asset_records: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    asset_data_by_zip_path: dict[str, bytes] = {}
    asset_by_id: dict[str, dict[str, Any]] = {}
    used_paths: set[str] = set()

    for asset in assets:
        asset_id = str(asset["id"])
        if asset_id in asset_by_id:
            raise _err("export_asset_duplicate", "asset.asset_id must be unique", {"asset_id": asset_id})

        fallback = f"{asset_id}{asset.get('extension') or ''}"
        rel = _safe_relative_path(str(asset.get("relative_path") or fallback), fallback)
        zip_path = f"assets/{rel}"
        n = 2
        while zip_path in used_paths:
            p = PurePosixPath(rel)
            stem = p.stem
            suffix = p.suffix
            parent = str(p.parent)
            renamed = f"{stem}_{n}{suffix}"
            rel2 = renamed if parent in ("", ".") else f"{parent}/{renamed}"
            zip_path = f"assets/{rel2}"
            n += 1
        used_paths.add(zip_path)

        content = load_asset_bytes(asset)
        if content is None:
            raise _err("export_asset_file_missing", "Asset file is missing and cannot be packaged", {"asset_id": asset_id})
        asset_data_by_zip_path[zip_path] = content

        width = int(asset.get("width") or 1)
        height = int(asset.get("height") or 1)
        if width <= 0 or height <= 0:
            raise _err("export_asset_dimensions_invalid", "Asset width and height must be positive", {"asset_id": asset_id})

        checksum = asset.get("checksum")
        sha256 = checksum if isinstance(checksum, str) and len(checksum) == 64 else None
        media_type = "video" if str(asset.get("type") or "image") == "video" else "image"

        item = {
            "asset_id": asset_id,
            "path": zip_path,
            "media_type": media_type,
            "width": width,
            "height": height,
            "hash": {"sha256": sha256},
            "meta": {
                "original_filename": asset.get("original_filename"),
                "captured_at": None,
                "source": None,
            },
            "coco": {"image_id": asset_id},
        }
        asset_records.append(item)
        asset_by_id[asset_id] = item
        coco_images.append({"id": asset_id, "file_name": zip_path, "width": width, "height": height})

    class_counts_image: dict[str, int] = {}
    class_counts_objects: dict[str, int] = {}
    annotation_records: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    coco_id = 1
    include_negative_images = _as_bool(selection_criteria.get("include_negative_images"), True)

    for ann in annotations:
        annotation_id = str(ann["id"])
        asset_id = str(ann["asset_id"])
        if asset_id not in asset_by_id:
            raise _err(
                "export_annotation_asset_missing",
                "Every annotations[].asset_id must exist in assets",
                {"annotation_id": annotation_id, "asset_id": asset_id},
            )

        payload = ann.get("payload") if isinstance(ann.get("payload"), dict) else {}
        class_ids, primary_class_id = _annotation_classes(payload)

        if task_type in {TaskType.classification, TaskType.classification_single}:
            for class_id in class_ids:
                if class_id not in class_id_set:
                    raise _err(
                        "export_class_id_invalid",
                        "Every class_id referenced must exist in label_schema.classes",
                        {"annotation_id": annotation_id, "class_id": class_id},
                    )
            ordered = sorted(dict.fromkeys(class_ids), key=lambda value: class_rank.get(value, 10**9))
            if not ordered:
                image_labels = {"mode": "none", "primary_class_id": None, "class_ids": [], "confidence": None}
            else:
                primary = primary_class_id if primary_class_id in ordered else ordered[0]
                if task_type == TaskType.classification_single:
                    image_labels = {"mode": "single", "primary_class_id": primary, "class_ids": [primary], "confidence": None}
                else:
                    image_labels = {"mode": "multi", "primary_class_id": primary, "class_ids": ordered, "confidence": None}
            for class_id in image_labels["class_ids"]:
                key = str(class_id)
                class_counts_image[key] = class_counts_image.get(key, 0) + 1
        else:
            image_labels = {"mode": "none", "primary_class_id": None, "class_ids": [], "confidence": None}

        raw_objects = _annotation_objects(payload)
        if task_type in {TaskType.classification, TaskType.classification_single} and raw_objects:
            raise _err("export_task_mode_mismatch", "Classification exports cannot include object labels", {"annotation_id": annotation_id})

        object_labels: list[dict[str, Any]] = []
        ann_coco_ids: list[int] = []
        width = int(asset_by_id[asset_id]["width"])
        height = int(asset_by_id[asset_id]["height"])
        basis = payload.get("image_basis")
        if isinstance(basis, dict):
            basis_width = basis.get("width")
            basis_height = basis.get("height")
            if isinstance(basis_width, int) and basis_width > 0 and isinstance(basis_height, int) and basis_height > 0:
                width = basis_width
                height = basis_height

        for idx, obj in enumerate(raw_objects):
            object_id = _obj_uuid(annotation_id, obj.get("id"), idx)
            class_id = obj.get("category_id")
            if not isinstance(class_id, int):
                raise _err("export_object_class_invalid", "Object class_id must be an integer", {"annotation_id": annotation_id, "object_id": object_id})
            if class_id not in class_id_set:
                raise _err(
                    "export_class_id_invalid",
                    "Every class_id referenced must exist in label_schema.classes",
                    {"annotation_id": annotation_id, "object_id": object_id, "class_id": class_id},
                )

            kind = obj.get("kind")
            if kind == "bbox":
                if task_type == TaskType.segmentation:
                    raise _err("export_task_mode_mismatch", "Segmentation exports cannot include bbox objects", {"annotation_id": annotation_id, "object_id": object_id})
                raw_bbox = obj.get("bbox")
                if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                    raise _err("export_bbox_invalid", "bbox_xywh is required and must have 4 values", {"annotation_id": annotation_id, "object_id": object_id})
                bbox = _numbers(raw_bbox)
                if len(bbox) != 4:
                    raise _err("export_bbox_invalid", "bbox_xywh is required and must have 4 numeric values", {"annotation_id": annotation_id, "object_id": object_id})
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    raise _err("export_bbox_invalid", "bbox_xywh width and height must be > 0", {"annotation_id": annotation_id, "object_id": object_id})
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    raise _err("export_bbox_out_of_bounds", "bbox_xywh must be within image bounds", {"annotation_id": annotation_id, "object_id": object_id})
                shape = {"type": "bbox", "bbox_xywh": bbox}
                coco_bbox = bbox
                coco_seg = []
                coco_area = w * h
            elif kind == "polygon":
                if task_type == TaskType.bbox:
                    raise _err("export_task_mode_mismatch", "Detection exports cannot include polygon objects", {"annotation_id": annotation_id, "object_id": object_id})
                raw_poly = obj.get("segmentation")
                if not isinstance(raw_poly, list) or not raw_poly:
                    raise _err("export_polygon_invalid", "polygon must include at least one segment", {"annotation_id": annotation_id, "object_id": object_id})
                polygon: list[list[float]] = []
                for seg in raw_poly:
                    if not isinstance(seg, list):
                        raise _err("export_polygon_invalid", "polygon segments must be arrays", {"annotation_id": annotation_id, "object_id": object_id})
                    numbers = _numbers(seg)
                    if len(numbers) < 6 or len(numbers) % 2 != 0:
                        raise _err("export_polygon_invalid", "polygon must include at least 3 points", {"annotation_id": annotation_id, "object_id": object_id})
                    if not _has_three_unique_points(numbers):
                        raise _err(
                            "export_polygon_invalid",
                            "polygon must contain at least 3 unique points",
                            {"annotation_id": annotation_id, "object_id": object_id},
                        )
                    for i in range(0, len(numbers), 2):
                        x = numbers[i]
                        y = numbers[i + 1]
                        if x < 0 or y < 0 or x > width or y > height:
                            raise _err("export_polygon_out_of_bounds", "polygon points must be within image bounds", {"annotation_id": annotation_id, "object_id": object_id})
                    polygon.append(numbers)
                shape = {"type": "polygon", "polygon": polygon}
                coco_bbox, coco_area = _poly_bbox_area(polygon)
                coco_seg = polygon
            else:
                raise _err("export_shape_invalid", "shape.type must be bbox or polygon", {"annotation_id": annotation_id, "object_id": object_id})

            if coco_area <= 0:
                raise _err("export_coco_area_invalid", "COCO object annotation area must be > 0", {"annotation_id": annotation_id, "object_id": object_id})
            if len(coco_bbox) != 4 or coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                raise _err("export_coco_bbox_invalid", "COCO object annotation bbox must be non-empty", {"annotation_id": annotation_id, "object_id": object_id})

            object_labels.append(
                {
                    "object_id": object_id,
                    "class_id": class_id,
                    "shape": shape,
                    "attributes": {},
                    "meta": {"occluded": None, "truncated": None},
                }
            )
            class_key = str(class_id)
            class_counts_objects[class_key] = class_counts_objects.get(class_key, 0) + 1

            coco_annotations.append(
                (
                    {
                        "id": coco_id,
                        "image_id": asset_id,
                        "category_id": class_id,
                        "bbox": coco_bbox,
                        "area": coco_area,
                        "iscrowd": 0,
                    }
                    if task_type == TaskType.bbox
                    else {
                        "id": coco_id,
                        "image_id": asset_id,
                        "category_id": class_id,
                        "bbox": coco_bbox,
                        "segmentation": coco_seg,
                        "area": coco_area,
                        "iscrowd": 0,
                    }
                )
            )
            ann_coco_ids.append(coco_id)
            coco_id += 1

        if task_type in {TaskType.bbox, TaskType.segmentation} and not include_negative_images and not object_labels:
            continue

        annotation_records.append(
            {
                "annotation_id": annotation_id,
                "asset_id": asset_id,
                "status": _status(str(ann.get("status") or "")),
                "created_at": _iso_utc(ann.get("created_at")),
                "updated_at": _iso_utc(ann.get("updated_at")),
                "annotator": {"id": None, "name": ann.get("annotated_by")},
                "labels": {"image": image_labels, "objects": object_labels},
                "exports": {"coco": {"image_id": asset_id, "annotation_ids": ann_coco_ids}},
            }
        )

    if task_type in {TaskType.bbox, TaskType.segmentation} and not include_negative_images:
        included_asset_ids = {item["asset_id"] for item in annotation_records}
        asset_records = [item for item in asset_records if item["asset_id"] in included_asset_ids]
        coco_images = [item for item in coco_images if item["id"] in included_asset_ids]
        asset_by_id = {item["asset_id"]: item for item in asset_records}
        valid_paths = {item["path"] for item in asset_records}
        asset_data_by_zip_path = {path: content for path, content in asset_data_by_zip_path.items() if path in valid_paths}

    coco_image_ids = {item["id"] for item in coco_images}
    manifest_asset_ids = {item["asset_id"] for item in asset_records}
    if coco_image_ids != manifest_asset_ids:
        raise _err("export_join_invalid", "COCO images and manifest assets must reference the same asset ids")

    coco_annotation_ids = {int(item["id"]) for item in coco_annotations}
    for annotation in annotation_records:
        coco_export = annotation.get("exports", {}).get("coco", {})
        image_id = coco_export.get("image_id")
        if image_id not in coco_image_ids:
            raise _err(
                "export_join_invalid",
                "Every manifest exports.coco.image_id must exist in COCO images",
                {"annotation_id": annotation["annotation_id"], "image_id": image_id},
            )
        for annotation_id in coco_export.get("annotation_ids", []):
            if int(annotation_id) not in coco_annotation_ids:
                raise _err(
                    "export_join_invalid",
                    "Every manifest exports.coco.annotation_id must exist in COCO annotations",
                    {"annotation_id": annotation["annotation_id"], "coco_annotation_id": annotation_id},
                )

    tasks, model_task = _task_contract(task_type)
    split_asset_ids = [asset["asset_id"] for asset in asset_records]
    split_asset_id_set = set(split_asset_ids)
    if split_asset_id_set - manifest_asset_ids:
        raise _err("export_split_invalid", "All split asset_ids must exist in assets")

    manifest = {
        "schema_version": "1.2",
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tool": {"name": "pixel-sheriff", "version": tool_version, "repo": None},
        "dataset": {"id": project_id, "name": project_name, "description": "", "license": "", "source": ""},
        "tasks": tasks,
        "label_schema": {
            "classes": classes,
            "class_order": class_order,
            "rules": {"names_normalized": "lowercase_slug", "id_stable": True, "order_stable": True},
        },
        "splits": {
            "train": {"asset_ids": split_asset_ids},
            "val": {"asset_ids": []},
            "test": {"asset_ids": []},
            "generation": {
                "method": "manual",
                "seed": selection_criteria.get("seed") if isinstance(selection_criteria.get("seed"), int) else None,
                "notes": f"include_negative_images={str(include_negative_images).lower()}",
            },
        },
        "assets": asset_records,
        "annotations": annotation_records,
        "training_defaults": {
            "input": {"recommended_size": [640, 640], "resize_policy": "letterbox", "interpolation": "bilinear"},
            "normalization": {"type": "imagenet", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "augmentation_profile": "none",
            "dataloader": {"batch_size": 16, "num_workers": 4, "shuffle": True},
            "model_hints": {
                "task": model_task,
                "num_classes": len(class_order),
                "class_order": class_order,
                "background_class": False,
            },
        },
        "stats": {
            "num_assets": len(asset_records),
            "num_labeled": sum(1 for item in annotation_records if item["status"] != "skipped"),
            "class_counts_image": class_counts_image,
            "class_counts_objects": class_counts_objects,
        },
    }

    if manifest["training_defaults"]["model_hints"]["num_classes"] != len(manifest["label_schema"]["class_order"]):
        raise _err("export_num_classes_invalid", "model_hints.num_classes must equal label_schema.class_order length")

    coco_payload = {
        "info": {
            "description": "Pixel Sheriff COCO export",
            "year": datetime.now(timezone.utc).year,
            "version": "pixel-sheriff-1.2",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": item["id"], "name": item["name"], "supercategory": "default"} for item in classes],
    }

    hash_manifest = dict(manifest)
    hash_manifest["exported_at"] = "stable"
    content_hash = stable_hash({"manifest": hash_manifest, "coco_instances": coco_payload})

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("coco_instances.json", json.dumps(coco_payload, indent=2, sort_keys=True))
        for zip_path in sorted(asset_data_by_zip_path.keys()):
            archive.writestr(zip_path, asset_data_by_zip_path[zip_path])

    return manifest, coco_payload, content_hash, buffer.getvalue()
