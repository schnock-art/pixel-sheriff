import json
from io import BytesIO
import math
from pathlib import PurePosixPath
from typing import Callable
import zipfile

from sheriff_api.services.hashing import stable_hash


def _safe_relative_path(value: str, fallback_filename: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    parts: list[str] = []
    for part in path.parts:
        if part in ("", ".", ".."):
            continue
        parts.append(part)
    if not parts:
        return fallback_filename
    return "/".join(parts)


def _annotation_category_ids(payload: dict) -> list[int]:
    classification = payload.get("classification")
    if isinstance(classification, dict):
        raw_ids = classification.get("category_ids")
        if isinstance(raw_ids, list):
            values = [value for value in raw_ids if isinstance(value, int)]
            if values:
                return values

        raw_primary = classification.get("primary_category_id")
        if isinstance(raw_primary, int):
            return [raw_primary]

    raw_ids = payload.get("category_ids")
    if isinstance(raw_ids, list):
        values = [value for value in raw_ids if isinstance(value, int)]
        if values:
            return values

    raw_id = payload.get("category_id")
    if isinstance(raw_id, int):
        return [raw_id]
    return []


def _annotation_objects(payload: dict) -> list[dict]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        return []
    objects = [item for item in raw_objects if isinstance(item, dict)]
    return sorted(objects, key=lambda item: str(item.get("id", "")))


def _polygon_bounds_and_area(segmentation: list[list[float]]) -> tuple[list[float], float]:
    x_values: list[float] = []
    y_values: list[float] = []
    total_area = 0.0

    for points in segmentation:
        if not points or len(points) < 6:
            continue
        local_x = points[0::2]
        local_y = points[1::2]
        if not local_x or not local_y:
            continue
        x_values.extend(local_x)
        y_values.extend(local_y)

        ring_area = 0.0
        point_count = len(local_x)
        for index in range(point_count):
            next_index = (index + 1) % point_count
            ring_area += local_x[index] * local_y[next_index] - local_x[next_index] * local_y[index]
        total_area += abs(ring_area) / 2.0

    if not x_values or not y_values:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    return [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)], total_area


def _number_list(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def build_manifest(
    project_id: str,
    selection_criteria: dict,
    categories: list[dict],
    assets: list[dict],
    annotations: list[dict],
    missing_asset_ids: list[str],
) -> dict:
    return {
        "project_id": project_id,
        "schema_version": "1.1.0",
        "selection_criteria_json": selection_criteria,
        "categories": categories,
        "assets": assets,
        "annotations": annotations,
        "missing_asset_ids": missing_asset_ids,
        "counts": {
            "categories": len(categories),
            "assets": len(assets),
            "annotations": len(annotations),
        },
    }


def build_export_result(
    project_id: str,
    selection_criteria: dict,
    categories: list[dict],
    assets: list[dict],
    annotations: list[dict],
    load_asset_bytes: Callable[[dict], bytes | None],
) -> tuple[dict, dict, str, bytes]:
    sorted_categories = sorted(categories, key=lambda item: (item.get("display_order", 0), item["id"]))
    sorted_assets = sorted(assets, key=lambda item: (str(item.get("relative_path", "")), item["id"]))
    sorted_annotations = sorted(annotations, key=lambda item: (item["asset_id"], item["id"]))

    image_records: list[dict] = []
    annotation_records: list[dict] = []
    category_records = [{"id": category["id"], "name": category["name"], "supercategory": "default"} for category in sorted_categories]
    image_id_by_asset_id: dict[str, int] = {}
    image_zip_path_by_asset_id: dict[str, str] = {}
    asset_binary_by_zip_path: dict[str, bytes] = {}
    missing_asset_ids: list[str] = []

    for index, asset in enumerate(sorted_assets, start=1):
        asset_id = asset["id"]
        fallback_filename = f"{asset_id}{asset.get('extension') or ''}"
        relative_path = _safe_relative_path(str(asset.get("relative_path") or fallback_filename), fallback_filename)
        zip_image_path = f"images/{relative_path}"

        duplicate_index = 2
        while zip_image_path in asset_binary_by_zip_path:
            stem = PurePosixPath(relative_path).stem
            suffix = PurePosixPath(relative_path).suffix
            parent = str(PurePosixPath(relative_path).parent)
            candidate = f"{stem}_{duplicate_index}{suffix}"
            relative_candidate = candidate if parent in ("", ".") else f"{parent}/{candidate}"
            zip_image_path = f"images/{relative_candidate}"
            duplicate_index += 1

        image_id_by_asset_id[asset_id] = index
        image_zip_path_by_asset_id[asset_id] = zip_image_path

        asset_bytes = load_asset_bytes(asset)
        if asset_bytes is None:
            missing_asset_ids.append(asset_id)
        else:
            asset_binary_by_zip_path[zip_image_path] = asset_bytes

        image_records.append(
            {
                "id": index,
                "file_name": zip_image_path.replace("images/", "", 1),
                "width": int(asset.get("width") or 0),
                "height": int(asset.get("height") or 0),
                "sheriff_asset_id": asset_id,
            }
        )

    annotation_id = 1
    for annotation in sorted_annotations:
        image_id = image_id_by_asset_id.get(annotation["asset_id"])
        if image_id is None:
            continue
        payload = annotation.get("payload", {})
        objects = _annotation_objects(payload)
        if not objects:
            category_ids = _annotation_category_ids(payload)
            for category_id in category_ids:
                annotation_records.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [],
                        "segmentation": [],
                        "area": 0,
                        "iscrowd": 0,
                        "sheriff_annotation_id": annotation["id"],
                        "status": annotation["status"],
                    }
                )
                annotation_id += 1

        for obj in objects:
            kind = obj.get("kind")
            category_id = obj.get("category_id")
            if not isinstance(category_id, int):
                continue

            if kind == "bbox":
                raw_bbox = obj.get("bbox")
                if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                    continue
                bbox = _number_list(raw_bbox)
                if len(bbox) != 4:
                    continue
                area = max(0.0, bbox[2]) * max(0.0, bbox[3])
                annotation_records.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": [],
                        "area": area,
                        "iscrowd": 0,
                        "sheriff_annotation_id": annotation["id"],
                        "sheriff_object_id": obj.get("id"),
                        "status": annotation["status"],
                    }
                )
                annotation_id += 1
                continue

            if kind == "polygon":
                raw_segmentation = obj.get("segmentation")
                if not isinstance(raw_segmentation, list):
                    continue
                segmentation: list[list[float]] = []
                for segment in raw_segmentation:
                    if not isinstance(segment, list):
                        continue
                    numbers = _number_list(segment)
                    if len(numbers) >= 6 and len(numbers) % 2 == 0:
                        segmentation.append(numbers)
                if not segmentation:
                    continue

                bbox, area = _polygon_bounds_and_area(segmentation)
                annotation_records.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "sheriff_annotation_id": annotation["id"],
                        "sheriff_object_id": obj.get("id"),
                        "status": annotation["status"],
                    }
                )
                annotation_id += 1

    coco_payload = {
        "info": {"description": "Pixel Sheriff COCO export", "version": "1.0"},
        "licenses": [],
        "images": image_records,
        "annotations": annotation_records,
        "categories": category_records,
    }

    manifest = build_manifest(
        project_id=project_id,
        selection_criteria=selection_criteria,
        categories=sorted_categories,
        assets=sorted_assets,
        annotations=sorted_annotations,
        missing_asset_ids=sorted(missing_asset_ids),
    )
    content_hash = stable_hash({"manifest": manifest, "coco": coco_payload})

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("annotations.json", json.dumps(coco_payload, indent=2, sort_keys=True))
        for zip_path in sorted(asset_binary_by_zip_path.keys()):
            archive.writestr(zip_path, asset_binary_by_zip_path[zip_path])

    return manifest, coco_payload, content_hash, buffer.getvalue()
