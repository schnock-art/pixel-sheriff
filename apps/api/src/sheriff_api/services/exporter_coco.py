import json
from io import BytesIO
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
    raw_ids = payload.get("category_ids")
    if isinstance(raw_ids, list):
        values = [value for value in raw_ids if isinstance(value, int)]
        if values:
            return values

    raw_id = payload.get("category_id")
    if isinstance(raw_id, int):
        return [raw_id]
    return []


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
        category_ids = _annotation_category_ids(annotation.get("payload", {}))
        for category_id in category_ids:
            annotation_records.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [],
                    "area": 0,
                    "iscrowd": 0,
                    "sheriff_annotation_id": annotation["id"],
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
