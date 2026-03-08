from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Project, Task, TaskKind, TaskLabelMode, TaskType
from sheriff_api.errors import api_error
from sheriff_api.services.exporter_coco import ExportValidationError, build_export_result
from sheriff_api.services.storage import LocalStorage


def task_type_for_task(task: Task) -> TaskType:
    if task.kind == TaskKind.classification:
        if task.label_mode == TaskLabelMode.multi_label:
            return TaskType.classification
        return TaskType.classification_single
    if task.kind == TaskKind.bbox:
        return TaskType.bbox
    return TaskType.segmentation


def map_uuid_payload_to_coco_int(payload_json: dict[str, Any], class_to_coco: dict[str, int]) -> dict[str, Any]:
    payload = copy.deepcopy(payload_json)

    def map_one(value: Any) -> Any:
        if isinstance(value, str) and value in class_to_coco:
            return class_to_coco[value]
        return value

    def map_many(values: Any) -> Any:
        if not isinstance(values, list):
            return values
        mapped: list[int] = []
        for value in values:
            converted = map_one(value)
            if isinstance(converted, int):
                mapped.append(converted)
        return mapped

    payload["category_id"] = map_one(payload.get("category_id"))
    payload["category_ids"] = map_many(payload.get("category_ids"))

    classification = payload.get("classification")
    if isinstance(classification, dict):
        classification["primary_category_id"] = map_one(classification.get("primary_category_id"))
        classification["category_ids"] = map_many(classification.get("category_ids"))

    coco = payload.get("coco")
    if isinstance(coco, dict):
        coco["category_id"] = map_one(coco.get("category_id"))

    objects = payload.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            item["category_id"] = map_one(item.get("category_id"))
    return payload


def load_asset_bytes(local_storage: LocalStorage, asset: dict[str, Any]) -> bytes | None:
    storage_uri = asset.get("storage_uri")
    if not isinstance(storage_uri, str) or not storage_uri:
        return None
    try:
        path = local_storage.resolve(storage_uri)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()


async def build_dataset_export(
    *,
    db: AsyncSession,
    storage: LocalStorage,
    project: Project,
    task: Task,
    dataset_version: dict[str, Any],
) -> tuple[str, str]:
    class_order = dataset_version.get("labels", {}).get("label_schema", {}).get("class_order")
    classes = dataset_version.get("labels", {}).get("label_schema", {}).get("classes")
    if not isinstance(class_order, list) or not isinstance(classes, list):
        raise api_error(
            status_code=422,
            code="dataset_split_invalid",
            message="Dataset version label snapshot is invalid",
            details={"dataset_version_id": dataset_version.get("dataset_version_id")},
        )

    class_to_coco: dict[str, int] = {}
    categories_for_export: list[dict[str, Any]] = []
    class_name_by_id: dict[str, str] = {}
    for class_row in classes:
        if not isinstance(class_row, dict):
            continue
        category_id = class_row.get("category_id")
        name = class_row.get("name")
        if isinstance(category_id, str) and isinstance(name, str):
            class_name_by_id[category_id] = name
    for index, category_id in enumerate(class_order):
        if not isinstance(category_id, str):
            continue
        class_to_coco[category_id] = index + 1
        categories_for_export.append(
            {
                "id": index + 1,
                "stable_id": category_id,
                "name": class_name_by_id.get(category_id, f"class_{index + 1}"),
                "display_order": index,
                "is_active": True,
            }
        )

    asset_ids = dataset_version.get("assets", {}).get("asset_ids")
    if not isinstance(asset_ids, list):
        asset_ids = []
    selected_asset_ids = {str(asset_id) for asset_id in asset_ids}

    assets = list((await db.execute(select(Asset).where(Asset.project_id == project.id))).scalars().all())
    annotations = list(
        (await db.execute(select(Annotation).where(Annotation.project_id == project.id, Annotation.task_id == task.id))).scalars().all()
    )
    selected_assets = [asset for asset in assets if asset.id in selected_asset_ids]
    selected_annotations = [annotation for annotation in annotations if annotation.asset_id in selected_asset_ids]

    try:
        manifest, _coco, content_hash, zip_bytes = build_export_result(
            project_id=project.id,
            project_name=project.name,
            task_type=task_type_for_task(task),
            selection_criteria=dataset_version.get("selection", {}).get("filters", {}),
            categories=categories_for_export,
            assets=[
                {
                    "id": asset.id,
                    "uri": asset.uri,
                    "type": asset.type.value,
                    "width": asset.width,
                    "height": asset.height,
                    "checksum": asset.checksum,
                    "relative_path": (asset.metadata_json or {}).get("relative_path")
                    if isinstance(asset.metadata_json, dict)
                    else None,
                    "original_filename": (asset.metadata_json or {}).get("original_filename")
                    if isinstance(asset.metadata_json, dict)
                    else None,
                    "storage_uri": (asset.metadata_json or {}).get("storage_uri")
                    if isinstance(asset.metadata_json, dict)
                    else None,
                    "extension": Path(str((asset.metadata_json or {}).get("storage_uri", ""))).suffix.lower()
                    if isinstance(asset.metadata_json, dict)
                    else "",
                }
                for asset in selected_assets
            ],
            annotations=[
                {
                    "id": annotation.id,
                    "asset_id": annotation.asset_id,
                    "status": annotation.status.value,
                    "payload": map_uuid_payload_to_coco_int(
                        annotation.payload_json if isinstance(annotation.payload_json, dict) else {},
                        class_to_coco,
                    ),
                    "created_at": annotation.created_at,
                    "updated_at": annotation.updated_at,
                    "annotated_by": annotation.annotated_by,
                }
                for annotation in selected_annotations
            ],
            load_asset_bytes=lambda asset: load_asset_bytes(storage, asset),
        )
    except ExportValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    storage_uri = f"exports/{project.id}/{content_hash}.zip"
    if not storage.resolve(storage_uri).exists():
        storage.write_bytes(storage_uri, zip_bytes)
    export_uri = f"/api/v1/projects/{project.id}/datasets/versions/{dataset_version['dataset_version_id']}/export/download"
    return content_hash, export_uri
