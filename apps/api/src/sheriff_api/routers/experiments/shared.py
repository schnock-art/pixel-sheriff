from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Asset, Project, Task, TaskKind, TaskLabelMode, TaskType
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import ExperimentSampleItem, ProjectExperimentRecord, ProjectExperimentSummary, TrainingConfigV0
from sheriff_api.services.dataset_store import DatasetStore
from sheriff_api.services.experiment_store import ExperimentStore
from sheriff_api.services.exporter_coco import ExportValidationError, build_export_result
from sheriff_api.services.model_store import ProjectModelStore, create_project_model_store
from sheriff_api.services.storage import LocalStorage
from sheriff_api.services.train_queue import TrainQueue

try:
    from pixel_sheriff_ml.model_factory import architecture_family as shared_architecture_family
except Exception:

    def shared_architecture_family(model_config: dict[str, Any]) -> str:
        architecture = model_config.get("architecture")
        if not isinstance(architecture, dict):
            return "unknown"
        family = architecture.get("family")
        return str(family).strip().lower() if family is not None else "unknown"


settings = get_settings()
model_store: ProjectModelStore = create_project_model_store(settings.storage_root)
dataset_store = DatasetStore(settings.storage_root)
experiment_store = ExperimentStore(settings.storage_root)
storage = LocalStorage(settings.storage_root)
train_queue = TrainQueue()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def as_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def normalize_task(raw_task: str) -> str:
    task = (raw_task or "").strip().lower()
    if task in {"classification", "classification_single"}:
        return "classification"
    if task in {"detection", "bbox"}:
        return "detection"
    if task == "segmentation":
        return "segmentation"
    return "classification"


def default_training_config(*, model_id: str, dataset_version_id: str, task_id: str | None, task: str) -> dict[str, Any]:
    normalized_task = normalize_task(task)
    return TrainingConfigV0(
        model_id=model_id,
        dataset_version_id=dataset_version_id,
        task_id=task_id,
        task=normalized_task,
    ).model_dump(mode="json")


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
            continue
        merged[key] = value
    return merged


def collect_config_issues(config: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not isinstance(config, dict):
        return [{"path": "$", "message": "config_json must be an object"}]

    try:
        TrainingConfigV0.model_validate(config)
    except Exception as exc:
        issues.append({"path": "$", "message": str(exc)})
        return issues

    optimizer = config.get("optimizer")
    if isinstance(optimizer, dict):
        lr = optimizer.get("lr")
        if not isinstance(lr, (int, float)) or lr <= 0:
            issues.append({"path": "optimizer.lr", "message": "learning rate must be > 0"})
    else:
        issues.append({"path": "optimizer", "message": "optimizer is required"})

    epochs = config.get("epochs")
    if not isinstance(epochs, int) or epochs < 1:
        issues.append({"path": "epochs", "message": "epochs must be >= 1"})

    batch_size = config.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        issues.append({"path": "batch_size", "message": "batch_size must be >= 1"})

    return issues


async def require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


def as_experiment_summary(record: dict[str, Any]) -> ProjectExperimentSummary:
    return ProjectExperimentSummary.model_validate(record)


def as_experiment_record(record: dict[str, Any]) -> ProjectExperimentRecord:
    return ProjectExperimentRecord.model_validate(record)


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed == parsed else None
    return None


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def series_row_value(row: dict[str, Any], key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    return safe_float(row.get(key))


def extract_experiment_config(config_json: dict[str, Any]) -> dict[str, Any]:
    optimizer = config_json.get("optimizer")
    optimizer_type = None
    optimizer_lr = None
    if isinstance(optimizer, dict):
        optimizer_type = str(optimizer.get("type") or "") or None
        optimizer_lr = safe_float(optimizer.get("lr"))
    return {
        "optimizer": {"type": optimizer_type, "lr": optimizer_lr},
        "batch_size": safe_int(config_json.get("batch_size")),
        "epochs": safe_int(config_json.get("epochs")),
        "augmentation": config_json.get("augmentation_profile"),
    }


def metric_objective_direction(metric_name: str | None) -> str:
    if isinstance(metric_name, str) and metric_name.endswith("loss"):
        return "min"
    return "max"


def _task_type_for_task(task: Task) -> TaskType:
    if task.kind == TaskKind.classification:
        if task.label_mode == TaskLabelMode.multi_label:
            return TaskType.classification
        return TaskType.classification_single
    if task.kind == TaskKind.bbox:
        return TaskType.bbox
    return TaskType.segmentation


def filter_predictions(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    true_class_index: int | None,
    pred_class_index: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        true_idx = safe_int(row.get("true_class_index"))
        pred_idx = safe_int(row.get("pred_class_index"))
        confidence = safe_float(row.get("confidence"))
        if true_idx is None or pred_idx is None or confidence is None:
            continue
        if mode == "misclassified" and true_idx == pred_idx:
            continue
        if mode == "lowest_confidence_correct" and true_idx != pred_idx:
            continue
        if mode == "highest_confidence_wrong" and true_idx == pred_idx:
            continue
        if isinstance(true_class_index, int) and true_idx != true_class_index:
            continue
        if isinstance(pred_class_index, int) and pred_idx != pred_class_index:
            continue
        filtered.append(row)

    if mode == "lowest_confidence_correct":
        filtered.sort(key=lambda item: safe_float(item.get("confidence")) or 0.0)
    else:
        filtered.sort(key=lambda item: safe_float(item.get("confidence")) or 0.0, reverse=True)
    return filtered


def as_sample_item(row: dict[str, Any]) -> ExperimentSampleItem:
    return ExperimentSampleItem(
        asset_id=str(row.get("asset_id") or ""),
        relative_path=str(row.get("relative_path") or ""),
        true_class_index=int(safe_int(row.get("true_class_index")) or 0),
        pred_class_index=int(safe_int(row.get("pred_class_index")) or 0),
        confidence=float(safe_float(row.get("confidence")) or 0.0),
        margin=safe_float(row.get("margin")),
    )


def _map_uuid_payload_to_coco_int(payload_json: dict[str, Any], class_to_coco: dict[str, int]) -> dict[str, Any]:
    payload = dict(payload_json)

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


def _load_asset_bytes(local_storage: LocalStorage, asset: dict[str, Any]) -> bytes | None:
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


async def ensure_dataset_export_zip(
    *,
    db: AsyncSession,
    project: Project,
    dataset_version: dict[str, Any],
) -> dict[str, Any]:
    dataset_version_id = str(dataset_version.get("dataset_version_id") or "")
    if not dataset_version_id:
        raise api_error(
            status_code=422,
            code="dataset_version_not_found",
            message="Dataset version payload is invalid",
            details={"project_id": project.id},
        )

    existing_artifact = dataset_store.get_export_artifact(project.id, dataset_version_id)
    if isinstance(existing_artifact, dict) and isinstance(existing_artifact.get("hash"), str):
        content_hash = existing_artifact["hash"]
        relpath = f"exports/{project.id}/{content_hash}.zip"
        if storage.resolve(relpath).exists():
            return {
                "content_hash": content_hash,
                "zip_relpath": relpath,
                "dataset_version_id": dataset_version_id,
            }

    task_id = str(dataset_version.get("task_id") or "")
    if not task_id:
        raise api_error(
            status_code=422,
            code="dataset_version_invalid",
            message="Dataset version payload is missing task_id",
            details={"project_id": project.id, "dataset_version_id": dataset_version_id},
        )
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project.id:
        raise api_error(
            status_code=422,
            code="task_not_found",
            message="Dataset task not found in project",
            details={"project_id": project.id, "dataset_version_id": dataset_version_id, "task_id": task_id},
        )

    class_order = dataset_version.get("labels", {}).get("label_schema", {}).get("class_order")
    classes = dataset_version.get("labels", {}).get("label_schema", {}).get("classes")
    if not isinstance(class_order, list) or not isinstance(classes, list):
        raise api_error(
            status_code=422,
            code="dataset_split_invalid",
            message="Dataset label schema is invalid",
            details={"project_id": project.id, "dataset_version_id": dataset_version_id},
        )

    class_to_coco: dict[str, int] = {}
    class_name_by_id: dict[str, str] = {}
    for row in classes:
        if not isinstance(row, dict):
            continue
        category_id = row.get("category_id")
        name = row.get("name")
        if isinstance(category_id, str) and isinstance(name, str):
            class_name_by_id[category_id] = name

    categories_for_export: list[dict[str, Any]] = []
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

    selected_asset_ids = dataset_version.get("assets", {}).get("asset_ids")
    if not isinstance(selected_asset_ids, list):
        selected_asset_ids = []
    selected_asset_id_set = {str(item) for item in selected_asset_ids}

    assets = list((await db.execute(select(Asset).where(Asset.project_id == project.id))).scalars().all())
    annotations = list(
        (await db.execute(select(Annotation).where(Annotation.project_id == project.id, Annotation.task_id == task_id))).scalars().all()
    )
    selected_assets = [asset for asset in assets if asset.id in selected_asset_id_set]
    selected_annotations = [annotation for annotation in annotations if annotation.asset_id in selected_asset_id_set]

    selection_criteria = dataset_version.get("selection", {}).get("filters", {})
    if not isinstance(selection_criteria, dict):
        selection_criteria = {}
    split_items = dataset_version.get("splits", {}).get("items")
    split_by_asset_id: dict[str, str] = {}
    if isinstance(split_items, list):
        for row in split_items:
            if not isinstance(row, dict):
                continue
            asset_id = row.get("asset_id")
            split_name = row.get("split")
            if isinstance(asset_id, str) and isinstance(split_name, str):
                split_by_asset_id[asset_id] = split_name

    try:
        _manifest, _coco, content_hash, zip_bytes = build_export_result(
            project_id=project.id,
            project_name=project.name,
            task_type=_task_type_for_task(task),
            selection_criteria=selection_criteria,
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
                    "payload": _map_uuid_payload_to_coco_int(
                        annotation.payload_json if isinstance(annotation.payload_json, dict) else {},
                        class_to_coco,
                    ),
                    "created_at": annotation.created_at,
                    "updated_at": annotation.updated_at,
                    "annotated_by": annotation.annotated_by,
                }
                for annotation in selected_annotations
            ],
            load_asset_bytes=lambda asset: _load_asset_bytes(storage, asset),
            split_by_asset_id=split_by_asset_id,
        )
    except ExportValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    relpath = f"exports/{project.id}/{content_hash}.zip"
    storage.write_bytes(relpath, zip_bytes)
    dataset_store.set_export_artifact(
        project.id,
        dataset_version_id,
        {
            "hash": content_hash,
            "export_uri": f"/api/v1/projects/{project.id}/datasets/versions/{dataset_version_id}/export/download",
        },
    )
    return {
        "content_hash": content_hash,
        "zip_relpath": relpath,
        "dataset_version_id": dataset_version_id,
    }


async def latest_dataset_version(_db: AsyncSession, project_id: str) -> dict[str, Any] | None:
    listed = dataset_store.list_versions(project_id)
    active_dataset_version_id = listed.get("active_dataset_version_id")
    if isinstance(active_dataset_version_id, str):
        loaded = dataset_store.get_version(project_id, active_dataset_version_id)
        if loaded is not None:
            return loaded["version"]

    items = listed.get("items")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict) and isinstance(first.get("version"), dict):
            return first["version"]
    return None


async def get_dataset_version(_db: AsyncSession, project_id: str, dataset_version_id: str) -> dict[str, Any] | None:
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        return None
    return loaded["version"]
