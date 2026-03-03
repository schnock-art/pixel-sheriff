from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import DatasetVersion, Project
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import ExperimentSampleItem, ProjectExperimentRecord, ProjectExperimentSummary, TrainingConfigV0
from sheriff_api.services.dataset_export_pipeline import build_export_bundle, prepare_export_inputs
from sheriff_api.services.experiment_store import ExperimentStore
from sheriff_api.services.exporter_coco import ExportValidationError
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


def default_training_config(*, model_id: str, dataset_version_id: str, task: str) -> dict[str, Any]:
    normalized_task = normalize_task(task)
    return TrainingConfigV0(
        model_id=model_id,
        dataset_version_id=dataset_version_id,
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


async def ensure_dataset_export_zip(
    *,
    db: AsyncSession,
    project: Project,
    dataset_version: DatasetVersion,
) -> dict[str, Any]:
    content_hash = str(dataset_version.hash)
    relpath = f"exports/{project.id}/{content_hash}.zip"
    path = storage.resolve(relpath)
    if path.exists():
        return {
            "content_hash": content_hash,
            "zip_relpath": relpath,
            "dataset_version_id": dataset_version.id,
        }

    selection_criteria = (
        dataset_version.selection_criteria_json if isinstance(dataset_version.selection_criteria_json, dict) else {}
    )
    export_inputs = await prepare_export_inputs(
        db=db,
        project_id=project.id,
        selection_criteria=selection_criteria,
    )
    try:
        rebuilt = build_export_bundle(
            project=project,
            selection_criteria=selection_criteria,
            inputs=export_inputs,
            storage=storage,
        )
    except ExportValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    if rebuilt.content_hash != content_hash:
        raise api_error(
            status_code=409,
            code="dataset_export_hash_mismatch",
            message="Dataset export could not be rebuilt deterministically for this dataset version",
            details={
                "dataset_version_id": dataset_version.id,
                "expected_hash": content_hash,
                "actual_hash": rebuilt.content_hash,
            },
        )
    storage.write_bytes(relpath, rebuilt.zip_bytes)
    return {
        "content_hash": content_hash,
        "zip_relpath": relpath,
        "dataset_version_id": dataset_version.id,
    }


async def latest_dataset_version(db: AsyncSession, project_id: str) -> DatasetVersion | None:
    return (
        (
            await db.execute(
                select(DatasetVersion)
                .where(DatasetVersion.project_id == project_id)
                .order_by(DatasetVersion.created_at.desc()),
            )
        )
        .scalars()
        .first()
    )


async def get_dataset_version(db: AsyncSession, project_id: str, dataset_version_id: str) -> DatasetVersion | None:
    return (
        (
            await db.execute(
                select(DatasetVersion).where(
                    DatasetVersion.id == dataset_version_id,
                    DatasetVersion.project_id == project_id,
                )
            )
        )
        .scalars()
        .first()
    )
