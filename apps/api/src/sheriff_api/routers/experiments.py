from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Asset, Category, DatasetVersion, Project
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import (
    ExperimentAnalyticsBest,
    ExperimentAnalyticsItem,
    ExperimentEvaluationResponse,
    ExperimentSampleItem,
    ExperimentSamplesResponse,
    ProjectExperimentActionResponse,
    ProjectExperimentAnalyticsResponse,
    ProjectExperimentCreate,
    ProjectExperimentListResponse,
    ProjectExperimentRecord,
    ProjectExperimentSummary,
    ProjectExperimentUpdate,
    TrainingConfigV0,
)
from sheriff_api.services.experiment_store import ExperimentStore
from sheriff_api.services.exporter_coco import ExportValidationError, build_export_result
from sheriff_api.services.model_store import ModelStore
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

router = APIRouter(tags=["experiments"])
settings = get_settings()
model_store = ModelStore(settings.storage_root)
experiment_store = ExperimentStore(settings.storage_root)
storage = LocalStorage(settings.storage_root)
train_queue = TrainQueue()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _normalize_task(raw_task: str) -> str:
    task = (raw_task or "").strip().lower()
    if task in {"classification", "classification_single"}:
        return "classification"
    if task in {"detection", "bbox"}:
        return "detection"
    if task == "segmentation":
        return "segmentation"
    return "classification"


def _default_training_config(*, model_id: str, dataset_version_id: str, task: str) -> dict[str, Any]:
    normalized_task = _normalize_task(task)
    return TrainingConfigV0(
        model_id=model_id,
        dataset_version_id=dataset_version_id,
        task=normalized_task,
    ).model_dump(mode="json")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
            continue
        merged[key] = value
    return merged


def _collect_config_issues(config: dict[str, Any]) -> list[dict[str, str]]:
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


def _as_status_filter(selection_criteria: dict) -> set[str] | None:
    statuses_raw = selection_criteria.get("statuses")
    if isinstance(statuses_raw, list):
        normalized = {str(value) for value in statuses_raw if str(value).strip()}
        return normalized if normalized else None

    status_raw = selection_criteria.get("status")
    if isinstance(status_raw, str) and status_raw.strip():
        return {status_raw}

    return None


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


def _as_experiment_summary(record: dict[str, Any]) -> ProjectExperimentSummary:
    return ProjectExperimentSummary.model_validate(record)


def _as_experiment_record(record: dict[str, Any]) -> ProjectExperimentRecord:
    return ProjectExperimentRecord.model_validate(record)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed == parsed else None
    return None


def _safe_int(value: Any) -> int | None:
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


def _series_row_value(row: dict[str, Any], key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    return _safe_float(row.get(key))


def _extract_experiment_config(config_json: dict[str, Any]) -> dict[str, Any]:
    optimizer = config_json.get("optimizer")
    optimizer_type = None
    optimizer_lr = None
    if isinstance(optimizer, dict):
        optimizer_type = str(optimizer.get("type") or "") or None
        optimizer_lr = _safe_float(optimizer.get("lr"))
    return {
        "optimizer": {"type": optimizer_type, "lr": optimizer_lr},
        "batch_size": _safe_int(config_json.get("batch_size")),
        "epochs": _safe_int(config_json.get("epochs")),
        "augmentation": config_json.get("augmentation_profile"),
    }


def _metric_objective_direction(metric_name: str | None) -> str:
    if isinstance(metric_name, str) and metric_name.endswith("loss"):
        return "min"
    return "max"


def _filter_predictions(
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
        true_idx = _safe_int(row.get("true_class_index"))
        pred_idx = _safe_int(row.get("pred_class_index"))
        confidence = _safe_float(row.get("confidence"))
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
        filtered.sort(key=lambda item: _safe_float(item.get("confidence")) or 0.0)
    else:
        filtered.sort(key=lambda item: _safe_float(item.get("confidence")) or 0.0, reverse=True)
    return filtered


def _as_sample_item(row: dict[str, Any]) -> ExperimentSampleItem:
    return ExperimentSampleItem(
        asset_id=str(row.get("asset_id") or ""),
        relative_path=str(row.get("relative_path") or ""),
        true_class_index=int(_safe_int(row.get("true_class_index")) or 0),
        pred_class_index=int(_safe_int(row.get("pred_class_index")) or 0),
        confidence=float(_safe_float(row.get("confidence")) or 0.0),
        margin=_safe_float(row.get("margin")),
    )


def _load_asset_bytes(asset: dict[str, Any]) -> bytes | None:
    storage_uri = asset.get("storage_uri")
    if not isinstance(storage_uri, str) or not storage_uri:
        return None
    try:
        path = storage.resolve(storage_uri)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()


async def _ensure_dataset_export_zip(
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

    categories = list((await db.execute(select(Category).where(Category.project_id == project.id))).scalars().all())
    all_assets = list((await db.execute(select(Asset).where(Asset.project_id == project.id))).scalars().all())
    all_annotations = list((await db.execute(select(Annotation).where(Annotation.project_id == project.id))).scalars().all())
    selection_criteria = (
        dataset_version.selection_criteria_json if isinstance(dataset_version.selection_criteria_json, dict) else {}
    )
    status_filter = _as_status_filter(selection_criteria)
    if status_filter is None:
        selected_annotations = all_annotations
        selected_assets = all_assets
    else:
        selected_annotations = [annotation for annotation in all_annotations if annotation.status.value in status_filter]
        selected_asset_ids = {annotation.asset_id for annotation in selected_annotations}
        selected_assets = [asset for asset in all_assets if asset.id in selected_asset_ids]

    storage.ensure_project_dirs(project.id)
    try:
        _manifest, _coco, rebuilt_hash, zip_bytes = build_export_result(
            project_id=project.id,
            project_name=project.name,
            task_type=project.task_type,
            selection_criteria=selection_criteria,
            categories=[
                {"id": c.id, "name": c.name, "display_order": c.display_order, "is_active": c.is_active}
                for c in categories
            ],
            assets=[
                {
                    "id": a.id,
                    "uri": a.uri,
                    "type": a.type.value,
                    "width": a.width,
                    "height": a.height,
                    "checksum": a.checksum,
                    "relative_path": a.metadata_json.get("relative_path"),
                    "original_filename": a.metadata_json.get("original_filename"),
                    "storage_uri": a.metadata_json.get("storage_uri"),
                    "extension": Path(str(a.metadata_json.get("storage_uri") or "")).suffix.lower(),
                }
                for a in selected_assets
            ],
            annotations=[
                {
                    "id": n.id,
                    "asset_id": n.asset_id,
                    "status": n.status.value,
                    "payload": n.payload_json,
                    "created_at": n.created_at,
                    "updated_at": n.updated_at,
                    "annotated_by": n.annotated_by,
                }
                for n in selected_annotations
            ],
            load_asset_bytes=lambda asset: _load_asset_bytes(asset),
        )
    except ExportValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    if rebuilt_hash != content_hash:
        raise api_error(
            status_code=409,
            code="dataset_export_hash_mismatch",
            message="Dataset export could not be rebuilt deterministically for this dataset version",
            details={"dataset_version_id": dataset_version.id, "expected_hash": content_hash, "actual_hash": rebuilt_hash},
        )
    storage.write_bytes(relpath, zip_bytes)
    return {
        "content_hash": content_hash,
        "zip_relpath": relpath,
        "dataset_version_id": dataset_version.id,
    }


@router.get("/projects/{project_id}/experiments", response_model=ProjectExperimentListResponse)
async def list_project_experiments(
    project_id: str,
    model_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentListResponse:
    await _require_project(db, project_id)
    records = experiment_store.list_by_project(project_id, model_id=model_id)
    return ProjectExperimentListResponse(items=[_as_experiment_summary(record) for record in records])


@router.post("/projects/{project_id}/experiments", response_model=ProjectExperimentRecord)
async def create_project_experiment(
    project_id: str,
    payload: ProjectExperimentCreate,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    project = await _require_project(db, project_id)
    model_record = model_store.get(project_id, payload.model_id)
    if model_record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": payload.model_id},
        )

    latest_dataset = (
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
    if latest_dataset is None:
        raise api_error(
            status_code=400,
            code="project_manifest_missing",
            message="Project has no exported dataset manifest yet. Export the dataset first.",
            details={"project_id": project_id},
        )

    model_config = model_record.get("config_json")
    task = project.task_type.value
    if isinstance(model_config, dict):
        source_dataset = model_config.get("source_dataset")
        if isinstance(source_dataset, dict) and isinstance(source_dataset.get("task"), str):
            task = source_dataset.get("task")

    default_config = _default_training_config(
        model_id=payload.model_id,
        dataset_version_id=latest_dataset.id,
        task=task,
    )
    overrides = payload.config_overrides if isinstance(payload.config_overrides, dict) else {}
    config_json = _deep_merge(default_config, overrides)

    issues = _collect_config_issues(config_json)
    if issues:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Experiment config validation failed",
            details={"issues": issues},
        )

    existing = experiment_store.list_by_project(project_id, model_id=payload.model_id)
    name = payload.name.strip() if isinstance(payload.name, str) and payload.name.strip() else f"training_run_{len(existing) + 1}"
    created = experiment_store.create(
        project_id=project_id,
        model_id=payload.model_id,
        name=name,
        config_json=config_json,
        status="draft",
    )
    return _as_experiment_record(created)


@router.get("/projects/{project_id}/experiments/analytics", response_model=ProjectExperimentAnalyticsResponse)
async def project_experiments_analytics(
    project_id: str,
    max_points: int = Query(default=200, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentAnalyticsResponse:
    await _require_project(db, project_id)
    records = experiment_store.list_by_project(project_id)
    available_series: set[str] = set()
    items: list[ExperimentAnalyticsItem] = []
    for row in records:
        experiment_id = str(row.get("id") or "")
        if not experiment_id:
            continue
        status = str(row.get("status") or "draft")
        if status not in {"draft", "queued", "running", "completed", "failed", "canceled"}:
            status = "draft"
        resolved_status: Literal["draft", "queued", "running", "completed", "failed", "canceled"] = status
        config_json = row.get("config_json") if isinstance(row.get("config_json"), dict) else {}
        resolved_attempt = row.get("last_completed_attempt")
        if not isinstance(resolved_attempt, int) or resolved_attempt < 1:
            current_attempt = row.get("current_run_attempt")
            if isinstance(current_attempt, int) and current_attempt >= 1:
                resolved_attempt = current_attempt
            else:
                resolved_attempt = None
        metrics_rows = experiment_store.read_metrics(
            project_id,
            experiment_id,
            limit=max_points,
            attempt=resolved_attempt,
        )

        valid_rows: list[dict[str, Any]] = []
        for metric_row in metrics_rows:
            if not isinstance(metric_row, dict):
                continue
            epoch = _safe_int(metric_row.get("epoch"))
            if epoch is None or epoch < 1:
                continue
            normalized_row = dict(metric_row)
            normalized_row["epoch"] = int(epoch)
            valid_rows.append(normalized_row)

        series: dict[str, Any] = {"epochs": [int(metric_row["epoch"]) for metric_row in valid_rows]}
        metric_keys: list[str] = []
        for metric_row in valid_rows:
            for key in metric_row.keys():
                if key in {"attempt", "created_at", "epoch"}:
                    continue
                if key not in metric_keys:
                    metric_keys.append(key)
        for key in metric_keys:
            values = [_series_row_value(metric_row, key) for metric_row in valid_rows]
            if any(value is not None for value in values):
                series[key] = values
                available_series.add(key)

        summary_json = row.get("summary_json") if isinstance(row.get("summary_json"), dict) else {}
        best_metric_name = summary_json.get("best_metric_name")
        best_metric_value = _safe_float(summary_json.get("best_metric_value"))
        best_epoch = _safe_int(summary_json.get("best_epoch"))
        if not isinstance(best_metric_name, str):
            best_metric_name = None
        if best_metric_name is None and "val_accuracy" in series:
            best_metric_name = "val_accuracy"
            objective = _metric_objective_direction(best_metric_name)
            candidates = [
                (epoch, value)
                for epoch, value in zip(series.get("epochs", []), series.get(best_metric_name, []))
                if isinstance(epoch, int) and isinstance(value, (int, float))
            ]
            if candidates:
                if objective == "min":
                    best_epoch, best_metric_value = min(candidates, key=lambda item: float(item[1]))
                else:
                    best_epoch, best_metric_value = max(candidates, key=lambda item: float(item[1]))

        final: dict[str, float | None] = {}
        if valid_rows:
            last_row = valid_rows[-1]
            for key in (
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_macro_precision",
                "val_macro_recall",
                "val_map",
                "val_iou",
            ):
                value = _safe_float(last_row.get(key))
                if value is not None:
                    final[key] = value

        model_id = str(row.get("model_id") or "")
        model_record = model_store.get(project_id, model_id) if model_id else None
        model_name = str(model_record.get("name")) if isinstance(model_record, dict) and model_record.get("name") else model_id
        updated_at = row.get("updated_at")
        if not isinstance(updated_at, str):
            updated_at = _utc_now_iso()

        items.append(
            ExperimentAnalyticsItem(
                experiment_id=experiment_id,
                name=str(row.get("name") or experiment_id),
                model_id=model_id,
                model_name=model_name,
                status=resolved_status,
                updated_at=updated_at,
                config=_extract_experiment_config(config_json),
                best=ExperimentAnalyticsBest(
                    metric_name=best_metric_name,
                    metric_value=best_metric_value,
                    epoch=best_epoch,
                ),
                final=final,
                series=series,
            )
        )

    return ProjectExperimentAnalyticsResponse(
        items=items,
        available_series=sorted(available_series),
    )


@router.get("/projects/{project_id}/experiments/{experiment_id}", response_model=ProjectExperimentRecord)
async def get_project_experiment(
    project_id: str,
    experiment_id: str,
    limit: int | None = None,
    attempt: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    await _require_project(db, project_id)
    record = experiment_store.get(project_id, experiment_id, metrics_limit=limit, attempt=attempt)
    if record is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
    )
    return _as_experiment_record(record)


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/evaluation",
    response_model=ExperimentEvaluationResponse,
)
async def get_project_experiment_evaluation(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentEvaluationResponse:
    await _require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    loaded = experiment_store.read_evaluation(project_id, experiment_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="evaluation_not_found",
            message="Evaluation not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt, payload = loaded
    response_payload = dict(payload)
    response_payload["attempt"] = attempt
    return ExperimentEvaluationResponse.model_validate(response_payload)


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/samples",
    response_model=ExperimentSamplesResponse,
)
async def get_project_experiment_samples(
    project_id: str,
    experiment_id: str,
    mode: str = Query(default="misclassified"),
    true_class_index: int | None = Query(default=None, ge=0),
    pred_class_index: int | None = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> ExperimentSamplesResponse:
    await _require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"misclassified", "lowest_confidence_correct", "highest_confidence_wrong"}:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Unsupported samples mode",
            details={"mode": mode},
        )
    resolved_mode: Literal["misclassified", "lowest_confidence_correct", "highest_confidence_wrong"] = normalized_mode

    loaded_predictions = experiment_store.read_predictions(project_id, experiment_id)
    if loaded_predictions is not None:
        attempt, rows, _meta = loaded_predictions
        filtered_rows = _filter_predictions(
            rows,
            mode=normalized_mode,
            true_class_index=true_class_index,
            pred_class_index=pred_class_index,
        )
        return ExperimentSamplesResponse(
            attempt=attempt,
            mode=resolved_mode,
            items=[_as_sample_item(row) for row in filtered_rows[:limit]],
        )

    loaded_evaluation = experiment_store.read_evaluation(project_id, experiment_id)
    if loaded_evaluation is None:
        raise api_error(
            status_code=404,
            code="evaluation_not_found",
            message="Evaluation not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt, evaluation_payload = loaded_evaluation
    sample_rows: list[dict[str, Any]] = []
    samples_block = evaluation_payload.get("samples")
    if isinstance(samples_block, dict):
        raw_items = samples_block.get(normalized_mode)
        if isinstance(raw_items, list):
            sample_rows = [row for row in raw_items if isinstance(row, dict)]
    filtered_rows = _filter_predictions(
        sample_rows,
        mode=normalized_mode,
        true_class_index=true_class_index,
        pred_class_index=pred_class_index,
    )
    message = None
    if not filtered_rows:
        message = "No matching samples found for this filter."
    return ExperimentSamplesResponse(
        attempt=attempt,
        mode=resolved_mode,
        items=[_as_sample_item(row) for row in filtered_rows[:limit]],
        message=message,
    )


@router.put("/projects/{project_id}/experiments/{experiment_id}", response_model=ProjectExperimentRecord)
async def update_project_experiment(
    project_id: str,
    experiment_id: str,
    payload: ProjectExperimentUpdate,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    await _require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    status = str(current.get("status", "draft"))
    updates_training_fields = payload.name is not None or payload.config_json is not None
    if updates_training_fields and status not in {"draft", "failed", "canceled"}:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment can only be edited in draft, failed, or canceled state",
            details={"experiment_id": experiment_id, "status": status},
        )

    if payload.config_json is not None:
        issues = _collect_config_issues(payload.config_json)
        if issues:
            raise api_error(
                status_code=422,
                code="validation_error",
                message="Experiment config validation failed",
                details={"issues": issues},
            )

    updated = experiment_store.update(
        project_id=project_id,
        experiment_id=experiment_id,
        name=payload.name.strip() if isinstance(payload.name, str) else None,
        config_json=payload.config_json,
        selected_checkpoint_kind=payload.selected_checkpoint_kind,
    )
    if updated is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return _as_experiment_record(updated)


@router.post("/projects/{project_id}/experiments/{experiment_id}/start", response_model=ProjectExperimentActionResponse)
async def start_project_experiment(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentActionResponse:
    project = await _require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    status = str(current.get("status", "draft"))
    if status not in {"draft", "failed", "canceled"}:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment can only be started from draft, failed, or canceled state",
            details={"experiment_id": experiment_id, "status": status},
        )

    model_id = str(current.get("model_id") or "")
    model_record = model_store.get(project_id, model_id)
    if model_record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": model_id},
        )

    config_json = current.get("config_json")
    if not isinstance(config_json, dict):
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Experiment config validation failed",
            details={"issues": [{"path": "config_json", "message": "Experiment config is required"}]},
        )
    issues = _collect_config_issues(config_json)
    if issues:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Experiment config validation failed",
            details={"issues": issues},
        )

    dataset_version_id = str(config_json.get("dataset_version_id") or "")
    dataset_version = (
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
    if dataset_version is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )

    dataset_export = await _ensure_dataset_export_zip(db=db, project=project, dataset_version=dataset_version)
    model_config = model_record.get("config_json")
    if not isinstance(model_config, dict):
        raise api_error(
            status_code=422,
            code="model_config_invalid",
            message="Model config is not available",
            details={"project_id": project_id, "model_id": model_id},
        )

    model_family = shared_architecture_family(model_config)
    task = str(config_json.get("task") or "classification")
    job_id = str(uuid.uuid4())

    initialized = experiment_store.init_run_attempt(
        project_id=project_id,
        experiment_id=experiment_id,
        job_id=job_id,
        dataset_export=dataset_export,
        task=task,
        model_family=model_family,
    )
    if initialized is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt = initialized.get("current_run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        raise api_error(
            status_code=500,
            code="experiment_attempt_init_failed",
            message="Failed to initialize experiment run attempt",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    experiment_store.append_event(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        event={"type": "status", "status": "queued", "attempt": attempt, "job_id": job_id, "ts": _utc_now_iso()},
    )

    job_payload = {
        "job_version": "1",
        "job_id": job_id,
        "job_type": "train",
        "attempt": attempt,
        "project_id": project_id,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "task": task,
        "model_config": model_config,
        "training_config": config_json,
        "dataset_export": dataset_export,
    }
    try:
        await train_queue.enqueue_train_job(job_payload)
    except Exception as exc:
        experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="failed", error=str(exc))
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            event={"type": "done", "status": "failed", "attempt": attempt, "ts": _utc_now_iso(), "error_code": "train_queue_unavailable"},
        )
        raise api_error(
            status_code=503,
            code="train_queue_unavailable",
            message="Training queue is unavailable",
            details={"project_id": project_id, "experiment_id": experiment_id},
        ) from exc

    return ProjectExperimentActionResponse(ok=True, status="queued", attempt=attempt, job_id=job_id)


@router.post("/projects/{project_id}/experiments/{experiment_id}/cancel", response_model=ProjectExperimentActionResponse)
async def cancel_project_experiment(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentActionResponse:
    await _require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    status = str(current.get("status", "draft"))
    attempt = current.get("current_run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment has no active run to cancel",
            details={"experiment_id": experiment_id, "status": status},
        )

    if status == "queued":
        experiment_store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=True)
        experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="canceled")
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            event={"type": "done", "status": "canceled", "attempt": attempt, "ts": _utc_now_iso()},
        )
        return ProjectExperimentActionResponse(ok=True, status="canceled", attempt=attempt)

    if status == "running":
        experiment_store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=True)
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            event={"type": "status", "status": "running", "attempt": attempt, "ts": _utc_now_iso()},
        )
        return ProjectExperimentActionResponse(ok=True, status="running", attempt=attempt)

    raise api_error(
        status_code=409,
        code="experiment_state_invalid",
        message="Only queued or running experiments can be canceled",
        details={"experiment_id": experiment_id, "status": status},
    )


@router.get("/projects/{project_id}/experiments/{experiment_id}/events")
async def stream_project_experiment_events(
    project_id: str,
    experiment_id: str,
    from_line: int = 0,
    attempt: int | None = None,
    follow: bool = True,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    await _require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    current_attempt = current.get("current_run_attempt")
    resolved_attempt = attempt if isinstance(attempt, int) and attempt >= 1 else current_attempt

    async def event_stream():
        if not isinstance(resolved_attempt, int) or resolved_attempt < 1:
            status = str(current.get("status", "draft"))
            yield _as_sse({"line": 0, "attempt": None, "event": {"type": "status", "status": status}})
            if status in {"completed", "failed", "canceled", "draft"}:
                yield _as_sse({"line": 0, "attempt": None, "event": {"type": "done", "status": status}})
            return

        cursor = max(0, int(from_line))
        done = False
        sent_snapshot = False
        while True:
            rows = experiment_store.read_events(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=resolved_attempt,
                from_line=cursor,
            )
            if rows:
                for row in rows:
                    cursor = int(row["line"])
                    event = row.get("event")
                    if isinstance(event, dict) and str(event.get("type")) == "done":
                        done = True
                    yield _as_sse(row)
                if done:
                    break
                if not follow:
                    break
                continue

            status_row = experiment_store.get_status_row(project_id, experiment_id)
            status = str(status_row.get("status", "draft"))
            line_count = experiment_store.get_event_line_count(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=resolved_attempt,
            )
            if not sent_snapshot:
                sent_snapshot = True
                yield _as_sse(
                    {
                        "line": cursor,
                        "attempt": resolved_attempt,
                        "event": {"type": "status", "status": status, "attempt": resolved_attempt},
                    }
                )
                if status in {"completed", "failed", "canceled", "draft"} and line_count <= cursor:
                    yield _as_sse(
                        {
                            "line": cursor,
                            "attempt": resolved_attempt,
                            "event": {"type": "done", "status": status, "attempt": resolved_attempt},
                        }
                    )
                    break
                if not follow:
                    break
                continue
            if status in {"completed", "failed", "canceled"} and line_count <= cursor:
                yield _as_sse(
                    {
                        "line": cursor,
                        "attempt": resolved_attempt,
                        "event": {"type": "done", "status": status, "attempt": resolved_attempt},
                    }
                )
                break

            if not follow:
                break
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.8)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
