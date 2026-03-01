from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import DatasetVersion, Project
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import (
    ProjectExperimentActionResponse,
    ProjectExperimentCreate,
    ProjectExperimentListResponse,
    ProjectExperimentRecord,
    ProjectExperimentSummary,
    ProjectExperimentUpdate,
    TrainingConfigV0,
)
from sheriff_api.services.experiment_runner import ExperimentRunnerManager
from sheriff_api.services.experiment_store import ExperimentStore
from sheriff_api.services.model_store import ModelStore

router = APIRouter(tags=["experiments"])
settings = get_settings()
model_store = ModelStore(settings.storage_root)
experiment_store = ExperimentStore(settings.storage_root)
runner_manager = ExperimentRunnerManager(experiment_store)


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


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


def _as_experiment_summary(record: dict[str, Any]) -> ProjectExperimentSummary:
    return ProjectExperimentSummary.model_validate(record)


def _as_experiment_record(record: dict[str, Any]) -> ProjectExperimentRecord:
    return ProjectExperimentRecord.model_validate(record)


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


@router.get("/projects/{project_id}/experiments/{experiment_id}", response_model=ProjectExperimentRecord)
async def get_project_experiment(
    project_id: str,
    experiment_id: str,
    limit: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    await _require_project(db, project_id)
    record = experiment_store.get(project_id, experiment_id, metrics_limit=limit)
    if record is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return _as_experiment_record(record)


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
    if updates_training_fields and status not in {"draft", "failed"}:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment can only be edited in draft or failed state",
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
    if status not in {"draft", "failed"}:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment can only be started from draft or failed state",
            details={"experiment_id": experiment_id, "status": status},
        )

    experiment_store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=False)
    experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="running")
    started = runner_manager.start(project_id, experiment_id)
    if not started:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment is already running",
            details={"experiment_id": experiment_id},
        )
    runner_manager.publish(project_id, experiment_id, {"type": "status", "status": "running"})
    return ProjectExperimentActionResponse(ok=True)


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
    if status != "running":
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Only running experiments can be canceled",
            details={"experiment_id": experiment_id, "status": status},
        )

    experiment_store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=True)
    experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="canceled")
    runner_manager.publish(project_id, experiment_id, {"type": "status", "status": "canceled"})
    return ProjectExperimentActionResponse(ok=True)


def _as_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


@router.get("/projects/{project_id}/experiments/{experiment_id}/events")
async def stream_project_experiment_events(
    project_id: str,
    experiment_id: str,
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

    async def event_stream():
        queue = runner_manager.subscribe(project_id, experiment_id)
        try:
            status = str(current.get("status", "draft"))
            yield _as_sse({"type": "status", "status": status})
            if status in {"completed", "failed", "canceled"}:
                yield _as_sse({"type": "done", "status": status})
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                yield _as_sse(event)
                if str(event.get("type")) == "done":
                    break
        finally:
            runner_manager.unsubscribe(project_id, experiment_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
