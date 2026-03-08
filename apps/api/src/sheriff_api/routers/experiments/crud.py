from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import (
    ProjectExperimentCreate,
    ProjectExperimentListResponse,
    ProjectExperimentRecord,
    ProjectExperimentUpdate,
)

from .shared import (
    as_experiment_record,
    as_experiment_summary,
    collect_config_issues,
    deep_merge,
    deployment_store,
    default_training_config,
    ensure_model_matches_dataset_version,
    experiment_store,
    get_dataset_version,
    latest_dataset_version,
    model_source_manifest_id,
    model_store,
    require_project,
)

router = APIRouter()


@router.get("/projects/{project_id}/experiments", response_model=ProjectExperimentListResponse)
async def list_project_experiments(
    project_id: str,
    model_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentListResponse:
    await require_project(db, project_id)
    records = experiment_store.list_by_project(project_id, model_id=model_id)
    return ProjectExperimentListResponse(items=[as_experiment_summary(record) for record in records])


@router.post("/projects/{project_id}/experiments", response_model=ProjectExperimentRecord)
async def create_project_experiment(
    project_id: str,
    payload: ProjectExperimentCreate,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    await require_project(db, project_id)
    model_record = model_store.get(project_id, payload.model_id)
    if model_record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": payload.model_id},
        )

    model_config = model_record.get("config_json")
    source_manifest_id = model_source_manifest_id(model_config)

    selected_dataset_version_id = payload.dataset_version_id if isinstance(payload.dataset_version_id, str) else source_manifest_id
    if selected_dataset_version_id:
        selected_dataset = await get_dataset_version(db, project_id, selected_dataset_version_id)
        if selected_dataset is None:
            raise api_error(
                status_code=404,
                code="dataset_version_not_found",
                message="Dataset version not found in project",
                details={"project_id": project_id, "dataset_version_id": selected_dataset_version_id},
            )
    else:
        selected_dataset = await latest_dataset_version(db, project_id)
    if selected_dataset is None:
        raise api_error(
            status_code=400,
            code="project_manifest_missing",
            message="Project has no dataset version yet. Create and activate a dataset version first.",
            details={"project_id": project_id},
        )

    dataset_task_id = str(selected_dataset.get("task_id") or "")
    if not dataset_task_id:
        raise api_error(
            status_code=422,
            code="dataset_version_invalid",
            message="Dataset version is missing task_id",
            details={"project_id": project_id, "dataset_version_id": selected_dataset.get("dataset_version_id")},
        )

    model_task_id = str(model_record.get("task_id") or "")
    if not model_task_id and isinstance(model_config, dict):
        source_dataset = model_config.get("source_dataset")
        if isinstance(source_dataset, dict):
            model_task_id = str(source_dataset.get("task_id") or "")
    if model_task_id and model_task_id != dataset_task_id:
        raise api_error(
            status_code=409,
            code="task_mismatch",
            message="Model and dataset tasks are incompatible",
            details={"project_id": project_id, "model_task_id": model_task_id, "dataset_task_id": dataset_task_id},
        )
    if isinstance(model_config, dict):
        ensure_model_matches_dataset_version(
            project_id=project_id,
            model_id=payload.model_id,
            model_config=model_config,
            dataset_version=selected_dataset,
        )

    task = str(selected_dataset.get("task") or "classification")
    if isinstance(model_config, dict):
        source_dataset = model_config.get("source_dataset")
        if isinstance(source_dataset, dict) and isinstance(source_dataset.get("task"), str):
            task = source_dataset.get("task")

    default_config = default_training_config(
        model_id=payload.model_id,
        dataset_version_id=str(selected_dataset.get("dataset_version_id")),
        task_id=dataset_task_id,
        task=task,
    )
    overrides = payload.config_overrides if isinstance(payload.config_overrides, dict) else {}
    config_json = deep_merge(default_config, overrides)
    config_json["task_id"] = dataset_task_id

    issues = collect_config_issues(config_json)
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
        task_id=dataset_task_id,
        name=name,
        config_json=config_json,
        status="draft",
    )
    return as_experiment_record(created)


@router.get("/projects/{project_id}/experiments/{experiment_id}", response_model=ProjectExperimentRecord)
async def get_project_experiment(
    project_id: str,
    experiment_id: str,
    limit: int | None = None,
    attempt: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    await require_project(db, project_id)
    record = experiment_store.get(project_id, experiment_id, metrics_limit=limit, attempt=attempt)
    if record is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return as_experiment_record(record)


@router.put("/projects/{project_id}/experiments/{experiment_id}", response_model=ProjectExperimentRecord)
async def update_project_experiment(
    project_id: str,
    experiment_id: str,
    payload: ProjectExperimentUpdate,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentRecord:
    await require_project(db, project_id)
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
        issues = collect_config_issues(payload.config_json)
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
    return as_experiment_record(updated)


@router.delete("/projects/{project_id}/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_experiment(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    status_value = str(current.get("status", "draft"))
    if status_value in {"queued", "running"}:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Queued or running experiments must be canceled before deletion",
            details={"experiment_id": experiment_id, "status": status_value},
        )

    blockers = deployment_store.list_referencing_experiment(project_id, experiment_id)
    if blockers:
        raise api_error(
            status_code=409,
            code="experiment_in_use",
            message="Experiment is referenced by active deployments",
            details={
                "experiment_id": experiment_id,
                "deployments": [
                    {
                        "deployment_id": str(item.get("deployment_id") or ""),
                        "name": str(item.get("name") or ""),
                        "status": str(item.get("status") or ""),
                    }
                    for item in blockers
                ],
            },
        )

    deleted = experiment_store.delete(project_id=project_id, experiment_id=experiment_id)
    if not deleted:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
