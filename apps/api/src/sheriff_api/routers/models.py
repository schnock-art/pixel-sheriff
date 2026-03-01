from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import DatasetVersion, Model, Project, Suggestion
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.models import (
    ModelCreate,
    ModelRead,
    ProjectModelCreate,
    ProjectModelCreateResponse,
    ProjectModelRecord,
    ProjectModelSummary,
    ProjectModelUpdate,
)
from sheriff_api.services.model_config_factory import (
    ManifestConfigError,
    ModelConfigValidationError,
    build_default_model_config,
    collect_model_config_issues,
    validate_model_config,
)
from sheriff_api.services.model_store import ModelStore

router = APIRouter(tags=["models"])
settings = get_settings()
model_store = ModelStore(settings.storage_root)


@router.post("/models", response_model=ModelRead)
async def create_model(payload: ModelCreate, db: AsyncSession = Depends(get_db)) -> Model:
    model = Model(**payload.model_dump())
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


@router.get("/models", response_model=list[ModelRead])
async def list_models(db: AsyncSession = Depends(get_db)) -> list[Model]:
    result = await db.execute(select(Model))
    return list(result.scalars().all())


def _model_summary_from_record(record: dict[str, Any]) -> ProjectModelSummary:
    config = record.get("config_json")
    if not isinstance(config, dict):
        config = {}

    source = config.get("source_dataset")
    if not isinstance(source, dict):
        source = {}

    architecture = config.get("architecture")
    if not isinstance(architecture, dict):
        architecture = {}

    backbone = architecture.get("backbone")
    if not isinstance(backbone, dict):
        backbone = {}

    return ProjectModelSummary(
        id=str(record.get("id", "")),
        name=str(record.get("name", "")),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
        task=str(source.get("task", "classification")),
        backbone_name=str(backbone.get("name", "unknown")),
        num_classes=int(source.get("num_classes", 0) or 0),
    )


def _resolve_model_name(*, requested_name: str | None, model_count: int, task: str) -> str:
    if isinstance(requested_name, str) and requested_name.strip():
        return requested_name.strip()
    return f"{task}_model_{model_count + 1}"


@router.get("/projects/{project_id}/models", response_model=list[ProjectModelSummary])
async def list_project_models(project_id: str, db: AsyncSession = Depends(get_db)) -> list[ProjectModelSummary]:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    records = model_store.list_by_project(project_id)
    return [_model_summary_from_record(record) for record in records]


@router.post("/projects/{project_id}/models", response_model=ProjectModelCreateResponse)
async def create_project_model(
    project_id: str,
    payload: ProjectModelCreate,
    db: AsyncSession = Depends(get_db),
) -> ProjectModelCreateResponse:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    latest_dataset_version = (
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
    if latest_dataset_version is None:
        raise api_error(
            status_code=400,
            code="project_manifest_missing",
            message="Project has no exported dataset manifest yet. Export the dataset first.",
            details={"project_id": project_id},
        )

    manifest = latest_dataset_version.manifest_json
    if not isinstance(manifest, dict):
        raise api_error(
            status_code=400,
            code="project_manifest_invalid",
            message="Latest dataset manifest is invalid",
            details={"project_id": project_id, "dataset_version_id": latest_dataset_version.id},
        )

    existing = model_store.list_by_project(project_id)
    task = manifest.get("tasks", {}).get("primary") if isinstance(manifest.get("tasks"), dict) else "classification"
    model_name = _resolve_model_name(requested_name=payload.name, model_count=len(existing), task=str(task))

    try:
        config = build_default_model_config(
            model_name=model_name,
            dataset_manifest_id=latest_dataset_version.id,
            manifest=manifest,
        )
        validate_model_config(config)
    except ManifestConfigError as exc:
        raise api_error(
            status_code=400,
            code="project_manifest_invalid",
            message=str(exc),
            details={"project_id": project_id, "dataset_version_id": latest_dataset_version.id},
        ) from exc
    except ModelConfigValidationError as exc:
        raise api_error(
            status_code=400,
            code="model_config_invalid",
            message="Generated model config failed validation",
            details={"reason": str(exc), "project_id": project_id},
        ) from exc

    created = model_store.create(project_id=project_id, name=model_name, config_json=config)
    return ProjectModelCreateResponse(id=created["id"], name=created["name"], config=created["config_json"])


@router.get("/projects/{project_id}/models/{model_id}", response_model=ProjectModelRecord)
async def get_project_model(project_id: str, model_id: str, db: AsyncSession = Depends(get_db)) -> ProjectModelRecord:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    record = model_store.get(project_id, model_id)
    if record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": model_id},
        )

    return ProjectModelRecord.model_validate(record)


@router.put("/projects/{project_id}/models/{model_id}", response_model=ProjectModelRecord)
async def update_project_model(
    project_id: str,
    model_id: str,
    payload: ProjectModelUpdate,
    db: AsyncSession = Depends(get_db),
) -> ProjectModelRecord:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    record = model_store.get(project_id, model_id)
    if record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": model_id},
        )

    issues = collect_model_config_issues(payload.config_json)
    if issues:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Model config validation failed",
            details={
                "project_id": project_id,
                "model_id": model_id,
                "issues": issues,
            },
        )

    updated = model_store.update_config(project_id=project_id, model_id=model_id, config_json=payload.config_json)
    if updated is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": model_id},
        )

    return ProjectModelRecord.model_validate(updated)


@router.get("/assets/{asset_id}/suggestions")
async def get_asset_suggestions(asset_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(Suggestion).where(Suggestion.asset_id == asset_id))
    return [{"id": s.id, "model_id": s.model_id, "payload_json": s.payload_json} for s in result.scalars().all()]


@router.post("/projects/{project_id}/suggestions/batch")
async def enqueue_batch_suggestions(project_id: str) -> dict:
    return {"project_id": project_id, "status": "queued"}
