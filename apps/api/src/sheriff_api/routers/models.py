import hashlib
import json
from datetime import datetime, timezone
from typing import Any
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Asset, DatasetVersion, Model, Project, Suggestion
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
from sheriff_api.services.storage import LocalStorage
from sheriff_api.services.suggestion_queue import SuggestionQueue

router = APIRouter(tags=["models"])
settings = get_settings()
model_store = ModelStore(settings.storage_root)
storage = LocalStorage(settings.storage_root)
suggestion_queue = SuggestionQueue()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


def _require_project_model(project_id: str, model_id: str) -> dict[str, Any]:
    record = model_store.get(project_id, model_id)
    if record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": model_id},
        )
    return record


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _normalized_suggestion_status(payload_json: dict[str, Any]) -> str:
    raw = payload_json.get("status")
    if isinstance(raw, str) and raw in {"pending", "accepted", "rejected"}:
        return raw
    return "pending"


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
    await _require_project(db, project_id)

    records = model_store.list_by_project(project_id)
    return [_model_summary_from_record(record) for record in records]


@router.post("/projects/{project_id}/models", response_model=ProjectModelCreateResponse)
async def create_project_model(
    project_id: str,
    payload: ProjectModelCreate,
    db: AsyncSession = Depends(get_db),
) -> ProjectModelCreateResponse:
    await _require_project(db, project_id)

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
    await _require_project(db, project_id)
    record = _require_project_model(project_id, model_id)

    return ProjectModelRecord.model_validate(record)


@router.put("/projects/{project_id}/models/{model_id}", response_model=ProjectModelRecord)
async def update_project_model(
    project_id: str,
    model_id: str,
    payload: ProjectModelUpdate,
    db: AsyncSession = Depends(get_db),
) -> ProjectModelRecord:
    await _require_project(db, project_id)
    _require_project_model(project_id, model_id)

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


@router.post("/projects/{project_id}/models/{model_id}/exports")
async def export_project_model(project_id: str, model_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await _require_project(db, project_id)
    model_record = _require_project_model(project_id, model_id)
    config_json = model_record.get("config_json")
    if not isinstance(config_json, dict):
        raise api_error(
            status_code=422,
            code="model_config_invalid",
            message="Model config validation failed",
            details={"project_id": project_id, "model_id": model_id},
        )

    issues = collect_model_config_issues(config_json)
    if issues:
        raise api_error(
            status_code=422,
            code="model_config_invalid",
            message="Model config validation failed",
            details={"project_id": project_id, "model_id": model_id, "issues": issues},
        )

    export_spec = config_json.get("export")
    onnx_spec = export_spec.get("onnx") if isinstance(export_spec, dict) else None
    if not isinstance(onnx_spec, dict) or not bool(onnx_spec.get("enabled")):
        raise api_error(
            status_code=422,
            code="model_export_disabled",
            message="Model ONNX export is disabled",
            details={"project_id": project_id, "model_id": model_id},
        )

    config_hash = hashlib.sha256(_canonical_json_bytes(config_json)).hexdigest()
    payload = {
        "schema_version": "1.0",
        "project_id": project_id,
        "model_id": model_id,
        "model_name": model_record.get("name"),
        "format": "onnx",
        "source_config_hash": config_hash,
        "onnx": {
            "enabled": True,
            "opset": onnx_spec.get("opset"),
            "dynamic_shapes": onnx_spec.get("dynamic_shapes"),
            "output_names": onnx_spec.get("output_names"),
        },
    }
    payload_bytes = _canonical_json_bytes(payload)
    content_hash = hashlib.sha256(payload_bytes).hexdigest()
    relpath = f"model_exports/{project_id}/{model_id}/{content_hash}.json"
    if not storage.resolve(relpath).exists():
        storage.write_bytes(relpath, payload_bytes)

    return {
        "project_id": project_id,
        "model_id": model_id,
        "format": "onnx",
        "hash": content_hash,
        "created_at": _utc_now_iso(),
        "export_uri": f"/api/v1/projects/{project_id}/models/{model_id}/exports/{content_hash}/download",
    }


@router.get("/projects/{project_id}/models/{model_id}/exports/{content_hash}/download")
async def download_project_model_export(
    project_id: str,
    model_id: str,
    content_hash: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    await _require_project(db, project_id)
    _require_project_model(project_id, model_id)

    relpath = f"model_exports/{project_id}/{model_id}/{content_hash}.json"
    try:
        path = storage.resolve(relpath)
    except ValueError as exc:
        raise api_error(
            status_code=400,
            code="model_export_path_invalid",
            message="Model export path is invalid",
            details={"reason": str(exc), "project_id": project_id, "model_id": model_id},
        ) from exc
    if not path.exists() or not path.is_file():
        raise api_error(
            status_code=404,
            code="model_export_not_found",
            message="Model export file not found",
            details={"project_id": project_id, "model_id": model_id, "hash": content_hash},
        )
    return FileResponse(
        path=path,
        media_type="application/json",
        filename=f"{project_id}-{model_id[:8]}-{content_hash[:12]}.json",
    )


@router.get("/assets/{asset_id}/suggestions")
async def get_asset_suggestions(asset_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(Suggestion).where(Suggestion.asset_id == asset_id))
    suggestions = list(result.scalars().all())
    return [
        {
            "id": suggestion.id,
            "asset_id": suggestion.asset_id,
            "model_id": suggestion.model_id,
            "status": _normalized_suggestion_status(suggestion.payload_json if isinstance(suggestion.payload_json, dict) else {}),
            "payload_json": suggestion.payload_json,
        }
        for suggestion in suggestions
    ]


@router.post("/projects/{project_id}/suggestions/batch")
async def enqueue_batch_suggestions(
    project_id: str,
    payload: dict[str, Any] | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await _require_project(db, project_id)
    body = payload if isinstance(payload, dict) else {}

    model_id_raw = body.get("model_id")
    model_id = str(model_id_raw).strip() if model_id_raw is not None else ""
    if not model_id:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="model_id is required",
            details={"project_id": project_id, "field": "model_id"},
        )

    model = await db.get(Model, model_id)
    if model is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found",
            details={"project_id": project_id, "model_id": model_id},
        )

    asset_rows = list((await db.execute(select(Asset.id).where(Asset.project_id == project_id))).scalars().all())
    request_id = str(uuid.uuid4())

    created_ids: list[str] = []
    for asset_id in asset_rows:
        suggestion = Suggestion(
            asset_id=asset_id,
            model_id=model_id,
            payload_json={
                "status": "pending",
                "request_id": request_id,
                "queued_at": _utc_now_iso(),
                "source": "batch",
            },
        )
        db.add(suggestion)
        await db.flush()
        created_ids.append(suggestion.id)
    await db.commit()

    try:
        await suggestion_queue.enqueue_batch_job(
            {
                "job_version": "1",
                "job_type": "suggest_batch",
                "request_id": request_id,
                "project_id": project_id,
                "model_id": model_id,
                "asset_ids": asset_rows,
                "suggestion_ids": created_ids,
            }
        )
    except Exception as exc:
        raise api_error(
            status_code=503,
            code="suggestion_queue_unavailable",
            message="Suggestion queue is unavailable",
            details={"project_id": project_id, "request_id": request_id},
        ) from exc

    return {"project_id": project_id, "status": "queued", "request_id": request_id, "queued": len(created_ids)}


async def _require_project_suggestion(db: AsyncSession, project_id: str, suggestion_id: str) -> tuple[Suggestion, Asset]:
    row = (
        (
            await db.execute(
                select(Suggestion, Asset)
                .join(Asset, Asset.id == Suggestion.asset_id)
                .where(Suggestion.id == suggestion_id, Asset.project_id == project_id)
            )
        )
        .first()
    )
    if row is None:
        raise api_error(
            status_code=404,
            code="suggestion_not_found",
            message="Suggestion not found in project",
            details={"project_id": project_id, "suggestion_id": suggestion_id},
        )
    suggestion, asset = row
    return suggestion, asset


@router.post("/projects/{project_id}/suggestions/{suggestion_id}/accept")
async def accept_suggestion(
    project_id: str,
    suggestion_id: str,
    payload: dict[str, Any] | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await _require_project(db, project_id)
    suggestion, asset = await _require_project_suggestion(db, project_id, suggestion_id)
    body = payload if isinstance(payload, dict) else {}
    next_payload = dict(suggestion.payload_json if isinstance(suggestion.payload_json, dict) else {})
    next_payload.update(
        {
            "status": "accepted",
            "decision": "accept",
            "decided_at": _utc_now_iso(),
        }
    )
    if isinstance(body.get("annotation_payload"), dict):
        next_payload["annotation_payload"] = body["annotation_payload"]
    suggestion.payload_json = next_payload
    await db.commit()
    await db.refresh(suggestion)
    return {
        "id": suggestion.id,
        "asset_id": asset.id,
        "project_id": project_id,
        "model_id": suggestion.model_id,
        "status": "accepted",
        "payload_json": suggestion.payload_json,
    }


@router.post("/projects/{project_id}/suggestions/{suggestion_id}/reject")
async def reject_suggestion(
    project_id: str,
    suggestion_id: str,
    payload: dict[str, Any] | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await _require_project(db, project_id)
    suggestion, asset = await _require_project_suggestion(db, project_id, suggestion_id)
    body = payload if isinstance(payload, dict) else {}
    next_payload = dict(suggestion.payload_json if isinstance(suggestion.payload_json, dict) else {})
    next_payload.update(
        {
            "status": "rejected",
            "decision": "reject",
            "decided_at": _utc_now_iso(),
        }
    )
    reason = body.get("reason")
    if isinstance(reason, str) and reason.strip():
        next_payload["reason"] = reason.strip()
    suggestion.payload_json = next_payload
    await db.commit()
    await db.refresh(suggestion)
    return {
        "id": suggestion.id,
        "asset_id": asset.id,
        "project_id": project_id,
        "model_id": suggestion.model_id,
        "status": "rejected",
        "payload_json": suggestion.payload_json,
    }
