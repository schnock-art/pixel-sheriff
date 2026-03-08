from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import AnnotationStatus, Project, Task, TaskKind
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.datasets import (
    DatasetPreviewRequest,
    DatasetPreviewResponse,
    DatasetSetActiveRequest,
    DatasetVersionAssetsResponse,
    DatasetVersionCreateRequest,
    DatasetVersionEnvelope,
    DatasetVersionExportResponse,
    DatasetVersionListResponse,
)
from sheriff_api.services.dataset_export_builder import build_dataset_export
from sheriff_api.services.dataset_selection import (
    build_category_snapshot,
    class_counts,
    load_asset_rows,
    resolve_dataset_preview,
    sample_asset_item,
    split_counts,
    to_selection_payload,
    validate_split_ratios,
)
from sheriff_api.services.dataset_store import DatasetStore, DatasetStoreValidationError
from sheriff_api.services.storage import LocalStorage

router = APIRouter(tags=["datasets"])
settings = get_settings()
dataset_store = DatasetStore(settings.storage_root)
storage = LocalStorage(settings.storage_root)


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


async def _require_task(db: AsyncSession, project_id: str, task_id: str) -> Task:
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )
    return task


@router.get("/projects/{project_id}/datasets/versions", response_model=DatasetVersionListResponse)
async def list_dataset_versions(project_id: str, task_id: str | None = None) -> DatasetVersionListResponse:
    listed = dataset_store.list_versions(project_id, task_id=task_id)
    return DatasetVersionListResponse(
        active_dataset_version_id=listed["active_dataset_version_id"],
        items=[DatasetVersionEnvelope(**item) for item in listed["items"]],
    )


@router.post("/projects/{project_id}/datasets/versions/preview", response_model=DatasetPreviewResponse)
async def preview_dataset_version(
    project_id: str,
    payload: DatasetPreviewRequest,
    db: AsyncSession = Depends(get_db),
) -> DatasetPreviewResponse:
    await _require_project(db, project_id)
    task = await _require_task(db, project_id, payload.task_id)
    try:
        ratios = validate_split_ratios(payload.split.ratios.model_dump())
    except ValueError as exc:
        raise api_error(status_code=422, code="dataset_split_invalid", message=str(exc)) from exc

    try:
        resolution = await resolve_dataset_preview(
            db=db,
            project_id=project_id,
            task_id=payload.task_id,
            task_kind=task.kind.value,
            mode=payload.selection.mode,
            filters=payload.selection.filters.model_dump(),
            explicit_asset_ids=payload.selection.explicit_asset_ids,
            seed=payload.split.seed,
            ratios=ratios,
            stratify_enabled=payload.split.stratify.enabled,
            strict_stratify=payload.split.stratify.strict_stratify,
        )
    except RuntimeError as exc:
        if str(exc) == "dataset_stratify_impossible":
            raise api_error(
                status_code=422,
                code="dataset_stratify_impossible",
                message="Stratified split cannot be satisfied with current selection",
            ) from exc
        raise

    selected = resolution.selected_rows
    split_by_asset = resolution.split_by_asset
    warnings = resolution.warnings
    all_asset_ids = [row.asset.id for row in selected]
    if payload.strict_preview_cap and len(all_asset_ids) > payload.preview_cap:
        raise api_error(
            status_code=422,
            code="dataset_preview_too_large",
            message="Preview result exceeds cap",
            details={"cap": payload.preview_cap, "count": len(all_asset_ids)},
        )

    sample_rows = selected[: min(120, payload.preview_cap)]
    category_snapshot = await build_category_snapshot(db, project_id, payload.task_id, selected)
    class_names = {
        class_id: category_snapshot.class_names_by_id.get(class_id, class_id)
        for class_id in class_counts(selected, task_kind=task.kind.value)
        if class_id != "__missing__"
    }

    return DatasetPreviewResponse(
        asset_ids=all_asset_ids[: payload.preview_cap],
        sample_asset_ids=all_asset_ids[: min(120, payload.preview_cap)],
        sample_assets=[sample_asset_item(row, split_by_asset) for row in sample_rows],
        class_names=class_names,
        counts={
            "total": len(all_asset_ids),
            "class_counts": class_counts(selected, task_kind=task.kind.value),
            "split_counts": split_counts(split_by_asset),
        },
        warnings=[*warnings, *category_snapshot.warnings],
    )


@router.post("/projects/{project_id}/datasets/versions", response_model=DatasetVersionEnvelope)
async def create_dataset_version(
    project_id: str,
    payload: DatasetVersionCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionEnvelope:
    project = await _require_project(db, project_id)
    task = await _require_task(db, project_id, payload.task_id)
    try:
        ratios = validate_split_ratios(payload.split.ratios.model_dump())
    except ValueError as exc:
        raise api_error(status_code=422, code="dataset_split_invalid", message=str(exc)) from exc

    try:
        resolution = await resolve_dataset_preview(
            db=db,
            project_id=project_id,
            task_id=payload.task_id,
            task_kind=task.kind.value,
            mode=payload.selection.mode,
            filters=payload.selection.filters.model_dump(),
            explicit_asset_ids=payload.selection.explicit_asset_ids,
            seed=payload.split.seed,
            ratios=ratios,
            stratify_enabled=payload.split.stratify.enabled,
            strict_stratify=payload.split.stratify.strict_stratify,
        )
    except RuntimeError as exc:
        if str(exc) == "dataset_stratify_impossible":
            raise api_error(
                status_code=422,
                code="dataset_stratify_impossible",
                message="Stratified split cannot be satisfied with current selection",
            ) from exc
        raise

    selected = resolution.selected_rows
    split_by_asset = resolution.split_by_asset
    warnings = resolution.warnings
    category_snapshot = await build_category_snapshot(db, project_id, payload.task_id, selected)
    selected_asset_ids = [row.asset.id for row in selected]
    split_items = [
        {"asset_id": asset_id, "split": split_by_asset.get(asset_id, "train")}
        for asset_id in selected_asset_ids
    ]
    dataset_version_payload = {
        "schema_version": "2.0",
        "dataset_version_id": str(uuid.uuid4()),
        "project_id": project_id,
        "task_id": payload.task_id,
        "name": payload.name,
        "task": task.kind.value,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"source_type": "project_assets_snapshot"},
        "assets": {
            "count": len(selected_asset_ids),
            "asset_ids": selected_asset_ids,
        },
        "labels": {
            "label_schema": {
                "version": "1.2",
                "class_order": category_snapshot.class_order,
                "classes": category_snapshot.classes,
                "rules": {"names_normalized": "lowercase_slug"},
            },
        },
        "selection": to_selection_payload(
            payload.selection.mode,
            payload.selection.filters.model_dump(mode="json", exclude_none=True),
            payload.selection.explicit_asset_ids,
        ),
        "splits": {
            "strategy": "random_seeded",
            "seed": payload.split.seed,
            "ratios": payload.split.ratios.model_dump(),
            "stratify": {
                "enabled": payload.split.stratify.enabled,
                "by": payload.split.stratify.by,
            },
            "items": split_items,
        },
        "stats": {
            "asset_count": len(selected_asset_ids),
            "class_counts": class_counts(selected, task_kind=task.kind.value),
            "split_counts": split_counts(split_by_asset),
            "warnings": [*warnings, *category_snapshot.warnings],
        },
    }
    if isinstance(payload.description, str) and payload.description.strip():
        dataset_version_payload["description"] = payload.description
    if isinstance(payload.created_by, str) and payload.created_by.strip():
        dataset_version_payload["created_by"] = payload.created_by
    if task.kind == TaskKind.classification and task.label_mode is not None:
        dataset_version_payload["labels"]["label_mode"] = task.label_mode.value
    try:
        created = dataset_store.create_version(project_id, dataset_version_payload)
    except DatasetStoreValidationError as exc:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Dataset version validation failed",
            details={"issues": exc.issues},
        ) from exc

    current = dataset_store.list_versions(project_id)
    should_set_active = payload.set_active or not current.get("active_dataset_version_id")
    if should_set_active:
        dataset_store.set_active(project_id, created["dataset_version_id"])
    loaded = dataset_store.get_version(project_id, created["dataset_version_id"])
    return DatasetVersionEnvelope(**loaded)


@router.patch("/projects/{project_id}/datasets/active")
async def set_active_dataset_version(project_id: str, payload: DatasetSetActiveRequest) -> dict[str, Any]:
    try:
        dataset_store.set_active(project_id, payload.active_dataset_version_id)
    except KeyError as exc:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": payload.active_dataset_version_id},
        ) from exc
    return {"ok": True, "active_dataset_version_id": payload.active_dataset_version_id}


@router.get("/projects/{project_id}/datasets/versions/{dataset_version_id}", response_model=DatasetVersionEnvelope)
async def get_dataset_version(project_id: str, dataset_version_id: str) -> DatasetVersionEnvelope:
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    return DatasetVersionEnvelope(**loaded)


@router.get("/projects/{project_id}/datasets/versions/{dataset_version_id}/assets", response_model=DatasetVersionAssetsResponse)
async def list_dataset_version_assets(
    project_id: str,
    dataset_version_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=250),
    split: str | None = None,
    status: AnnotationStatus | None = None,
    class_id: str | None = None,
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionAssetsResponse:
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    version = loaded["version"]
    task_id = str(version.get("task_id") or "")
    if not task_id:
        raise api_error(
            status_code=422,
            code="dataset_version_invalid",
            message="Dataset version is missing task_id",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    version_asset_ids = version.get("assets", {}).get("asset_ids")
    split_items = version.get("splits", {}).get("items")
    if not isinstance(version_asset_ids, list):
        version_asset_ids = []
    split_by_asset = {}
    if isinstance(split_items, list):
        for item in split_items:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id")
            split_value = item.get("split")
            if isinstance(asset_id, str) and isinstance(split_value, str):
                split_by_asset[asset_id] = split_value

    rows = await load_asset_rows(db, project_id, task_id)
    version_asset_id_set = {str(asset_id) for asset_id in version_asset_ids}
    # For saved dataset versions, membership comes only from stored version assets/splits.
    # Live DB rows are used only to enrich display fields such as status/labels/path.
    scoped = [row for row in rows if row.asset.id in version_asset_id_set]
    if split in {"train", "val", "test"}:
        scoped = [row for row in scoped if split_by_asset.get(row.asset.id) == split]
    if status is not None:
        scoped = [row for row in scoped if row.status == status.value]
    if isinstance(class_id, str) and class_id.strip():
        normalized = class_id.strip()
        scoped = [row for row in scoped if normalized in row.category_ids]
    if isinstance(search, str) and search.strip():
        needle = search.strip().lower()
        scoped = [
            row
            for row in scoped
            if needle in row.filename.lower() or needle in row.relative_path.lower() or needle in row.asset.id.lower()
        ]

    total = len(scoped)
    start = (page - 1) * page_size
    end = start + page_size
    paged = scoped[start:end]

    return DatasetVersionAssetsResponse(
        items=[
            {
                "asset_id": row.asset.id,
                "filename": row.filename,
                "relative_path": row.relative_path,
                "status": row.status,
                "split": split_by_asset.get(row.asset.id),
                "label_summary": {
                    "primary_category_id": row.primary_category_id,
                    "category_ids": row.category_ids,
                },
            }
            for row in paged
        ],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("/projects/{project_id}/datasets/versions/{dataset_version_id}/export", response_model=DatasetVersionExportResponse)
async def export_dataset_version(
    project_id: str,
    dataset_version_id: str,
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionExportResponse:
    project = await _require_project(db, project_id)
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )

    existing_artifact = dataset_store.get_export_artifact(project_id, dataset_version_id)
    if isinstance(existing_artifact, dict):
        existing_hash = existing_artifact.get("hash")
        if isinstance(existing_hash, str):
            relpath = f"exports/{project_id}/{existing_hash}.zip"
            if storage.resolve(relpath).exists():
                return DatasetVersionExportResponse(
                    dataset_version_id=dataset_version_id,
                    hash=existing_hash,
                    export_uri=str(existing_artifact.get("export_uri")),
                )

    version = loaded["version"]
    task_id = str(version.get("task_id") or "")
    if not task_id:
        raise api_error(
            status_code=422,
            code="dataset_version_invalid",
            message="Dataset version is missing task_id",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    task = await _require_task(db, project_id, task_id)

    content_hash, export_uri = await build_dataset_export(
        db=db,
        storage=storage,
        project=project,
        task=task,
        dataset_version=version,
    )
    dataset_store.set_export_artifact(
        project_id,
        dataset_version_id,
        {
            "hash": content_hash,
            "export_uri": export_uri,
        },
    )
    return DatasetVersionExportResponse(
        dataset_version_id=dataset_version_id,
        hash=content_hash,
        export_uri=export_uri,
    )


@router.get("/projects/{project_id}/datasets/versions/{dataset_version_id}/export/download")
async def download_dataset_version_export(project_id: str, dataset_version_id: str) -> FileResponse:
    artifact = dataset_store.get_export_artifact(project_id, dataset_version_id)
    if not isinstance(artifact, dict):
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    content_hash = artifact.get("hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    relpath = f"exports/{project_id}/{content_hash}.zip"
    try:
        path = storage.resolve(relpath)
    except ValueError as exc:
        raise api_error(
            status_code=400,
            code="export_path_invalid",
            message="Invalid export path",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id, "reason": str(exc)},
        ) from exc
    if not path.exists() or not path.is_file():
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    return FileResponse(
        path=path,
        media_type="application/zip",
        filename=f"{project_id}-{dataset_version_id[:8]}-{content_hash[:8]}.zip",
    )
