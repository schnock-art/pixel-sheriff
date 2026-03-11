from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Asset, AssetSequence, AssetType, Folder, Project, Task
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.assets import AssetRead
from sheriff_api.schemas.sequences import AssetSequenceRead, SequenceStatusRead, WebcamSessionCreate, WebcamSessionCreateResponse
from sheriff_api.services.asset_ingest import persist_asset_bytes
from sheriff_api.services.sequences import (
    annotated_asset_ids_for_sequence,
    asset_to_read,
    create_sequence_with_folder,
    refresh_sequence_counts,
    sequence_status_to_read,
    sequence_to_read,
)
from sheriff_api.services.storage import LocalStorage

router = APIRouter(tags=["sequences"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status.HTTP_404_NOT_FOUND, code="project_not_found", message="Project not found")
    return project


async def _require_task(db: AsyncSession, project_id: str, task_id: str | None) -> Task | None:
    if not task_id:
        return None
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )
    return task


async def _require_sequence(db: AsyncSession, project_id: str, sequence_id: str) -> AssetSequence:
    sequence = await db.get(AssetSequence, sequence_id)
    if sequence is None or sequence.project_id != project_id:
        raise api_error(status.HTTP_404_NOT_FOUND, code="sequence_not_found", message="Sequence not found")
    return sequence


@router.get("/projects/{project_id}/sequences", response_model=list[AssetSequenceRead])
async def list_sequences(
    project_id: str,
    task_id: str | None = None,
    folder_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[AssetSequenceRead]:
    await _require_project(db, project_id)
    stmt = select(AssetSequence).where(AssetSequence.project_id == project_id)
    if task_id:
        stmt = stmt.where(AssetSequence.task_id == task_id)
    if folder_id:
        stmt = stmt.where(AssetSequence.folder_id == folder_id)

    sequences = list((await db.execute(stmt)).scalars().all())
    folders = {
        folder.id: folder
        for folder in (await db.execute(select(Folder).where(Folder.project_id == project_id))).scalars().all()
    }
    sequences.sort(key=lambda sequence: (sequence.created_at, sequence.id))
    return [sequence_to_read(sequence, folder=folders.get(sequence.folder_id)) for sequence in sequences]


@router.get("/projects/{project_id}/sequences/{sequence_id}", response_model=AssetSequenceRead)
async def get_sequence(
    project_id: str,
    sequence_id: str,
    task_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> AssetSequenceRead:
    await _require_project(db, project_id)
    sequence = await _require_sequence(db, project_id, sequence_id)
    folder = await db.get(Folder, sequence.folder_id) if sequence.folder_id else None
    assets = list((await db.execute(select(Asset).where(Asset.sequence_id == sequence.id))).scalars().all())
    asset_ids = [asset.id for asset in assets]
    annotated_asset_ids = await annotated_asset_ids_for_sequence(db, task_id=task_id, asset_ids=asset_ids)
    return sequence_to_read(
        sequence,
        folder=folder,
        assets=assets,
        annotated_asset_ids=annotated_asset_ids,
    )


@router.get("/projects/{project_id}/sequences/{sequence_id}/status", response_model=SequenceStatusRead)
async def get_sequence_status(project_id: str, sequence_id: str, db: AsyncSession = Depends(get_db)) -> SequenceStatusRead:
    await _require_project(db, project_id)
    sequence = await _require_sequence(db, project_id, sequence_id)
    return sequence_status_to_read(sequence)


@router.post("/projects/{project_id}/webcam-sessions", response_model=WebcamSessionCreateResponse)
async def create_webcam_session(
    project_id: str,
    payload: WebcamSessionCreate,
    db: AsyncSession = Depends(get_db),
) -> WebcamSessionCreateResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, payload.task_id)

    try:
        folder, sequence = await create_sequence_with_folder(
            db,
            project_id=project_id,
            task_id=payload.task_id,
            folder_id=payload.folder_id,
            folder_path=payload.folder_path,
            requested_name=payload.name,
            source_type="webcam",
            source_filename=None,
            status="ready",
            fps=payload.fps,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "folder_not_found":
            raise api_error(status.HTTP_404_NOT_FOUND, code=code, message="Folder not found") from exc
        if code == "folder_sequence_exists":
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Folder already belongs to a sequence") from exc
        if code == "folder_not_empty":
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Folder already contains assets") from exc
        raise api_error(status.HTTP_500_INTERNAL_SERVER_ERROR, code="sequence_create_failed", message="Failed to create webcam session") from exc

    await db.commit()
    await db.refresh(sequence)
    return WebcamSessionCreateResponse(sequence=sequence_to_read(sequence, folder=folder))


@router.post("/projects/{project_id}/sequences/{sequence_id}/frames", response_model=AssetRead)
async def upload_webcam_frame(
    project_id: str,
    sequence_id: str,
    file: UploadFile = File(...),
    frame_index: int = Form(...),
    timestamp_seconds: float | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
) -> AssetRead:
    await _require_project(db, project_id)
    sequence = await _require_sequence(db, project_id, sequence_id)
    if sequence.source_type != "webcam":
        raise api_error(
            status.HTTP_409_CONFLICT,
            code="sequence_frame_upload_invalid",
            message="Frames can only be uploaded to webcam sequences",
        )
    if frame_index < 0:
        raise api_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="frame_index_invalid",
            message="Frame index must be >= 0",
            details={"frame_index": frame_index},
        )

    existing = (
        await db.execute(select(Asset.id).where(Asset.sequence_id == sequence.id, Asset.frame_index == frame_index))
    ).scalar_one_or_none()
    if existing is not None:
        raise api_error(
            status.HTTP_409_CONFLICT,
            code="sequence_frame_exists",
            message="Frame index already exists in this sequence",
            details={"frame_index": frame_index},
        )

    folder = await db.get(Folder, sequence.folder_id) if sequence.folder_id else None
    content = await file.read()
    if not content:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            code="uploaded_file_empty",
            message="Uploaded file is empty",
            details={"filename": file.filename},
        )

    storage.ensure_project_dirs(project_id)
    asset: Asset | None = None
    storage_uri: str | None = None
    try:
        asset = await persist_asset_bytes(
            db=db,
            storage=storage,
            project_id=project_id,
            content=content,
            file_name=file.filename or f"frame_{frame_index + 1:06d}.jpg",
            mime_type=file.content_type or "image/jpeg",
            folder=folder,
            original_filename=file.filename or f"frame_{frame_index + 1:06d}.jpg",
            asset_type=AssetType.frame,
            sequence_id=sequence.id,
            sequence_name=sequence.name,
            source_kind="webcam_frame",
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            commit=False,
        )
        storage_uri = asset.metadata_json.get("storage_uri") if isinstance(asset.metadata_json, dict) else None
        await refresh_sequence_counts(db, sequence.id)
        if asset.width is not None:
            sequence.width = asset.width
        if asset.height is not None:
            sequence.height = asset.height
        if timestamp_seconds is not None:
            sequence.duration_seconds = max(float(sequence.duration_seconds or 0.0), float(timestamp_seconds))
        await db.commit()
        await db.refresh(asset)
    except Exception as exc:
        await db.rollback()
        if isinstance(storage_uri, str):
            try:
                storage.delete_file(storage_uri)
            except ValueError:
                pass
        raise api_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="sequence_frame_upload_failed",
            message="Failed to upload webcam frame",
            details={"sequence_id": sequence.id, "filename": file.filename},
        ) from exc

    return asset_to_read(asset)
