from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Project, Task
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.prelabels import PrelabelConfigCreate
from sheriff_api.schemas.video_imports import VideoImportResponse
from sheriff_api.services.media_queue import MediaQueue
from sheriff_api.services.prelabels import create_prelabel_session
from sheriff_api.services.sequences import create_sequence_with_folder, sequence_to_read
from sheriff_api.services.storage import LocalStorage
from sheriff_api.services.video_frames import validate_video_import_params, VideoImportValidationError

router = APIRouter(tags=["video-imports"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)
media_queue = MediaQueue()


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


@router.post("/projects/{project_id}/video-imports", response_model=VideoImportResponse)
async def import_video(
    project_id: str,
    file: UploadFile = File(...),
    task_id: str | None = Form(default=None),
    folder_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    fps: float = Form(default=2.0),
    max_frames: int = Form(default=500),
    resize_mode: str = Form(default="original"),
    resize_width: int | None = Form(default=None),
    resize_height: int | None = Form(default=None),
    prelabel_config: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
) -> VideoImportResponse:
    await _require_project(db, project_id)
    task = await _require_task(db, project_id, task_id)

    try:
        params = validate_video_import_params(
            filename=file.filename,
            fps=fps,
            max_frames=max_frames,
            resize_mode=resize_mode,
            resize_width=resize_width,
            resize_height=resize_height,
        )
    except VideoImportValidationError as exc:
        raise api_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ) from exc

    content = await file.read()
    if not content:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            code="uploaded_file_empty",
            message="Uploaded file is empty",
            details={"filename": file.filename},
        )

    parsed_prelabel_config: PrelabelConfigCreate | None = None
    if prelabel_config is not None:
        try:
            parsed_prelabel_config = PrelabelConfigCreate.model_validate_json(prelabel_config)
        except Exception as exc:
            raise api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="prelabel_config_invalid",
                message="Prelabel config is invalid",
                details={"reason": str(exc)},
            ) from exc

    requested_name = (name or Path(file.filename or "video_session").stem).strip() or "video_session"
    try:
        folder, sequence = await create_sequence_with_folder(
            db,
            project_id=project_id,
            task_id=task_id,
            folder_id=folder_id,
            requested_name=requested_name,
            source_type="video_file",
            source_filename=file.filename,
            status="processing",
            fps=params["fps"],
        )
    except ValueError as exc:
        code = str(exc)
        if code == "folder_not_found":
            raise api_error(status.HTTP_404_NOT_FOUND, code=code, message="Folder not found") from exc
        if code == "folder_sequence_exists":
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Folder already belongs to a sequence") from exc
        if code == "folder_not_empty":
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Folder already contains assets") from exc
        raise api_error(status.HTTP_500_INTERNAL_SERVER_ERROR, code="video_import_create_failed", message="Failed to create video import") from exc

    prelabel_session_id: str | None = None
    if parsed_prelabel_config is not None:
        if task is None:
            raise api_error(status.HTTP_422_UNPROCESSABLE_ENTITY, code="task_id_required", message="task_id is required for prelabels")
        try:
            prelabel_session = await create_prelabel_session(
                db,
                project_id=project_id,
                task=task,
                sequence=sequence,
                config=parsed_prelabel_config,
                live_mode=False,
            )
            prelabel_session_id = prelabel_session.id
        except ValueError as exc:
            code = str(exc)
            if code in {"active_deployment_not_found", "active_deployment_incompatible"}:
                raise api_error(status.HTTP_409_CONFLICT, code=code, message="Active deployment is unavailable for this task") from exc
            if code == "task_kind_unsupported":
                raise api_error(status.HTTP_409_CONFLICT, code=code, message="Prelabels are supported only for bbox tasks") from exc
            raise

    storage.ensure_project_dirs(project_id)
    import_storage_uri = f"imports/{project_id}/{sequence.id}/{Path(file.filename or 'source.mp4').name}"
    wrote_file = False
    try:
        storage.write_bytes(import_storage_uri, content)
        wrote_file = True
        await media_queue.enqueue_extract_video_job(
            {
                "job_version": "1",
                "job_type": "extract_video_frames",
                "project_id": project_id,
                "sequence_id": sequence.id,
                "task_id": task_id,
                "folder_id": folder.id,
                "video_storage_uri": import_storage_uri,
                "fps": params["fps"],
                "max_frames": params["max_frames"],
                "resize_mode": params["resize_mode"],
                "resize_width": params["resize_width"],
                "resize_height": params["resize_height"],
                "prelabel_session_id": prelabel_session_id,
            }
        )
        await db.commit()
        await db.refresh(sequence)
    except Exception as exc:
        await db.rollback()
        if wrote_file:
            try:
                storage.delete_file(import_storage_uri)
            except ValueError:
                pass
        raise api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="media_queue_unavailable",
            message="Video import queue is unavailable",
            details={"project_id": project_id},
        ) from exc

    return VideoImportResponse(sequence=sequence_to_read(sequence, folder=folder), prelabel_session_id=prelabel_session_id)
