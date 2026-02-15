import hashlib
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, AssetType, Project, Suggestion
from sheriff_api.config import get_settings
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.assets import AssetCreate, AssetRead
from sheriff_api.services.image_metadata import extract_image_dimensions
from sheriff_api.services.storage import LocalStorage

router = APIRouter(tags=["assets"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)


@router.post("/projects/{project_id}/assets", response_model=AssetRead)
async def create_asset(project_id: str, payload: AssetCreate, db: AsyncSession = Depends(get_db)) -> Asset:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status.HTTP_404_NOT_FOUND, code="project_not_found", message="Project not found")

    asset = Asset(project_id=project_id, **payload.model_dump())
    db.add(asset)
    await db.commit()
    await db.refresh(asset)
    return asset


@router.get("/projects/{project_id}/assets", response_model=list[AssetRead])
async def list_assets(project_id: str, status: str | None = None, db: AsyncSession = Depends(get_db)) -> list[Asset]:
    stmt = select(Asset).where(Asset.project_id == project_id)
    if status:
        stmt = stmt.join(Annotation, Annotation.asset_id == Asset.id).where(Annotation.status == status)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.post("/projects/{project_id}/assets/upload", response_model=AssetRead)
async def upload_asset(
    project_id: str,
    file: UploadFile = File(...),
    relative_path: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
) -> Asset:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            code="project_not_found",
            message="Project not found",
            details={"project_id": project_id},
        )

    content = await file.read()
    if len(content) == 0:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            code="uploaded_file_empty",
            message="Uploaded file is empty",
            details={"filename": file.filename},
        )

    storage.ensure_project_dirs(project_id)

    suffix = Path(file.filename or "upload.bin").suffix.lower()
    generated_id = str(uuid.uuid4())
    relative_uri = f"assets/{project_id}/{generated_id}{suffix}"

    mime_type = file.content_type or "application/octet-stream"
    checksum = hashlib.sha256(content).hexdigest()
    width, height = extract_image_dimensions(content)

    asset = Asset(
        id=generated_id,
        project_id=project_id,
        type=AssetType.image,
        uri=f"/api/v1/assets/{generated_id}/content",
        mime_type=mime_type,
        width=width,
        height=height,
        checksum=checksum,
        metadata_json={
            "storage_uri": relative_uri,
            "original_filename": file.filename,
            "relative_path": relative_path or file.filename,
            "size_bytes": len(content),
        },
    )

    wrote_file = False
    try:
        storage.write_bytes(relative_uri, content)
        wrote_file = True
        db.add(asset)
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        if wrote_file:
            try:
                storage.delete_file(relative_uri)
            except ValueError:
                pass
        raise api_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="asset_persist_failed",
            message="Failed to persist uploaded asset",
            details={"filename": file.filename, "project_id": project_id},
        ) from exc

    await db.refresh(asset)
    return asset


@router.get("/assets/{asset_id}/content")
async def get_asset_content(asset_id: str, db: AsyncSession = Depends(get_db)) -> FileResponse:
    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise api_error(status.HTTP_404_NOT_FOUND, code="asset_not_found", message="Asset not found")

    storage_uri = asset.metadata_json.get("storage_uri")
    if not isinstance(storage_uri, str) or not storage_uri:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            code="asset_path_missing",
            message="Asset file path missing",
            details={"asset_id": asset_id},
        )

    try:
        path = storage.resolve(storage_uri)
    except ValueError as exc:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            code="asset_path_invalid",
            message="Asset file path is invalid",
            details={"asset_id": asset_id, "reason": str(exc)},
        ) from exc

    if not path.exists() or not path.is_file():
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            code="asset_file_missing",
            message="Asset file not found on disk",
            details={"asset_id": asset_id},
        )

    return FileResponse(path=path, media_type=asset.mime_type, filename=os.path.basename(path))


@router.delete("/projects/{project_id}/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(project_id: str, asset_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    asset = await db.get(Asset, asset_id)
    if asset is None or asset.project_id != project_id:
        raise api_error(status.HTTP_404_NOT_FOUND, code="asset_not_found", message="Asset not found")

    storage_uri = asset.metadata_json.get("storage_uri")

    await db.execute(delete(Annotation).where(Annotation.asset_id == asset_id))
    await db.execute(delete(Suggestion).where(Suggestion.asset_id == asset_id))
    await db.delete(asset)
    await db.commit()

    if isinstance(storage_uri, str) and storage_uri:
        try:
            storage.delete_file(storage_uri)
        except ValueError:
            pass

    return Response(status_code=status.HTTP_204_NO_CONTENT)
