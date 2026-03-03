from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import DatasetVersion, Project
from sheriff_api.config import get_settings
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.exports import ExportCreate, ExportRead
from sheriff_api.services.dataset_export_pipeline import build_export_bundle, prepare_export_inputs
from sheriff_api.services.exporter_coco import ExportValidationError
from sheriff_api.services.storage import LocalStorage

router = APIRouter(tags=["exports"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)

@router.post("/projects/{project_id}/exports", response_model=ExportRead)
async def create_export(project_id: str, payload: ExportCreate, db: AsyncSession = Depends(get_db)) -> DatasetVersion:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    export_inputs = await prepare_export_inputs(
        db=db,
        project_id=project_id,
        selection_criteria=payload.selection_criteria_json,
    )

    try:
        built = build_export_bundle(
            project=project,
            selection_criteria=payload.selection_criteria_json,
            inputs=export_inputs,
            storage=storage,
        )
    except ExportValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    manifest = built.manifest
    content_hash = built.content_hash
    storage_uri = f"exports/{project_id}/{content_hash}.zip"
    zip_path = storage.resolve(storage_uri)
    if not zip_path.exists():
        storage.write_bytes(storage_uri, built.zip_bytes)

    dataset = DatasetVersion(
        project_id=project_id,
        selection_criteria_json=payload.selection_criteria_json,
        manifest_json=manifest,
        export_uri=f"/api/v1/projects/{project_id}/exports/{content_hash}/download",
        hash=content_hash,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get("/projects/{project_id}/exports", response_model=list[ExportRead])
async def list_exports(project_id: str, db: AsyncSession = Depends(get_db)) -> list[DatasetVersion]:
    result = await db.execute(select(DatasetVersion).where(DatasetVersion.project_id == project_id))
    return list(result.scalars().all())

@router.get("/projects/{project_id}/exports/{content_hash}/download")
async def download_export(project_id: str, content_hash: str) -> FileResponse:
    storage_uri = f"exports/{project_id}/{content_hash}.zip"
    try:
        path = storage.resolve(storage_uri)
    except ValueError as exc:
        raise api_error(
            status_code=400,
            code="export_path_invalid",
            message="Invalid export path",
            details={"project_id": project_id, "content_hash": content_hash, "reason": str(exc)},
        ) from exc

    if not path.exists() or not path.is_file():
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "content_hash": content_hash},
        )

    return FileResponse(
        path=path,
        media_type="application/zip",
        filename=f"{project_id}-{content_hash[:12]}.zip",
    )
