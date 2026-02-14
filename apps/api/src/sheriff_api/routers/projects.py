from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Asset, Category, DatasetVersion, Project, Suggestion
from sheriff_api.db.session import get_db
from sheriff_api.schemas.projects import ProjectCreate, ProjectRead
from sheriff_api.services.storage import LocalStorage

router = APIRouter(prefix="/projects", tags=["projects"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)


@router.post("", response_model=ProjectRead)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)) -> Project:
    project = Project(name=payload.name, task_type=payload.task_type)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=list[ProjectRead])
async def list_projects(db: AsyncSession = Depends(get_db)) -> list[Project]:
    result = await db.execute(select(Project))
    return list(result.scalars().all())


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    asset_rows = list(
        (
            await db.execute(
                select(Asset.id, Asset.metadata_json).where(Asset.project_id == project_id),
            )
        ).all()
    )
    asset_ids = [asset_id for asset_id, _metadata in asset_rows]
    storage_uris = [
        metadata.get("storage_uri")
        for _asset_id, metadata in asset_rows
        if isinstance(metadata, dict) and isinstance(metadata.get("storage_uri"), str)
    ]

    if asset_ids:
        await db.execute(delete(Suggestion).where(Suggestion.asset_id.in_(asset_ids)))
    await db.execute(delete(Annotation).where(Annotation.project_id == project_id))
    await db.execute(delete(DatasetVersion).where(DatasetVersion.project_id == project_id))
    await db.execute(delete(Category).where(Category.project_id == project_id))
    await db.execute(delete(Asset).where(Asset.project_id == project_id))
    await db.delete(project)
    await db.commit()

    for storage_uri in storage_uris:
        try:
            storage.delete_file(storage_uri)
        except ValueError:
            continue

    for relative_dir in (f"assets/{project_id}", f"exports/{project_id}"):
        try:
            storage.delete_tree(relative_dir)
        except ValueError:
            continue

    return Response(status_code=status.HTTP_204_NO_CONTENT)
