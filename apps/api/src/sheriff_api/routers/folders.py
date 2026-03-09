from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, AssetSequence, Folder, Project, Suggestion
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.folders import FolderRead
from sheriff_api.services.sequences import folder_to_read
from sheriff_api.services.storage import LocalStorage
from sheriff_api.config import get_settings

router = APIRouter(tags=["folders"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)


@router.get("/projects/{project_id}/folders", response_model=list[FolderRead])
async def list_folders(project_id: str, db: AsyncSession = Depends(get_db)) -> list[FolderRead]:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status.HTTP_404_NOT_FOUND, code="project_not_found", message="Project not found")

    folders = list((await db.execute(select(Folder).where(Folder.project_id == project_id))).scalars().all())
    folders.sort(key=lambda folder: (folder.path.count("/"), folder.path))
    assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id))).scalars().all())
    sequences = list((await db.execute(select(AssetSequence).where(AssetSequence.project_id == project_id))).scalars().all())
    asset_count_by_folder: dict[str, int] = {}
    for asset in assets:
        if asset.folder_id:
            asset_count_by_folder[asset.folder_id] = asset_count_by_folder.get(asset.folder_id, 0) + 1
    sequence_by_folder = {
        sequence.folder_id: sequence
        for sequence in sequences
        if sequence.folder_id
    }
    return [
        folder_to_read(
            folder,
            asset_count=asset_count_by_folder.get(folder.id, 0),
            sequence=sequence_by_folder.get(folder.id),
        )
        for folder in folders
    ]


@router.delete("/projects/{project_id}/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(project_id: str, folder_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status.HTTP_404_NOT_FOUND, code="project_not_found", message="Project not found")

    folder = await db.get(Folder, folder_id)
    if folder is None or folder.project_id != project_id:
        raise api_error(status.HTTP_404_NOT_FOUND, code="folder_not_found", message="Folder not found")

    all_folders = list((await db.execute(select(Folder).where(Folder.project_id == project_id))).scalars().all())
    descendant_folders = [
        item
        for item in all_folders
        if item.path == folder.path or item.path.startswith(f"{folder.path}/")
    ]
    descendant_ids = [item.id for item in descendant_folders]

    assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id, Asset.folder_id.in_(descendant_ids)))).scalars().all())
    sequences = list(
        (await db.execute(select(AssetSequence).where(AssetSequence.project_id == project_id, AssetSequence.folder_id.in_(descendant_ids)))).scalars().all()
    )
    asset_ids = [asset.id for asset in assets]
    storage_uris = [
        asset.metadata_json.get("storage_uri")
        for asset in assets
        if isinstance(asset.metadata_json, dict) and isinstance(asset.metadata_json.get("storage_uri"), str)
    ]

    if asset_ids:
        await db.execute(delete(Annotation).where(Annotation.asset_id.in_(asset_ids)))
        await db.execute(delete(Suggestion).where(Suggestion.asset_id.in_(asset_ids)))
        await db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
    if sequences:
        await db.execute(delete(AssetSequence).where(AssetSequence.id.in_([sequence.id for sequence in sequences])))
    await db.execute(delete(Folder).where(Folder.id.in_(descendant_ids)))
    await db.commit()

    for storage_uri in storage_uris:
        try:
            storage.delete_file(storage_uri)
        except ValueError:
            continue

    return Response(status_code=status.HTTP_204_NO_CONTENT)
