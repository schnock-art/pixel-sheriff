from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Category, DatasetVersion
from sheriff_api.config import get_settings
from sheriff_api.db.session import get_db
from sheriff_api.schemas.exports import ExportCreate, ExportRead
from sheriff_api.services.exporter_coco import build_export_result
from sheriff_api.services.storage import LocalStorage

router = APIRouter(tags=["exports"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)


def _as_status_filter(selection_criteria: dict) -> set[str] | None:
    statuses_raw = selection_criteria.get("statuses")
    if isinstance(statuses_raw, list):
        normalized = {str(value) for value in statuses_raw if str(value).strip()}
        return normalized if normalized else None

    status_raw = selection_criteria.get("status")
    if isinstance(status_raw, str) and status_raw.strip():
        return {status_raw}

    return None


@router.post("/projects/{project_id}/exports", response_model=ExportRead)
async def create_export(project_id: str, payload: ExportCreate, db: AsyncSession = Depends(get_db)) -> DatasetVersion:
    categories = list((await db.execute(select(Category).where(Category.project_id == project_id))).scalars().all())
    all_assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id))).scalars().all())
    all_annotations = list((await db.execute(select(Annotation).where(Annotation.project_id == project_id))).scalars().all())

    status_filter = _as_status_filter(payload.selection_criteria_json)
    if status_filter is None:
        selected_annotations = all_annotations
        selected_assets = all_assets
    else:
        selected_annotations = [annotation for annotation in all_annotations if annotation.status.value in status_filter]
        selected_asset_ids = {annotation.asset_id for annotation in selected_annotations}
        selected_assets = [asset for asset in all_assets if asset.id in selected_asset_ids]

    storage.ensure_project_dirs(project_id)

    manifest, _coco, content_hash, zip_bytes = build_export_result(
        project_id,
        payload.selection_criteria_json,
        [
            {"id": c.id, "name": c.name, "display_order": c.display_order, "is_active": c.is_active}
            for c in categories
        ],
        [
            {
                "id": a.id,
                "uri": a.uri,
                "type": a.type.value,
                "width": a.width,
                "height": a.height,
                "relative_path": a.metadata_json.get("relative_path"),
                "storage_uri": a.metadata_json.get("storage_uri"),
                "extension": Path(str(a.metadata_json.get("storage_uri") or "")).suffix.lower(),
            }
            for a in selected_assets
        ],
        [{"id": n.id, "asset_id": n.asset_id, "status": n.status.value, "payload": n.payload_json} for n in selected_annotations],
        load_asset_bytes=lambda asset: _load_asset_bytes(asset),
    )

    storage_uri = f"exports/{project_id}/{content_hash}.zip"
    zip_path = storage.resolve(storage_uri)
    if not zip_path.exists():
        storage.write_bytes(storage_uri, zip_bytes)

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


def _load_asset_bytes(asset: dict) -> bytes | None:
    storage_uri = asset.get("storage_uri")
    if not isinstance(storage_uri, str) or not storage_uri:
        return None
    try:
        path = storage.resolve(storage_uri)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()


@router.get("/projects/{project_id}/exports/{content_hash}/download")
async def download_export(project_id: str, content_hash: str) -> FileResponse:
    storage_uri = f"exports/{project_id}/{content_hash}.zip"
    try:
        path = storage.resolve(storage_uri)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=path,
        media_type="application/zip",
        filename=f"{project_id}-{content_hash[:12]}.zip",
    )
