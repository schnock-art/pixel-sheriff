from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Category, DatasetVersion
from sheriff_api.db.session import get_db
from sheriff_api.schemas.exports import ExportCreate, ExportRead
from sheriff_api.services.exporter_coco import build_export_result

router = APIRouter(tags=["exports"])


@router.post("/projects/{project_id}/exports", response_model=ExportRead)
async def create_export(project_id: str, payload: ExportCreate, db: AsyncSession = Depends(get_db)) -> DatasetVersion:
    categories = list((await db.execute(select(Category).where(Category.project_id == project_id))).scalars().all())
    assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id))).scalars().all())
    annotations = list((await db.execute(select(Annotation).where(Annotation.project_id == project_id))).scalars().all())

    manifest, content_hash = build_export_result(
        project_id,
        [
            {"id": c.id, "name": c.name, "display_order": c.display_order, "is_active": c.is_active}
            for c in categories
        ],
        [{"id": a.id, "uri": a.uri, "type": a.type.value} for a in assets],
        [{"id": n.id, "asset_id": n.asset_id, "status": n.status.value, "payload": n.payload_json} for n in annotations],
    )

    dataset = DatasetVersion(
        project_id=project_id,
        selection_criteria_json=payload.selection_criteria_json,
        manifest_json=manifest,
        export_uri=f"exports/{project_id}/{content_hash}.zip",
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
