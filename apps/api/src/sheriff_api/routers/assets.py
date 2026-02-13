from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset
from sheriff_api.db.session import get_db
from sheriff_api.schemas.assets import AssetCreate, AssetRead

router = APIRouter(tags=["assets"])


@router.post("/projects/{project_id}/assets", response_model=AssetRead)
async def create_asset(project_id: str, payload: AssetCreate, db: AsyncSession = Depends(get_db)) -> Asset:
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
