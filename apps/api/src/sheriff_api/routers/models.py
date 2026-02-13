from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Model, Suggestion
from sheriff_api.db.session import get_db
from sheriff_api.schemas.models import ModelCreate, ModelRead

router = APIRouter(tags=["models"])


@router.post("/models", response_model=ModelRead)
async def create_model(payload: ModelCreate, db: AsyncSession = Depends(get_db)) -> Model:
    model = Model(**payload.model_dump())
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


@router.get("/models", response_model=list[ModelRead])
async def list_models(db: AsyncSession = Depends(get_db)) -> list[Model]:
    result = await db.execute(select(Model))
    return list(result.scalars().all())


@router.get("/assets/{asset_id}/suggestions")
async def get_asset_suggestions(asset_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(Suggestion).where(Suggestion.asset_id == asset_id))
    return [{"id": s.id, "model_id": s.model_id, "payload_json": s.payload_json} for s in result.scalars().all()]


@router.post("/projects/{project_id}/suggestions/batch")
async def enqueue_batch_suggestions(project_id: str) -> dict:
    return {"project_id": project_id, "status": "queued"}
