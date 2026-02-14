from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Project
from sheriff_api.db.session import get_db
from sheriff_api.schemas.annotations import AnnotationRead, AnnotationUpsert

router = APIRouter(tags=["annotations"])


@router.post("/projects/{project_id}/annotations", response_model=AnnotationRead)
async def upsert_annotation(project_id: str, payload: AnnotationUpsert, db: AsyncSession = Depends(get_db)) -> Annotation:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    asset = await db.get(Asset, payload.asset_id)
    if asset is None or asset.project_id != project_id:
        raise HTTPException(status_code=404, detail="Asset not found in project")

    result = await db.execute(
        select(Annotation).where(Annotation.project_id == project_id, Annotation.asset_id == payload.asset_id),
    )
    annotation = result.scalar_one_or_none()
    if annotation is None:
        annotation = Annotation(project_id=project_id, **payload.model_dump())
        db.add(annotation)
    else:
        annotation.status = payload.status
        annotation.payload_json = payload.payload_json
        annotation.annotated_by = payload.annotated_by
    await db.commit()
    await db.refresh(annotation)
    return annotation


@router.get("/projects/{project_id}/annotations", response_model=list[AnnotationRead])
async def list_annotations(project_id: str, db: AsyncSession = Depends(get_db)) -> list[Annotation]:
    result = await db.execute(select(Annotation).where(Annotation.project_id == project_id))
    return list(result.scalars().all())
