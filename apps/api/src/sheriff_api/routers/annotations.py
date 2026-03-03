from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Category, Project
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.annotations import AnnotationRead, AnnotationUpsert
from sheriff_api.services.annotation_payload import PayloadValidationError, normalize_annotation_payload

router = APIRouter(tags=["annotations"])


@router.post("/projects/{project_id}/annotations", response_model=AnnotationRead)
async def upsert_annotation(project_id: str, payload: AnnotationUpsert, db: AsyncSession = Depends(get_db)) -> Annotation:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    asset = await db.get(Asset, payload.asset_id)
    if asset is None or asset.project_id != project_id:
        raise api_error(status_code=404, code="not_found", message="Asset not found in project")

    category_ids = {
        str(value)
        for value in list((await db.execute(select(Category.id).where(Category.project_id == project_id))).scalars().all())
    }

    try:
        normalized_payload = normalize_annotation_payload(
            payload.payload_json,
            task_type=project.task_type,
            allowed_category_ids=category_ids,
            asset_width=asset.width,
            asset_height=asset.height,
        )
    except PayloadValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    result = await db.execute(
        select(Annotation).where(Annotation.project_id == project_id, Annotation.asset_id == payload.asset_id),
    )
    annotation = result.scalar_one_or_none()
    if annotation is None:
        annotation = Annotation(
            project_id=project_id,
            asset_id=payload.asset_id,
            status=payload.status,
            payload_json=normalized_payload,
            annotated_by=payload.annotated_by,
        )
        db.add(annotation)
    else:
        annotation.status = payload.status
        annotation.payload_json = normalized_payload
        annotation.annotated_by = payload.annotated_by
    await db.commit()
    await db.refresh(annotation)
    return annotation


@router.get("/projects/{project_id}/annotations", response_model=list[AnnotationRead])
async def list_annotations(project_id: str, db: AsyncSession = Depends(get_db)) -> list[Annotation]:
    result = await db.execute(select(Annotation).where(Annotation.project_id == project_id))
    return list(result.scalars().all())
