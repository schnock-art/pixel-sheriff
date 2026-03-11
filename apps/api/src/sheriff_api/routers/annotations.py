from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Category, Project, Task
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.annotations import AnnotationRead, AnnotationUpsert
from sheriff_api.services.annotation_payload import PayloadValidationError, normalize_annotation_payload
from sheriff_api.services.prelabels import sync_annotation_prelabel_proposals

router = APIRouter(tags=["annotations"])


@router.post("/projects/{project_id}/annotations", response_model=AnnotationRead)
async def upsert_annotation(project_id: str, payload: AnnotationUpsert, db: AsyncSession = Depends(get_db)) -> Annotation:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")

    task = await db.get(Task, payload.task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": payload.task_id},
        )

    asset = await db.get(Asset, payload.asset_id)
    if asset is None or asset.project_id != project_id:
        raise api_error(status_code=404, code="not_found", message="Asset not found in project")

    category_ids = {
        str(value)
        for value in list(
            (await db.execute(select(Category.id).where(Category.project_id == project_id, Category.task_id == task.id))).scalars().all()
        )
    }

    try:
        normalized_payload = normalize_annotation_payload(
            payload.payload_json,
            task_kind=task.kind,
            label_mode=task.label_mode,
            allowed_category_ids=category_ids,
            asset_width=asset.width,
            asset_height=asset.height,
        )
    except PayloadValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    result = await db.execute(
        select(Annotation).where(
            Annotation.project_id == project_id,
            Annotation.asset_id == payload.asset_id,
            Annotation.task_id == payload.task_id,
        ),
    )
    annotation = result.scalar_one_or_none()
    if annotation is None:
        annotation = Annotation(
            project_id=project_id,
            asset_id=payload.asset_id,
            task_id=payload.task_id,
            status=payload.status,
            payload_json=normalized_payload,
            annotated_by=payload.annotated_by,
        )
        db.add(annotation)
    else:
        annotation.status = payload.status
        annotation.payload_json = normalized_payload
        annotation.annotated_by = payload.annotated_by
    await db.flush()
    await sync_annotation_prelabel_proposals(db, annotation=annotation)
    await db.commit()
    await db.refresh(annotation)
    return annotation


@router.get("/projects/{project_id}/annotations", response_model=list[AnnotationRead])
async def list_annotations(project_id: str, task_id: str, db: AsyncSession = Depends(get_db)) -> list[Annotation]:
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )

    result = await db.execute(select(Annotation).where(Annotation.project_id == project_id, Annotation.task_id == task_id))
    return list(result.scalars().all())
