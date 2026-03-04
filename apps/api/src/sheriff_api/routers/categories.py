from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Category, Task
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.categories import CategoryCreate, CategoryRead, CategoryUpdate
from sheriff_api.services.dataset_store import DatasetStore

router = APIRouter(tags=["categories"])
settings = get_settings()
dataset_store = DatasetStore(settings.storage_root)


def _task_has_dataset_versions(project_id: str, task_id: str) -> bool:
    listed = dataset_store.list_versions(project_id, task_id=task_id)
    items = listed.get("items")
    return isinstance(items, list) and len(items) > 0


def _payload_references_category(payload_json: Any, category_id: str) -> bool:
    if not isinstance(payload_json, dict):
        return False

    top_level_category_id = payload_json.get("category_id")
    if isinstance(top_level_category_id, str) and top_level_category_id == category_id:
        return True

    top_level_category_ids = payload_json.get("category_ids")
    if isinstance(top_level_category_ids, list):
        for value in top_level_category_ids:
            if isinstance(value, str) and value == category_id:
                return True

    classification = payload_json.get("classification")
    if isinstance(classification, dict):
        primary = classification.get("primary_category_id")
        if isinstance(primary, str) and primary == category_id:
            return True
        category_ids = classification.get("category_ids")
        if isinstance(category_ids, list):
            for value in category_ids:
                if isinstance(value, str) and value == category_id:
                    return True

    objects = payload_json.get("objects")
    if isinstance(objects, list):
        for row in objects:
            if not isinstance(row, dict):
                continue
            object_category = row.get("category_id")
            if isinstance(object_category, str) and object_category == category_id:
                return True

    return False


@router.post("/projects/{project_id}/categories", response_model=CategoryRead)
async def create_category(project_id: str, payload: CategoryCreate, db: AsyncSession = Depends(get_db)) -> Category:
    task = await db.get(Task, payload.task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": payload.task_id},
        )
    if _task_has_dataset_versions(project_id, payload.task_id):
        raise api_error(
            status_code=409,
            code="task_locked_by_dataset",
            message="Task labels are locked because dataset versions already exist",
            details={"project_id": project_id, "task_id": payload.task_id},
        )

    category = Category(project_id=project_id, task_id=payload.task_id, name=payload.name, display_order=payload.display_order)
    db.add(category)
    await db.commit()
    await db.refresh(category)
    return category


@router.get("/projects/{project_id}/categories", response_model=list[CategoryRead])
async def list_categories(project_id: str, task_id: str, db: AsyncSession = Depends(get_db)) -> list[Category]:
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )
    result = await db.execute(
        select(Category).where(Category.project_id == project_id, Category.task_id == task_id).order_by(Category.display_order)
    )
    return list(result.scalars().all())


@router.patch("/categories/{category_id}", response_model=CategoryRead)
async def patch_category(category_id: str, payload: CategoryUpdate, db: AsyncSession = Depends(get_db)) -> Category:
    category = await db.get(Category, category_id)
    if not category:
        raise api_error(
            status_code=404,
            code="category_not_found",
            message="Category not found",
            details={"category_id": category_id},
        )
    if _task_has_dataset_versions(category.project_id, category.task_id):
        raise api_error(
            status_code=409,
            code="task_locked_by_dataset",
            message="Task labels are locked because dataset versions already exist",
            details={"project_id": category.project_id, "task_id": category.task_id, "category_id": category_id},
        )
    for field in ["name", "display_order", "is_active"]:
        value = getattr(payload, field)
        if value is not None:
            setattr(category, field, value)
    await db.commit()
    await db.refresh(category)
    return category


@router.delete("/categories/{category_id}")
async def delete_category(category_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    category = await db.get(Category, category_id)
    if category is None:
        raise api_error(
            status_code=404,
            code="category_not_found",
            message="Category not found",
            details={"category_id": category_id},
        )

    if _task_has_dataset_versions(category.project_id, category.task_id):
        raise api_error(
            status_code=409,
            code="task_locked_by_dataset",
            message="Task labels are locked because dataset versions already exist",
            details={"project_id": category.project_id, "task_id": category.task_id, "category_id": category_id},
        )

    annotations = list(
        (
            await db.execute(
                select(Annotation.payload_json).where(
                    Annotation.project_id == category.project_id,
                    Annotation.task_id == category.task_id,
                )
            )
        ).scalars().all()
    )
    annotation_refs = sum(1 for payload_json in annotations if _payload_references_category(payload_json, category_id))
    if annotation_refs > 0:
        raise api_error(
            status_code=409,
            code="category_in_use",
            message="Category cannot be deleted while annotations reference it",
            details={
                "category_id": category_id,
                "project_id": category.project_id,
                "task_id": category.task_id,
                "annotation_references": annotation_refs,
            },
        )

    await db.delete(category)
    await db.commit()
    return {"ok": True, "category_id": category_id}
