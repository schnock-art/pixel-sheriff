from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Category
from sheriff_api.db.session import get_db
from sheriff_api.schemas.categories import CategoryCreate, CategoryRead, CategoryUpdate

router = APIRouter(tags=["categories"])


@router.post("/projects/{project_id}/categories", response_model=CategoryRead)
async def create_category(project_id: str, payload: CategoryCreate, db: AsyncSession = Depends(get_db)) -> Category:
    category = Category(project_id=project_id, name=payload.name, display_order=payload.display_order)
    db.add(category)
    await db.commit()
    await db.refresh(category)
    return category


@router.get("/projects/{project_id}/categories", response_model=list[CategoryRead])
async def list_categories(project_id: str, db: AsyncSession = Depends(get_db)) -> list[Category]:
    result = await db.execute(select(Category).where(Category.project_id == project_id).order_by(Category.display_order))
    return list(result.scalars().all())


@router.patch("/categories/{category_id}", response_model=CategoryRead)
async def patch_category(category_id: int, payload: CategoryUpdate, db: AsyncSession = Depends(get_db)) -> Category:
    category = await db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    for field in ["name", "display_order", "is_active"]:
        value = getattr(payload, field)
        if value is not None:
            setattr(category, field, value)
    await db.commit()
    await db.refresh(category)
    return category
