from pydantic import BaseModel


class CategoryCreate(BaseModel):
    name: str
    display_order: int = 0


class CategoryUpdate(BaseModel):
    name: str | None = None
    display_order: int | None = None
    is_active: bool | None = None


class CategoryRead(BaseModel):
    id: int
    project_id: str
    name: str
    display_order: int
    is_active: bool

    class Config:
        from_attributes = True
