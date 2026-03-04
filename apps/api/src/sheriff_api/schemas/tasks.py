from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from sheriff_api.db.models import TaskKind, TaskLabelMode


class TaskCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: TaskKind
    label_mode: TaskLabelMode | None = None


class TaskRead(BaseModel):
    id: str
    project_id: str
    name: str
    kind: TaskKind
    label_mode: TaskLabelMode | None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_default: bool = False

    class Config:
        from_attributes = True
