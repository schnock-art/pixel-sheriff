from pydantic import BaseModel

from sheriff_api.db.models import TaskType


class ProjectCreate(BaseModel):
    name: str
    task_type: TaskType = TaskType.classification_single


class ProjectRead(BaseModel):
    id: str
    name: str
    task_type: TaskType
    schema_version: str

    class Config:
        from_attributes = True
