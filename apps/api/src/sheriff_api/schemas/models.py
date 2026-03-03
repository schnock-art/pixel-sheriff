from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelCreate(BaseModel):
    name: str
    uri: str


class ModelRead(BaseModel):
    id: str
    name: str
    uri: str

    class Config:
        from_attributes = True


class ProjectModelCreate(BaseModel):
    name: str | None = None


class ProjectModelUpdate(BaseModel):
    config_json: dict[str, Any]


class ProjectModelCreateResponse(BaseModel):
    id: str
    name: str
    config: dict[str, Any]


class ProjectModelSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    task: str
    backbone_name: str
    num_classes: int = Field(ge=0)


class ProjectModelRecord(BaseModel):
    id: str
    project_id: str
    name: str
    config_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime
