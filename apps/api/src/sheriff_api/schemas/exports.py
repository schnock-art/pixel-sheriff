from pydantic import BaseModel, Field


class ExportCreate(BaseModel):
    selection_criteria_json: dict = Field(default_factory=lambda: {"statuses": ["labeled", "approved"]})


class ExportRead(BaseModel):
    id: str
    project_id: str
    selection_criteria_json: dict
    manifest_json: dict
    export_uri: str
    hash: str

    class Config:
        from_attributes = True
