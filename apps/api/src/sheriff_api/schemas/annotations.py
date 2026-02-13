from pydantic import BaseModel

from sheriff_api.db.models import AnnotationStatus


class AnnotationUpsert(BaseModel):
    asset_id: str
    status: AnnotationStatus = AnnotationStatus.labeled
    payload_json: dict
    annotated_by: str | None = None


class AnnotationRead(BaseModel):
    id: str
    asset_id: str
    project_id: str
    status: AnnotationStatus
    payload_json: dict
    annotated_by: str | None

    class Config:
        from_attributes = True
