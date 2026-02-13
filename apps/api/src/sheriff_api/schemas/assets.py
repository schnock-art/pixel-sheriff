from pydantic import BaseModel

from sheriff_api.db.models import AnnotationStatus, AssetType


class AssetCreate(BaseModel):
    type: AssetType = AssetType.image
    uri: str
    mime_type: str = "image/jpeg"
    width: int | None = None
    height: int | None = None
    checksum: str
    metadata_json: dict = {}


class AssetRead(BaseModel):
    id: str
    project_id: str
    type: AssetType
    uri: str
    mime_type: str
    width: int | None
    height: int | None
    checksum: str
    metadata_json: dict

    class Config:
        from_attributes = True


class AssetListFilters(BaseModel):
    status: AnnotationStatus | None = None
