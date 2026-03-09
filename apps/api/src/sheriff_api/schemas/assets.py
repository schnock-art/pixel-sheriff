from pydantic import BaseModel, Field

from sheriff_api.db.models import AnnotationStatus, AssetType


class AssetCreate(BaseModel):
    type: AssetType = AssetType.image
    folder_id: str | None = None
    file_name: str | None = None
    sequence_id: str | None = None
    source_kind: str = "image"
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    uri: str
    mime_type: str = "image/jpeg"
    width: int | None = None
    height: int | None = None
    checksum: str
    metadata_json: dict = Field(default_factory=dict)


class AssetRead(BaseModel):
    id: str
    project_id: str
    type: AssetType
    folder_id: str | None = None
    folder_path: str | None = None
    file_name: str | None = None
    relative_path: str | None = None
    sequence_id: str | None = None
    source_kind: str = "image"
    frame_index: int | None = None
    timestamp_seconds: float | None = None
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
