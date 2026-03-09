from __future__ import annotations

from pydantic import BaseModel, Field


class SequenceFrameAssetRead(BaseModel):
    id: str
    file_name: str
    folder_id: str | None = None
    folder_path: str | None = None
    relative_path: str
    source_kind: str
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    image_url: str
    thumbnail_url: str
    has_annotations: bool = False


class AssetSequenceRead(BaseModel):
    id: str
    project_id: str
    task_id: str | None
    folder_id: str | None
    folder_path: str | None = None
    name: str
    source_type: str
    source_filename: str | None = None
    status: str
    frame_count: int
    processed_frames: int
    fps: float | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    error_message: str | None = None
    assets: list[SequenceFrameAssetRead] = Field(default_factory=list)


class AssetSequenceListResponse(BaseModel):
    items: list[AssetSequenceRead]


class SequenceStatusRead(BaseModel):
    id: str
    status: str
    frame_count: int
    processed_frames: int
    error_message: str | None = None


class WebcamSessionCreate(BaseModel):
    task_id: str | None = None
    folder_id: str | None = None
    name: str
    fps: float = 2.0


class WebcamSessionCreateResponse(BaseModel):
    sequence: AssetSequenceRead
