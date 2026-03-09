from __future__ import annotations

from pydantic import BaseModel


class FolderRead(BaseModel):
    id: str
    project_id: str
    parent_id: str | None
    name: str
    path: str
    asset_count: int = 0
    sequence_id: str | None = None
    sequence_name: str | None = None
    sequence_source_type: str | None = None
    sequence_status: str | None = None
    sequence_frame_count: int | None = None
    sequence_processed_frames: int | None = None


class FolderListResponse(BaseModel):
    items: list[FolderRead]
