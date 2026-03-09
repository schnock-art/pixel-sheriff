from __future__ import annotations

from pydantic import BaseModel, Field

from sheriff_api.schemas.sequences import AssetSequenceRead


class VideoImportResponse(BaseModel):
    sequence: AssetSequenceRead


class VideoImportParams(BaseModel):
    task_id: str | None = None
    folder_id: str | None = None
    name: str | None = None
    fps: float = Field(default=2.0, gt=0, le=10)
    max_frames: int = Field(default=500, gt=0, le=5000)
    resize_mode: str = "original"
    resize_width: int | None = Field(default=None, ge=1)
    resize_height: int | None = Field(default=None, ge=1)
