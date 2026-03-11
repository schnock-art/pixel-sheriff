from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


PrelabelSourceType = Literal["active_deployment", "florence2"]
PrelabelSamplingMode = Literal["every_n_frames", "every_n_seconds"]
PrelabelSessionStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
PrelabelProposalStatus = Literal["pending", "accepted", "edited", "rejected"]


class PrelabelFrameSampling(BaseModel):
    mode: PrelabelSamplingMode = "every_n_frames"
    value: float = Field(default=15.0, gt=0)


class PrelabelConfigCreate(BaseModel):
    source_type: PrelabelSourceType = "florence2"
    prompts: list[str] = Field(default_factory=list)
    frame_sampling: PrelabelFrameSampling = Field(default_factory=PrelabelFrameSampling)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_detections_per_frame: int = Field(default=20, ge=1, le=200)


class PrelabelSessionCreate(PrelabelConfigCreate):
    sequence_id: str


class PrelabelSessionRead(BaseModel):
    id: str
    project_id: str
    task_id: str
    sequence_id: str
    source_type: PrelabelSourceType
    source_ref: str | None = None
    prompts: list[str] = Field(default_factory=list)
    sampling_mode: PrelabelSamplingMode
    sampling_value: float
    confidence_threshold: float
    max_detections_per_frame: int
    live_mode: bool
    status: PrelabelSessionStatus
    input_closed_at: datetime | None = None
    enqueued_assets: int
    processed_assets: int
    generated_proposals: int
    skipped_unmatched: int
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class PrelabelProposalRead(BaseModel):
    id: str
    session_id: str
    asset_id: str
    project_id: str
    task_id: str
    category_id: str
    label_text: str
    prompt_text: str | None = None
    confidence: float
    bbox: list[float]
    status: PrelabelProposalStatus
    reviewed_bbox: list[float] | None = None
    reviewed_category_id: str | None = None
    promoted_annotation_id: str | None = None
    promoted_object_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class PrelabelProposalListResponse(BaseModel):
    items: list[PrelabelProposalRead]


class PrelabelSessionListResponse(BaseModel):
    items: list[PrelabelSessionRead]


class PrelabelSessionCreateResponse(BaseModel):
    session: PrelabelSessionRead


class PrelabelReviewAction(BaseModel):
    asset_id: str | None = None
    proposal_ids: list[str] = Field(default_factory=list)


class PrelabelReviewResponse(BaseModel):
    session: PrelabelSessionRead
    updated: int = 0
    annotation_ids: list[str] = Field(default_factory=list)


class PrelabelCloseResponse(BaseModel):
    session: PrelabelSessionRead
