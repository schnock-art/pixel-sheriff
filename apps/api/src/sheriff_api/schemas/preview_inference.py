from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PreviewInferenceTask = Literal["bbox", "classification"]
PreviewDebugStatus = Literal["matched", "unmatched", "discarded"]


class PreviewInferenceBoxRead(BaseModel):
    class_id: str | None = None
    class_name: str
    score: float
    bbox: list[float] = Field(description="[x, y, w, h] in pixel coordinates")
    matched: bool


class PreviewInferencePredictionRead(BaseModel):
    class_id: str
    class_name: str
    score: float


class PreviewInferenceDebugRead(BaseModel):
    label_text: str
    confidence: float
    bbox: list[float] = Field(description="[x, y, w, h] in pixel coordinates")
    status: PreviewDebugStatus
    category_id: str | None = None
    category_name: str | None = None


class PreviewInferenceResponse(BaseModel):
    task: PreviewInferenceTask
    source_label: str
    device_selected: Literal["cuda", "cpu"] | None = None
    preview_width: int | None = None
    preview_height: int | None = None
    boxes: list[PreviewInferenceBoxRead] = Field(default_factory=list)
    predictions: list[PreviewInferencePredictionRead] = Field(default_factory=list)
    debug: list[PreviewInferenceDebugRead] = Field(default_factory=list)
