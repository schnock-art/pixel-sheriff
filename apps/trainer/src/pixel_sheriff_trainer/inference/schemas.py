from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InferClassificationRequest(BaseModel):
    onnx_relpath: str
    metadata_relpath: str
    asset_relpath: str
    device_preference: Literal["auto", "cuda", "cpu"] = "auto"
    top_k: int = Field(default=5, ge=1, le=100)
    model_key: str | None = None


class InferClassificationWarmupRequest(BaseModel):
    onnx_relpath: str
    metadata_relpath: str
    device_preference: Literal["auto", "cuda", "cpu"] = "auto"
    model_key: str | None = None


class PredictionRow(BaseModel):
    class_index: int = Field(ge=0)
    score: float


class InferClassificationResponse(BaseModel):
    device_selected: Literal["cuda", "cpu"]
    predictions: list[PredictionRow]
    output_dim: int = Field(ge=1)


class InferWarmupResponse(BaseModel):
    device_selected: Literal["cuda", "cpu"]
    warmed: bool = True


# --- Detection ---

class InferDetectionRequest(BaseModel):
    onnx_relpath: str
    metadata_relpath: str
    asset_relpath: str
    device_preference: Literal["auto", "cuda", "cpu"] = "auto"
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    model_key: str | None = None


class DetectionBox(BaseModel):
    class_index: int = Field(ge=0)
    class_name: str = ""
    score: float
    bbox: list[float] = Field(description="[x, y, w, h] in pixel coordinates")


class InferDetectionResponse(BaseModel):
    device_selected: Literal["cuda", "cpu"]
    boxes: list[DetectionBox]


# --- Segmentation ---

class InferSegmentationRequest(BaseModel):
    onnx_relpath: str
    metadata_relpath: str
    asset_relpath: str
    device_preference: Literal["auto", "cuda", "cpu"] = "auto"
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    model_key: str | None = None


class SegmentationObject(BaseModel):
    class_index: int = Field(ge=0)
    class_name: str = ""
    score: float
    polygon: list[list[float]] = Field(description="List of [x, y] points")


class InferSegmentationResponse(BaseModel):
    device_selected: Literal["cuda", "cpu"]
    objects: list[SegmentationObject]
