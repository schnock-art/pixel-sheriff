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
