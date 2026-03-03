from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


DeploymentTask = Literal["classification"]
DeploymentProvider = Literal["onnxruntime"]
DevicePreference = Literal["auto", "cuda", "cpu"]
DeploymentStatus = Literal["available", "archived"]
CheckpointKind = Literal["best_metric", "best_loss", "latest"]


class DeploymentSourceCreate(BaseModel):
    experiment_id: str
    attempt: int = Field(ge=1)
    checkpoint_kind: CheckpointKind = "best_metric"


class DeploymentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    task: DeploymentTask = "classification"
    device_preference: DevicePreference = "auto"
    source: DeploymentSourceCreate
    is_active: bool = False


class DeploymentSourceRead(BaseModel):
    experiment_id: str
    attempt: int = Field(ge=1)
    checkpoint_kind: CheckpointKind
    onnx_relpath: str
    metadata_relpath: str


class DeploymentItem(BaseModel):
    deployment_id: str
    name: str
    task: DeploymentTask
    provider: DeploymentProvider
    device_preference: DevicePreference
    model_key: str
    source: DeploymentSourceRead
    status: DeploymentStatus
    created_at: datetime
    updated_at: datetime


class DeploymentListResponse(BaseModel):
    active_deployment_id: str | None = None
    items: list[DeploymentItem] = Field(default_factory=list)


class DeploymentCreateResponse(BaseModel):
    deployment: DeploymentItem


class DeploymentPatch(BaseModel):
    is_active: bool | None = None
    name: str | None = Field(default=None, min_length=1, max_length=128)
    device_preference: DevicePreference | None = None
    status: DeploymentStatus | None = None


class PredictRequest(BaseModel):
    asset_id: str
    deployment_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)


class PredictPrediction(BaseModel):
    class_index: int = Field(ge=0)
    class_id: str
    class_name: str
    score: float


class PredictResponse(BaseModel):
    asset_id: str
    deployment_id: str
    task: DeploymentTask = "classification"
    device_selected: Literal["cuda", "cpu"]
    predictions: list[PredictPrediction] = Field(default_factory=list)
    deployment_name: str | None = None
    device_preference: DevicePreference | None = None


class InferencePrediction(BaseModel):
    class_index: int = Field(ge=0)
    score: float


class InferenceClassificationResponse(BaseModel):
    device_selected: Literal["cuda", "cpu"]
    predictions: list[InferencePrediction] = Field(default_factory=list)
    output_dim: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None
