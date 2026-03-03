from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ExperimentStatus = Literal["draft", "queued", "running", "completed", "failed", "canceled"]
TrainingTask = Literal["classification", "detection", "segmentation"]


class TrainingOptimizer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class TrainingScheduler(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["none", "step", "cosine"] = "none"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingAdvanced(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 1337
    num_workers: int = Field(default=0, ge=0)
    grad_clip_norm: float | None = None


class TrainingBatching(BaseModel):
    model_config = ConfigDict(extra="allow")

    drop_last: bool = True


class TrainingEvaluation(BaseModel):
    model_config = ConfigDict(extra="allow")

    eval_interval_epochs: int = Field(default=1, ge=1)


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(extra="allow")

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool | None = None
    persistent_workers: bool | None = None


class TrainingLogging(BaseModel):
    model_config = ConfigDict(extra="allow")

    save_every_epochs: int = Field(default=1, ge=1)
    keep_last: int = Field(default=1, ge=1)
    keep_best: bool = True


class TrainingResume(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    checkpoint_kind: Literal["latest", "best_loss", "best_metric"] = "latest"


class TrainingHpoBudget(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_trials: int = Field(default=10, ge=1)


class TrainingHpo(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    strategy: Literal["random", "grid", "bayes"] = "random"
    budget: TrainingHpoBudget = Field(default_factory=TrainingHpoBudget)
    search_space: dict[str, Any] = Field(default_factory=dict)


class TrainingConfigV0(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["0.1"] = "0.1"
    model_id: str
    dataset_version_id: str
    task: TrainingTask
    optimizer: TrainingOptimizer = Field(default_factory=TrainingOptimizer)
    scheduler: TrainingScheduler = Field(default_factory=TrainingScheduler)
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=16, ge=1)
    augmentation_profile: Literal["none", "light", "medium", "heavy"] = "light"
    precision: Literal["fp32", "amp"] = "fp32"
    advanced: TrainingAdvanced = Field(default_factory=TrainingAdvanced)
    training: TrainingBatching = Field(default_factory=TrainingBatching)
    evaluation: TrainingEvaluation = Field(default_factory=TrainingEvaluation)
    runtime: TrainingRuntime = Field(default_factory=TrainingRuntime)
    logging: TrainingLogging = Field(default_factory=TrainingLogging)
    resume: TrainingResume = Field(default_factory=TrainingResume)
    hpo: TrainingHpo = Field(default_factory=TrainingHpo)


class ExperimentSummaryJson(BaseModel):
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    best_epoch: int | None = None
    last_epoch: int | None = None


class ExperimentMetricPoint(BaseModel):
    attempt: int | None = Field(default=None, ge=1)
    epoch: int = Field(ge=1)
    train_loss: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None
    val_macro_f1: float | None = None
    val_macro_precision: float | None = None
    val_macro_recall: float | None = None
    val_map: float | None = None
    val_iou: float | None = None
    created_at: datetime | None = None


class ExperimentCheckpoint(BaseModel):
    kind: Literal["best_loss", "best_metric", "latest"]
    epoch: int | None = None
    metric_name: str | None = None
    value: float | None = None
    uri: str | None = None
    updated_at: datetime | None = None
    status: Literal["pending", "ok", "error"] | None = None
    error: str | None = None


class ProjectExperimentSummary(BaseModel):
    id: str
    project_id: str
    model_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    status: ExperimentStatus
    summary_json: ExperimentSummaryJson = Field(default_factory=ExperimentSummaryJson)
    current_run_attempt: int | None = None
    last_completed_attempt: int | None = None
    active_job_id: str | None = None
    error: str | None = None


class ProjectExperimentListResponse(BaseModel):
    items: list[ProjectExperimentSummary]


class ProjectExperimentRecord(ProjectExperimentSummary):
    config_json: dict[str, Any]
    artifacts_json: dict[str, Any] = Field(default_factory=dict)
    checkpoints: list[ExperimentCheckpoint] = Field(default_factory=list)
    metrics: list[ExperimentMetricPoint] = Field(default_factory=list)


class ProjectExperimentCreate(BaseModel):
    model_id: str
    name: str | None = None
    config_overrides: dict[str, Any] | None = None


class ProjectExperimentUpdate(BaseModel):
    name: str | None = None
    config_json: dict[str, Any] | None = None
    selected_checkpoint_kind: Literal["best_loss", "best_metric", "latest"] | None = None


class ProjectExperimentActionResponse(BaseModel):
    ok: bool = True
    status: ExperimentStatus | None = None
    attempt: int | None = None
    job_id: str | None = None


class ExperimentAnalyticsBest(BaseModel):
    metric_name: str | None = None
    metric_value: float | None = None
    epoch: int | None = None


class ExperimentAnalyticsItem(BaseModel):
    experiment_id: str
    name: str
    model_id: str
    model_name: str
    status: ExperimentStatus
    updated_at: datetime
    config: dict[str, Any] = Field(default_factory=dict)
    best: ExperimentAnalyticsBest = Field(default_factory=ExperimentAnalyticsBest)
    final: dict[str, float | None] = Field(default_factory=dict)
    series: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] | None = None


class ProjectExperimentAnalyticsResponse(BaseModel):
    items: list[ExperimentAnalyticsItem] = Field(default_factory=list)
    available_series: list[str] = Field(default_factory=list)


class ExperimentEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    attempt: int = Field(ge=1)


class ExperimentSampleItem(BaseModel):
    asset_id: str
    relative_path: str = ""
    true_class_index: int
    pred_class_index: int
    confidence: float
    margin: float | None = None


class ExperimentSamplesResponse(BaseModel):
    attempt: int = Field(ge=1)
    mode: Literal["misclassified", "lowest_confidence_correct", "highest_confidence_wrong"]
    items: list[ExperimentSampleItem] = Field(default_factory=list)
    message: str | None = None


class ExperimentRuntimeResponse(BaseModel):
    attempt: int = Field(ge=1)
    device_selected: str
    cuda_available: bool
    mps_available: bool
    amp_enabled: bool
    torch_version: str
    torchvision_version: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ExperimentLogsChunkResponse(BaseModel):
    from_byte: int = Field(ge=0)
    to_byte: int = Field(ge=0)
    content: str


class ExperimentOnnxResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    attempt: int = Field(ge=1)
    status: Literal["exported", "failed"]
    model_onnx_url: str | None = None
    metadata_url: str
    input_shape: list[int] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)
    class_order: list[str] = Field(default_factory=list)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] | None = None
    error: str | None = None
