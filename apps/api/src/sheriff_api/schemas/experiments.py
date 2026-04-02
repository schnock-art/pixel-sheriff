from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sheriff_api.services.augmentation import AUGMENTATION_STEP_TYPES, task_default_augmentation_profile


ExperimentStatus = Literal["draft", "queued", "running", "completed", "failed", "canceled"]
TrainingTask = Literal["classification", "detection", "segmentation"]
ModelVariantKey = Literal["fp32", "fp16", "ptq_int8", "qat_int8"]
ModelVariantStatus = Literal["queued", "running", "ready", "failed", "unsupported"]


class TrainingOptimizer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_optimizer(self) -> "TrainingOptimizer":
        if self.type != "sgd" and self.momentum is not None:
            raise ValueError("optimizer momentum is only supported for sgd")
        return self


class TrainingScheduler(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["none", "step", "cosine"] = "none"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scheduler(self) -> "TrainingScheduler":
        if self.type != "step":
            return self
        step_size = self.params.get("step_size")
        gamma = self.params.get("gamma")
        if not isinstance(step_size, int) or isinstance(step_size, bool) or int(step_size) < 1:
            raise ValueError("step scheduler step_size must be an integer >= 1")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or float(gamma) <= 0:
            raise ValueError("step scheduler gamma must be > 0")
        return self


class TrainingAdvanced(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 1337
    num_workers: int = Field(default=0, ge=0)
    grad_clip_norm: float | None = Field(default=None, ge=0)


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
    prefetch_factor: int = Field(default=2, ge=1)
    cache_resized_images: bool | None = None
    max_cached_images: int = Field(default=1024, ge=0)


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


class TrainingAugmentationStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["horizontal_flip", "vertical_flip", "color_jitter", "rotate"]
    p: float = Field(default=1.0, ge=0, le=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params(self) -> "TrainingAugmentationStep":
        if self.type not in set(AUGMENTATION_STEP_TYPES):
            raise ValueError(f"Unsupported augmentation type: {self.type}")

        params = dict(self.params)
        if self.type in {"horizontal_flip", "vertical_flip"}:
            if params:
                raise ValueError(f"{self.type} does not accept params")
            return self

        if self.type == "rotate":
            if set(params.keys()) == {"degrees"}:
                degrees = params.get("degrees")
                if not isinstance(degrees, (int, float)) or isinstance(degrees, bool) or float(degrees) <= 0:
                    raise ValueError("rotate degrees must be > 0")
                degrees_value = float(degrees)
                self.params = {"min_degrees": -degrees_value, "max_degrees": degrees_value}
                return self
            if set(params.keys()) != {"min_degrees", "max_degrees"}:
                raise ValueError("rotate requires degrees or min_degrees/max_degrees")
            min_degrees = params.get("min_degrees")
            max_degrees = params.get("max_degrees")
            if not isinstance(min_degrees, (int, float)) or isinstance(min_degrees, bool):
                raise ValueError("rotate min_degrees must be numeric")
            if not isinstance(max_degrees, (int, float)) or isinstance(max_degrees, bool):
                raise ValueError("rotate max_degrees must be numeric")
            min_value = float(min_degrees)
            max_value = float(max_degrees)
            if max_value < min_value:
                raise ValueError("rotate max_degrees must be >= min_degrees")
            if max_value == min_value:
                raise ValueError("rotate min_degrees and max_degrees must define a non-zero range")
            self.params = {"min_degrees": min_value, "max_degrees": max_value}
            return self

        allowed_params = {"brightness", "contrast", "saturation", "hue"}
        unknown = sorted(set(params.keys()) - allowed_params)
        if unknown:
            raise ValueError(f"color_jitter params are invalid: {', '.join(unknown)}")

        normalized: dict[str, float] = {}
        for key, value in params.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"color_jitter {key} must be numeric")
            parsed = float(value)
            if parsed < 0:
                raise ValueError(f"color_jitter {key} must be >= 0")
            if key == "hue" and parsed > 0.5:
                raise ValueError("color_jitter hue must be <= 0.5")
            normalized[key] = parsed
        self.params = normalized
        return self


class TrainingConfigV0(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["0.1"] = "0.1"
    model_id: str
    dataset_version_id: str
    task_id: str | None = None
    task: TrainingTask
    optimizer: TrainingOptimizer = Field(default_factory=TrainingOptimizer)
    scheduler: TrainingScheduler = Field(default_factory=TrainingScheduler)
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=16, ge=1)
    augmentation_profile: Literal["none", "light", "medium", "heavy", "custom"] = "light"
    augmentation_spec_version: Literal[1] | None = None
    augmentation_steps: list[TrainingAugmentationStep] = Field(default_factory=list)
    precision: Literal["fp32", "amp"] = "fp32"
    advanced: TrainingAdvanced = Field(default_factory=TrainingAdvanced)
    training: TrainingBatching = Field(default_factory=TrainingBatching)
    evaluation: TrainingEvaluation = Field(default_factory=TrainingEvaluation)
    runtime: TrainingRuntime = Field(default_factory=TrainingRuntime)
    logging: TrainingLogging = Field(default_factory=TrainingLogging)
    resume: TrainingResume = Field(default_factory=TrainingResume)
    hpo: TrainingHpo = Field(default_factory=TrainingHpo)

    @model_validator(mode="after")
    def validate_augmentation(self) -> "TrainingConfigV0":
        if "augmentation_profile" not in self.model_fields_set:
            self.augmentation_profile = task_default_augmentation_profile(self.task)  # type: ignore[assignment]
        if self.augmentation_profile == "custom" and not self.augmentation_steps:
            raise ValueError("augmentation_steps must include at least one step when augmentation_profile is custom")
        return self


class ExperimentSummaryJson(BaseModel):
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    best_epoch: int | None = None
    last_epoch: int | None = None


class ExperimentMetricPoint(BaseModel):
    attempt: int | None = Field(default=None, ge=1)
    epoch: int = Field(ge=1)
    train_loss: float | None = None
    train_accuracy: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None
    val_macro_f1: float | None = None
    val_macro_precision: float | None = None
    val_macro_recall: float | None = None
    val_map: float | None = None
    val_map_50_95: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    val_matched_mean_iou: float | None = None
    val_tp: int | None = None
    val_fp: int | None = None
    val_fn: int | None = None
    val_duplicate_fp: int | None = None
    val_iou: float | None = None
    epoch_seconds: float | None = None
    eta_seconds: float | None = None
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
    task_id: str | None = None
    task: str | None = None
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
    dataset_version_id: str | None = None
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
    task_id: str | None = None
    task: str | None = None
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
    prefetch_factor: int | None = None
    cache_resized_images: bool | None = None
    max_cached_images: int | None = None


class ExperimentLogsChunkResponse(BaseModel):
    attempt: int = Field(ge=1)
    from_byte: int = Field(ge=0)
    to_byte: int = Field(ge=0)
    content: str


class ExperimentOnnxResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    attempt: int = Field(ge=1)
    status: Literal["exported", "failed"]
    variant_key: ModelVariantKey | None = None
    preferred_variant_key: ModelVariantKey | None = None
    model_onnx_url: str | None = None
    metadata_url: str
    input_shape: list[int] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)
    class_order: list[str] = Field(default_factory=list)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] | None = None
    error: str | None = None
    available_variants: list[ModelVariantKey] = Field(default_factory=list)


class ExperimentVariantBenchmarkResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str | None = None
    provider: str | None = None
    batch_size: int | None = None
    sample_count: int | None = None
    mean_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    throughput_items_per_second: float | None = None
    message: str | None = None


class ExperimentVariantSplitResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str | None = None
    overall: dict[str, Any] | None = None
    relpath: str | None = None


class ExperimentVariantSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    variant_key: ModelVariantKey
    label: str
    kind: str
    attempt: int = Field(ge=1)
    status: ModelVariantStatus
    preferred: bool = False
    error: str | None = None
    checkpoint_kind: str | None = None
    quantized: bool | None = None
    quantization_strategy: str | None = None
    onnx: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, ExperimentVariantSplitResponse] = Field(default_factory=dict)
    benchmark: ExperimentVariantBenchmarkResponse | dict[str, Any] | None = None
    benchmarks: dict[str, ExperimentVariantBenchmarkResponse] = Field(default_factory=dict)
    metrics: list[ExperimentMetricPoint] = Field(default_factory=list)
    qat: dict[str, Any] | None = None


class ExperimentVariantSupportResponse(BaseModel):
    fp16_supported: bool = False
    fp16_reason: str | None = None
    ptq_supported: bool = False
    qat_supported: bool = False
    qat_reason: str | None = None


class ExperimentVariantsResponse(BaseModel):
    attempt: int = Field(ge=1)
    preferred_variant_key: ModelVariantKey | None = None
    support: ExperimentVariantSupportResponse = Field(default_factory=ExperimentVariantSupportResponse)
    variants: dict[ModelVariantKey, ExperimentVariantSummaryResponse] = Field(default_factory=dict)


class ExperimentVariantActionResponse(BaseModel):
    ok: bool = True
    attempt: int = Field(ge=1)
    variant_key: ModelVariantKey
    status: ModelVariantStatus
    job_id: str | None = None


class ExperimentVariantPtqRequest(BaseModel):
    calibration_max_samples: int = Field(default=256, ge=1)


class ExperimentVariantFp16Request(BaseModel):
    checkpoint_kind: Literal["best_loss", "best_metric", "latest"] | None = None


class ExperimentVariantQatRequest(BaseModel):
    epochs_override: int | None = Field(default=None, ge=1)
    learning_rate_override: float | None = Field(default=None, gt=0)
    calibration_max_samples: int = Field(default=256, ge=1)
