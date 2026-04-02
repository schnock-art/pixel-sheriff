from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseJob:
    job_id: str
    job_version: str
    job_type: str
    attempt: int
    project_id: str
    experiment_id: str
    model_id: str
    task: str
    task_id: str | None
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    dataset_export: dict[str, Any]


@dataclass(frozen=True)
class TrainJob(BaseJob):
    pass


@dataclass(frozen=True)
class EvaluateVariantJob(BaseJob):
    variant_key: str


@dataclass(frozen=True)
class QuantizePtqJob(BaseJob):
    variant_key: str
    checkpoint_kind: str | None
    calibration_max_samples: int = 256


@dataclass(frozen=True)
class QuantizeFp16Job(BaseJob):
    variant_key: str
    checkpoint_kind: str | None


@dataclass(frozen=True)
class QuantizeQatJob(BaseJob):
    variant_key: str
    checkpoint_kind: str | None
    epochs_override: int | None = None
    learning_rate_override: float | None = None
    calibration_max_samples: int = 256


ExperimentJob = TrainJob | EvaluateVariantJob | QuantizePtqJob | QuantizeFp16Job | QuantizeQatJob


def _as_dict(raw_payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be an object")
    return payload


def _parse_base_job(payload: dict[str, Any]) -> dict[str, Any]:
    required = [
        "job_id",
        "job_version",
        "job_type",
        "attempt",
        "project_id",
        "experiment_id",
        "model_id",
        "task",
        "model_config",
        "training_config",
        "dataset_export",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing job fields: {', '.join(missing)}")

    attempt = int(payload["attempt"])
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    model_config = payload["model_config"]
    training_config = payload["training_config"]
    dataset_export = payload["dataset_export"]
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be an object")
    if not isinstance(training_config, dict):
        raise ValueError("training_config must be an object")
    if not isinstance(dataset_export, dict):
        raise ValueError("dataset_export must be an object")

    return {
        "job_id": str(payload["job_id"]),
        "job_version": str(payload["job_version"]),
        "job_type": str(payload["job_type"]),
        "attempt": attempt,
        "project_id": str(payload["project_id"]),
        "experiment_id": str(payload["experiment_id"]),
        "model_id": str(payload["model_id"]),
        "task": str(payload["task"]),
        "task_id": str(payload["task_id"]) if payload.get("task_id") is not None else None,
        "model_config": model_config,
        "training_config": training_config,
        "dataset_export": dataset_export,
    }


def parse_job(raw_payload: str | dict[str, Any]) -> ExperimentJob:
    payload = _as_dict(raw_payload)
    base = _parse_base_job(payload)
    job_type = base["job_type"]
    if job_type == "train":
        return TrainJob(**base)
    if job_type == "evaluate_variant":
        variant_key = str(payload.get("variant_key") or "").strip().lower()
        if not variant_key:
            raise ValueError("variant_key is required for evaluate_variant jobs")
        return EvaluateVariantJob(variant_key=variant_key, **base)
    if job_type == "quantize_ptq":
        variant_key = str(payload.get("variant_key") or "").strip().lower() or "ptq_int8"
        calibration_max_samples = int(payload.get("calibration_max_samples") or 256)
        return QuantizePtqJob(
            variant_key=variant_key,
            checkpoint_kind=str(payload.get("checkpoint_kind")) if payload.get("checkpoint_kind") is not None else None,
            calibration_max_samples=max(1, calibration_max_samples),
            **base,
        )
    if job_type == "quantize_fp16":
        variant_key = str(payload.get("variant_key") or "").strip().lower() or "fp16"
        return QuantizeFp16Job(
            variant_key=variant_key,
            checkpoint_kind=str(payload.get("checkpoint_kind")) if payload.get("checkpoint_kind") is not None else None,
            **base,
        )
    if job_type == "quantize_qat":
        variant_key = str(payload.get("variant_key") or "").strip().lower() or "qat_int8"
        epochs_override = payload.get("epochs_override")
        learning_rate_override = payload.get("learning_rate_override")
        calibration_max_samples = int(payload.get("calibration_max_samples") or 256)
        return QuantizeQatJob(
            variant_key=variant_key,
            checkpoint_kind=str(payload.get("checkpoint_kind")) if payload.get("checkpoint_kind") is not None else None,
            epochs_override=int(epochs_override) if isinstance(epochs_override, int) and epochs_override >= 1 else None,
            learning_rate_override=float(learning_rate_override)
            if isinstance(learning_rate_override, (int, float)) and float(learning_rate_override) > 0
            else None,
            calibration_max_samples=max(1, calibration_max_samples),
            **base,
        )
    raise ValueError(f"unsupported_job_type:{job_type}")


def parse_train_job(raw_payload: str | dict[str, Any]) -> TrainJob:
    job = parse_job(raw_payload)
    if not isinstance(job, TrainJob):
        raise ValueError(f"expected train job, got {job.job_type}")
    return job
