from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainJob:
    job_id: str
    job_version: str
    job_type: str
    attempt: int
    project_id: str
    experiment_id: str
    model_id: str
    task: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    dataset_export: dict[str, Any]


def _as_dict(raw_payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be an object")
    return payload


def parse_train_job(raw_payload: str | dict[str, Any]) -> TrainJob:
    payload = _as_dict(raw_payload)
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

    return TrainJob(
        job_id=str(payload["job_id"]),
        job_version=str(payload["job_version"]),
        job_type=str(payload["job_type"]),
        attempt=attempt,
        project_id=str(payload["project_id"]),
        experiment_id=str(payload["experiment_id"]),
        model_id=str(payload["model_id"]),
        task=str(payload["task"]),
        model_config=model_config,
        training_config=training_config,
        dataset_export=dataset_export,
    )

