from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import (
    ExperimentVariantFp16Request,
    ExperimentVariantActionResponse,
    ExperimentVariantPtqRequest,
    ExperimentVariantQatRequest,
    ExperimentVariantSummaryResponse,
    ExperimentVariantsResponse,
)

from .shared import experiment_store, model_store, require_project, train_queue

router = APIRouter()

VARIANT_FP32 = "fp32"
VARIANT_FP16 = "fp16"
VARIANT_PTQ_INT8 = "ptq_int8"
VARIANT_QAT_INT8 = "qat_int8"
VARIANT_KEYS = {VARIANT_FP32, VARIANT_FP16, VARIANT_PTQ_INT8, VARIANT_QAT_INT8}
VARIANT_PREFERRED_ORDER = (VARIANT_QAT_INT8, VARIANT_PTQ_INT8, VARIANT_FP16, VARIANT_FP32)


def _read_json(path: Path | None, default: Any) -> Any:
    if path is None or not path.exists() or not path.is_file():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _variant_task_support(task: str) -> dict[str, Any]:
    normalized = str(task or "").strip().lower()
    return {
        "fp16_supported": normalized in {"classification", "detection"},
        "fp16_reason": None if normalized in {"classification", "detection"} else "FP16 is not supported for this task",
        "ptq_supported": normalized in {"classification", "detection"},
        "qat_supported": normalized in {"classification", "detection"},
        "qat_reason": None if normalized in {"classification", "detection"} else "QAT is not supported for this task",
    }


def _resolved_attempt(current: dict[str, Any]) -> int | None:
    for key in ("last_completed_attempt", "current_run_attempt"):
        value = current.get(key)
        if isinstance(value, int) and value >= 1:
            return value
    return None


def _preferred_variant_key(variants: dict[str, Any]) -> str | None:
    for variant_key in VARIANT_PREFERRED_ORDER:
        row = variants.get(variant_key)
        if isinstance(row, dict) and str(row.get("status") or "") == "ready":
            return variant_key
    for variant_key in VARIANT_PREFERRED_ORDER:
        if variant_key in variants:
            return variant_key
    return None


def _load_variant_index(project_id: str, experiment_id: str, attempt: int) -> dict[str, Any] | None:
    payload = experiment_store.read_variants_index(project_id, experiment_id, attempt=attempt)
    if payload is None:
        return None
    _resolved_attempt, index = payload
    return index if isinstance(index, dict) else None


def _legacy_variant_summary(project_id: str, experiment_id: str, attempt: int) -> dict[str, Any] | None:
    model_path = experiment_store.get_onnx_path(project_id, experiment_id, attempt, file_name="model.onnx")
    metadata_path = experiment_store.get_onnx_path(project_id, experiment_id, attempt, file_name="onnx.metadata.json")
    if not model_path.exists() and not metadata_path.exists():
        return None
    metadata = _read_json(metadata_path if metadata_path.exists() else None, {})
    if not isinstance(metadata, dict):
        metadata = {}

    evaluation = {}
    for split in ("val", "test"):
        if split == "val":
            eval_path = experiment_store.variant_evaluation_path(project_id, experiment_id, attempt, VARIANT_FP32, split)
            legacy_path = experiment_store._run_dir(project_id, experiment_id, attempt) / "evaluation.json"  # type: ignore[attr-defined]
            payload = _read_json(legacy_path if legacy_path.exists() else None, None)
        else:
            eval_path = experiment_store.variant_evaluation_path(project_id, experiment_id, attempt, VARIANT_FP32, split)
            payload = _read_json(None, None)
        if isinstance(payload, dict):
            evaluation[split] = {
                "status": str(payload.get("status") or "ready"),
                "overall": payload.get("overall") if isinstance(payload.get("overall"), dict) else None,
                "relpath": str(eval_path.relative_to(experiment_store._storage.root)).replace("\\", "/"),  # type: ignore[attr-defined]
            }

    model_relpath = None
    metadata_relpath = None
    if model_path.exists():
        model_relpath = str(model_path.relative_to(experiment_store._storage.root)).replace("\\", "/")  # type: ignore[attr-defined]
    if metadata_path.exists():
        metadata_relpath = str(metadata_path.relative_to(experiment_store._storage.root)).replace("\\", "/")  # type: ignore[attr-defined]
    return {
        "variant_key": VARIANT_FP32,
        "label": "FP32",
        "kind": "baseline",
        "attempt": attempt,
        "status": "ready" if model_path.exists() else "failed",
        "preferred": True,
        "error": metadata.get("error"),
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "quantized": False,
        "onnx": {
            "model_relpath": model_relpath,
            "metadata_relpath": metadata_relpath,
            "size_bytes": int(model_path.stat().st_size) if model_path.exists() else None,
        },
        "evaluation": evaluation,
        "benchmark": {},
    }


def _list_variant_rows(project_id: str, experiment_id: str, attempt: int) -> dict[str, Any]:
    index = _load_variant_index(project_id, experiment_id, attempt)
    if isinstance(index, dict):
        variants = index.get("variants")
        if isinstance(variants, dict) and variants:
            preferred_variant_key = _preferred_variant_key(variants)
            for key, row in list(variants.items()):
                if isinstance(row, dict):
                    row["preferred"] = key == preferred_variant_key
            return {
                "attempt": attempt,
                "preferred_variant_key": preferred_variant_key,
                "variants": variants,
            }

    legacy = _legacy_variant_summary(project_id, experiment_id, attempt)
    variants: dict[str, Any] = {}
    preferred_variant_key = None
    if legacy is not None:
        variants[VARIANT_FP32] = legacy
        preferred_variant_key = VARIANT_FP32
    return {
        "attempt": attempt,
        "preferred_variant_key": preferred_variant_key,
        "variants": variants,
    }


def _upsert_variant_row(project_id: str, experiment_id: str, attempt: int, variant_key: str, patch: dict[str, Any]) -> dict[str, Any]:
    current_listing = _list_variant_rows(project_id, experiment_id, attempt)
    variants = current_listing.get("variants")
    if not isinstance(variants, dict):
        variants = {}
    current = variants.get(variant_key)
    if not isinstance(current, dict):
        current = {
            "variant_key": variant_key,
            "label": {"fp32": "FP32", "fp16": "FP16", "ptq_int8": "PTQ INT8", "qat_int8": "QAT INT8"}.get(variant_key, variant_key),
            "kind": {"fp32": "baseline", "fp16": "fp16", "ptq_int8": "ptq", "qat_int8": "qat"}.get(variant_key, "baseline"),
            "attempt": attempt,
            "status": "queued",
            "preferred": False,
        }
    merged = dict(current)
    merged.update(patch)
    merged["attempt"] = attempt
    merged["variant_key"] = variant_key
    merged["updated_at"] = merged.get("updated_at") or patch.get("updated_at")
    variants[variant_key] = merged
    preferred_variant_key = _preferred_variant_key(variants)
    for key, row in list(variants.items()):
        if isinstance(row, dict):
            row["preferred"] = key == preferred_variant_key

    payload = {
        "schema_version": "1",
        "attempt": attempt,
        "preferred_variant_key": preferred_variant_key,
        "variants": variants,
    }
    index_path = experiment_store._run_dir(project_id, experiment_id, attempt) / "variants" / "index.json"  # type: ignore[attr-defined]
    _write_json(index_path, payload)
    status_path = experiment_store.variant_status_path(project_id, experiment_id, attempt, variant_key)
    _write_json(status_path, merged)
    return merged


def _require_variant_attempt(current: dict[str, Any], *, project_id: str, experiment_id: str) -> int:
    attempt = _resolved_attempt(current)
    if not isinstance(attempt, int) or attempt < 1:
        attempt = experiment_store.latest_attempt_with_onnx(project_id, experiment_id)
    if not isinstance(attempt, int) or attempt < 1:
        raise api_error(
            status_code=404,
            code="onnx_not_found",
            message="ONNX export not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return attempt


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/variants",
    response_model=ExperimentVariantsResponse,
)
async def get_project_experiment_variants(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentVariantsResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt = _require_variant_attempt(current, project_id=project_id, experiment_id=experiment_id)
    config_json = current.get("config_json")
    task = str(config_json.get("task") or "classification") if isinstance(config_json, dict) else "classification"
    listing = _list_variant_rows(project_id, experiment_id, attempt)
    variants = {
        key: ExperimentVariantSummaryResponse.model_validate(value)
        for key, value in listing["variants"].items()
        if key in VARIANT_KEYS and isinstance(value, dict)
    }
    return ExperimentVariantsResponse(
        attempt=attempt,
        preferred_variant_key=listing.get("preferred_variant_key"),
        support=_variant_task_support(task),
        variants=variants,
    )


async def _enqueue_variant_job(
    *,
    project_id: str,
    experiment_id: str,
    current: dict[str, Any],
    attempt: int,
    variant_key: str,
    job_type: str,
    extra: dict[str, Any] | None = None,
) -> ExperimentVariantActionResponse:
    model_id = str(current.get("model_id") or "")
    model_record = model_store.get(project_id, model_id)
    if model_record is None:
        raise api_error(
            status_code=404,
            code="model_not_found",
            message="Model not found in project",
            details={"project_id": project_id, "model_id": model_id},
        )
    config_json = current.get("config_json")
    if not isinstance(config_json, dict):
        raise api_error(status_code=422, code="validation_error", message="Experiment config is missing")
    model_config = model_record.get("config_json")
    if not isinstance(model_config, dict):
        raise api_error(status_code=422, code="model_config_invalid", message="Model config is not available")
    task = str(config_json.get("task") or "classification")
    task_id = str(current.get("task_id") or config_json.get("task_id") or "") or None
    artifacts = current.get("artifacts_json")
    dataset_export = artifacts.get("last_dataset_export") if isinstance(artifacts, dict) else None
    if not isinstance(dataset_export, dict):
        dataset_export = experiment_store.run_metadata(project_id=project_id, experiment_id=experiment_id, attempt=attempt).get("dataset_export")
    if not isinstance(dataset_export, dict):
        raise api_error(status_code=409, code="dataset_export_missing", message="Dataset export is not available for this run")

    job_id = str(uuid.uuid4())
    checkpoint_kind = None
    if isinstance(artifacts, dict) and isinstance(artifacts.get("selected_checkpoint_kind"), str):
        checkpoint_kind = str(artifacts.get("selected_checkpoint_kind"))

    _upsert_variant_row(
        project_id,
        experiment_id,
        attempt,
        variant_key,
        {"status": "queued", "error": None, "checkpoint_kind": checkpoint_kind},
    )
    experiment_store.append_event(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        event={"type": "variant_status", "variant_key": variant_key, "status": "queued", "attempt": attempt},
    )

    payload = {
        "job_version": "1",
        "job_id": job_id,
        "job_type": job_type,
        "attempt": attempt,
        "project_id": project_id,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "task": task,
        "task_id": task_id,
        "model_config": model_config,
        "training_config": config_json,
        "dataset_export": dataset_export,
        "variant_key": variant_key,
        "checkpoint_kind": checkpoint_kind,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    try:
        await train_queue.enqueue_job(payload)
    except Exception as exc:
        _upsert_variant_row(project_id, experiment_id, attempt, variant_key, {"status": "failed", "error": str(exc)})
        raise api_error(
            status_code=503,
            code="train_queue_unavailable",
            message="Training queue is unavailable",
            details={"project_id": project_id, "experiment_id": experiment_id, "variant_key": variant_key},
        ) from exc

    return ExperimentVariantActionResponse(
        ok=True,
        attempt=attempt,
        variant_key=variant_key,
        status="queued",
        job_id=job_id,
    )


@router.post(
    "/projects/{project_id}/experiments/{experiment_id}/variants/ptq",
    response_model=ExperimentVariantActionResponse,
)
async def trigger_project_experiment_ptq(
    project_id: str,
    experiment_id: str,
    payload: ExperimentVariantPtqRequest = Body(default=ExperimentVariantPtqRequest()),
    db: AsyncSession = Depends(get_db),
) -> ExperimentVariantActionResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(status_code=404, code="experiment_not_found", message="Experiment not found in project")
    config_json = current.get("config_json")
    task = str(config_json.get("task") or "classification") if isinstance(config_json, dict) else "classification"
    support = _variant_task_support(task)
    if not support["ptq_supported"]:
        raise api_error(status_code=409, code="ptq_unsupported", message="PTQ is not supported for this task")
    attempt = _require_variant_attempt(current, project_id=project_id, experiment_id=experiment_id)
    return await _enqueue_variant_job(
        project_id=project_id,
        experiment_id=experiment_id,
        current=current,
        attempt=attempt,
        variant_key=VARIANT_PTQ_INT8,
        job_type="quantize_ptq",
        extra={"calibration_max_samples": int(payload.calibration_max_samples)},
    )


@router.post(
    "/projects/{project_id}/experiments/{experiment_id}/variants/fp16",
    response_model=ExperimentVariantActionResponse,
)
async def trigger_project_experiment_fp16(
    project_id: str,
    experiment_id: str,
    payload: ExperimentVariantFp16Request = Body(default=ExperimentVariantFp16Request()),
    db: AsyncSession = Depends(get_db),
) -> ExperimentVariantActionResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(status_code=404, code="experiment_not_found", message="Experiment not found in project")
    config_json = current.get("config_json")
    task = str(config_json.get("task") or "classification") if isinstance(config_json, dict) else "classification"
    support = _variant_task_support(task)
    if not support["fp16_supported"]:
        raise api_error(
            status_code=409,
            code="fp16_unsupported",
            message=str(support["fp16_reason"] or "FP16 is not supported for this task"),
        )
    attempt = _require_variant_attempt(current, project_id=project_id, experiment_id=experiment_id)
    return await _enqueue_variant_job(
        project_id=project_id,
        experiment_id=experiment_id,
        current=current,
        attempt=attempt,
        variant_key=VARIANT_FP16,
        job_type="quantize_fp16",
        extra={"checkpoint_kind": payload.checkpoint_kind} if payload.checkpoint_kind is not None else {},
    )


@router.post(
    "/projects/{project_id}/experiments/{experiment_id}/variants/qat",
    response_model=ExperimentVariantActionResponse,
)
async def trigger_project_experiment_qat(
    project_id: str,
    experiment_id: str,
    payload: ExperimentVariantQatRequest = Body(default=ExperimentVariantQatRequest()),
    db: AsyncSession = Depends(get_db),
) -> ExperimentVariantActionResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(status_code=404, code="experiment_not_found", message="Experiment not found in project")
    config_json = current.get("config_json")
    task = str(config_json.get("task") or "classification") if isinstance(config_json, dict) else "classification"
    support = _variant_task_support(task)
    if not support["qat_supported"]:
        raise api_error(
            status_code=409,
            code="qat_unsupported",
            message=str(support["qat_reason"] or "QAT is not supported for this task"),
        )
    attempt = _require_variant_attempt(current, project_id=project_id, experiment_id=experiment_id)
    return await _enqueue_variant_job(
        project_id=project_id,
        experiment_id=experiment_id,
        current=current,
        attempt=attempt,
        variant_key=VARIANT_QAT_INT8,
        job_type="quantize_qat",
        extra={
            "epochs_override": payload.epochs_override,
            "learning_rate_override": payload.learning_rate_override,
            "calibration_max_samples": int(payload.calibration_max_samples),
        },
    )
