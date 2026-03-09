from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Asset, Category
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.deployments import (
    DeploymentCreate,
    DeploymentCreateResponse,
    DeploymentItem,
    DeploymentListResponse,
    DeploymentPatch,
    PredictBBoxResponse,
    PredictClassificationResponse,
    PredictRequest,
    PredictResponse,
)
from sheriff_api.services.deployment_store import DeploymentStore
from sheriff_api.services.inference_client import InferenceClient
from sheriff_api.services.storage import LocalStorage

from .experiments.shared import experiment_store, normalize_task, require_project

router = APIRouter(tags=["deployments"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)
deployment_store = DeploymentStore(settings.storage_root)
inference_client = InferenceClient(
    base_url=settings.trainer_inference_base_url,
    timeout_seconds=float(settings.trainer_inference_timeout_seconds),
)


def _relpath(path: Path) -> str:
    return str(path.relative_to(storage.root.resolve())).replace("\\", "/")


def _deployment_task_from_experiment(task_name: str) -> str:
    normalized = normalize_task(task_name)
    if normalized == "classification":
        return "classification"
    if normalized == "detection":
        return "bbox"
    raise api_error(
        status_code=409,
        code="task_not_supported_for_inference",
        message="Inference is currently supported only for classification and detection tasks",
    )


def _require_supported_deployment(deployment: dict[str, Any]) -> tuple[str, str]:
    task = str(deployment.get("task") or "classification").strip().lower()
    if task not in {"classification", "bbox"}:
        raise api_error(
            status_code=409,
            code="task_not_supported_for_inference",
            message="Inference is currently supported only for classification and detection tasks",
        )
    task_id = str(deployment.get("task_id") or "").strip()
    if not task_id:
        raise api_error(
            status_code=409,
            code="deployment_invalid",
            message="Deployment is missing task_id",
        )
    return task, task_id


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _onnx_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_deployment_metadata(source: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata_relpath = source.get("metadata_relpath")
    if not isinstance(metadata_relpath, str):
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment metadata path is invalid")
    return metadata_relpath, _read_json(storage.resolve(metadata_relpath))


async def _task_categories(project_id: str, task_id: str, db: AsyncSession) -> list[dict[str, str]]:
    categories = (
        await db.execute(
            select(Category.id, Category.name)
            .where(Category.project_id == project_id, Category.task_id == task_id)
            .order_by(Category.display_order, Category.id)
        )
    ).all()
    return [{"id": str(row[0]), "name": str(row[1])} for row in categories]


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for row in value:
        if isinstance(row, str) and row.strip():
            result.append(row.strip())
        elif isinstance(row, int):
            result.append(str(row))
    return result


def _normalize_label_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _map_category_names(values: list[str], *, categories: list[dict[str, str]]) -> list[str] | None:
    if not values:
        return None
    category_ids_by_name: dict[str, list[str]] = {}
    for category in categories:
        name_key = _normalize_label_name(category["name"])
        category_ids_by_name.setdefault(name_key, []).append(category["id"])

    resolved: list[str] = []
    for value in values:
        matches = category_ids_by_name.get(_normalize_label_name(value), [])
        if len(matches) != 1:
            return None
        resolved.append(matches[0])
    return resolved


def _resolve_class_mapping(
    metadata: dict[str, Any],
    *,
    categories: list[dict[str, str]],
) -> list[str]:
    category_name_by_id = {category["id"]: category["name"] for category in categories}

    for key in ("class_ids", "class_order"):
        values = _as_str_list(metadata.get(key))
        if values and all(class_id in category_name_by_id for class_id in values):
            return values

    for key in ("class_ids", "class_order", "class_names"):
        values = _as_str_list(metadata.get(key))
        resolved = _map_category_names(values, categories=categories)
        if resolved is not None:
            return resolved

    raw_class_ids = metadata.get("class_ids")
    raw_class_order = metadata.get("class_order")
    if (
        (isinstance(raw_class_ids, list) and any(isinstance(value, int) for value in raw_class_ids))
        or (isinstance(raw_class_order, list) and any(isinstance(value, int) for value in raw_class_order))
    ):
        raise api_error(
            status_code=409,
            code="deployment_legacy_metadata_incompatible",
            message="Deployment metadata uses legacy integer class_ids. Redeploy from a new dataset version.",
        )

    raise api_error(
        status_code=409,
        code="deployment_class_mapping_invalid",
        message="Deployment classes do not match project categories",
    )


def _resolve_deployment_for_predict(project_id: str, deployment_id: str | None) -> dict[str, Any]:
    listing = deployment_store.list(project_id)
    resolved_id = deployment_id or listing.get("active_deployment_id")
    if not isinstance(resolved_id, str) or not resolved_id:
        raise api_error(status_code=409, code="no_active_deployment", message="No active deployment is configured")
    deployment = deployment_store.get(project_id, resolved_id)
    if deployment is None:
        raise api_error(status_code=404, code="deployment_not_found", message="Deployment not found")
    if str(deployment.get("status")) == "archived":
        raise api_error(status_code=409, code="deployment_archived", message="Deployment is archived")
    return deployment


@router.post("/projects/{project_id}/deployments", response_model=DeploymentCreateResponse)
async def create_deployment(
    project_id: str,
    payload: DeploymentCreate,
    db: AsyncSession = Depends(get_db),
) -> DeploymentCreateResponse:
    await require_project(db, project_id)

    experiment = experiment_store.get(project_id, payload.source.experiment_id, metrics_limit=1, attempt=payload.source.attempt)
    if experiment is None:
        raise api_error(status_code=404, code="experiment_not_found", message="Experiment not found in project")
    experiment_task = "classification"
    config_json = experiment.get("config_json")
    if isinstance(config_json, dict) and isinstance(config_json.get("task"), str):
        experiment_task = str(config_json.get("task"))
    deployment_task = _deployment_task_from_experiment(experiment_task)
    task_id = str(experiment.get("task_id") or "")
    if not task_id and isinstance(config_json, dict):
        task_id = str(config_json.get("task_id") or "")
    if not task_id:
        raise api_error(
            status_code=409,
            code="deployment_invalid",
            message="Experiment is missing task_id",
            details={"project_id": project_id, "experiment_id": payload.source.experiment_id},
        )

    onnx_path = experiment_store.get_onnx_path(project_id, payload.source.experiment_id, payload.source.attempt, file_name="model.onnx")
    metadata_path = experiment_store.get_onnx_path(
        project_id, payload.source.experiment_id, payload.source.attempt, file_name="onnx.metadata.json"
    )
    if not onnx_path.exists() or not metadata_path.exists():
        raise api_error(status_code=404, code="onnx_not_found", message="ONNX export not available for this experiment")

    model_key = _onnx_sha256(onnx_path)
    source = {
        "experiment_id": payload.source.experiment_id,
        "attempt": payload.source.attempt,
        "checkpoint_kind": payload.source.checkpoint_kind,
        "onnx_relpath": _relpath(onnx_path),
        "metadata_relpath": _relpath(metadata_path),
    }
    item = deployment_store.create(
        project_id=project_id,
        name=payload.name.strip(),
        task_id=task_id,
        task=deployment_task,
        device_preference=payload.device_preference,
        source=source,
        model_key=model_key,
        is_active=bool(payload.is_active),
    )
    return DeploymentCreateResponse(deployment=DeploymentItem.model_validate(item))


@router.get("/projects/{project_id}/deployments", response_model=DeploymentListResponse)
async def list_deployments(project_id: str, db: AsyncSession = Depends(get_db)) -> DeploymentListResponse:
    await require_project(db, project_id)
    payload = deployment_store.list(project_id)
    return DeploymentListResponse.model_validate(payload)


@router.patch("/projects/{project_id}/deployments/{deployment_id}", response_model=DeploymentCreateResponse)
async def patch_deployment(
    project_id: str,
    deployment_id: str,
    payload: DeploymentPatch,
    db: AsyncSession = Depends(get_db),
) -> DeploymentCreateResponse:
    await require_project(db, project_id)
    patched = deployment_store.patch(
        project_id=project_id,
        deployment_id=deployment_id,
        name=payload.name.strip() if isinstance(payload.name, str) else None,
        device_preference=payload.device_preference,
        status=payload.status,
        is_active=payload.is_active,
    )
    if patched is None:
        raise api_error(status_code=404, code="deployment_not_found", message="Deployment not found")
    return DeploymentCreateResponse(deployment=DeploymentItem.model_validate(patched))


@router.post("/projects/{project_id}/predict", response_model=PredictResponse)
async def predict_classification(
    project_id: str,
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    await require_project(db, project_id)
    deployment = _resolve_deployment_for_predict(project_id, payload.deployment_id)
    deployment_task, task_id = _require_supported_deployment(deployment)

    asset = await db.get(Asset, payload.asset_id)
    if asset is None or asset.project_id != project_id:
        raise api_error(status_code=404, code="asset_not_found", message="Asset not found in project")

    storage_uri = asset.metadata_json.get("storage_uri") if isinstance(asset.metadata_json, dict) else None
    if not isinstance(storage_uri, str) or not storage_uri:
        raise api_error(status_code=404, code="asset_path_missing", message="Asset file path missing")

    source = deployment.get("source")
    if not isinstance(source, dict):
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment source is invalid")
    metadata_relpath, metadata = _load_deployment_metadata(source)
    categories = await _task_categories(project_id, task_id, db)
    category_name_by_id = {category["id"]: category["name"] for category in categories}
    class_ids = _resolve_class_mapping(metadata, categories=categories)

    if deployment_task == "classification":
        infer_payload = {
            "onnx_relpath": source.get("onnx_relpath"),
            "metadata_relpath": metadata_relpath,
            "asset_relpath": storage_uri,
            "device_preference": deployment.get("device_preference", "auto"),
            "top_k": payload.top_k,
            "model_key": deployment.get("model_key"),
        }
        try:
            infer_response = await inference_client.infer_classification(infer_payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 503:
                raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable") from exc
            raise api_error(status_code=502, code="inference_failed", message="Inference request failed") from exc
        except Exception as exc:
            raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable") from exc

        predictions_raw = infer_response.get("predictions")
        if not isinstance(predictions_raw, list):
            predictions_raw = []
        output_dim = infer_response.get("output_dim")
        if isinstance(output_dim, int) and output_dim > len(class_ids):
            raise api_error(
                status_code=409,
                code="deployment_output_dim_mismatch",
                message="Inference output does not match deployment class_ids",
            )
        max_class_index = max((int(item.get("class_index")) for item in predictions_raw if isinstance(item, dict)), default=-1)
        if max_class_index >= len(class_ids):
            raise api_error(
                status_code=409,
                code="deployment_output_dim_mismatch",
                message="Inference output does not match deployment class_ids",
            )

        predictions: list[dict[str, Any]] = []
        for row in predictions_raw:
            if not isinstance(row, dict):
                continue
            class_index = row.get("class_index")
            score = row.get("score")
            if not isinstance(class_index, int) or class_index < 0 or class_index >= len(class_ids):
                continue
            class_id = class_ids[class_index]
            predictions.append(
                {
                    "class_index": class_index,
                    "class_id": class_id,
                    "class_name": category_name_by_id[class_id],
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                }
            )

        return PredictClassificationResponse(
            asset_id=payload.asset_id,
            deployment_id=str(deployment.get("deployment_id")),
            task="classification",
            device_selected=str(infer_response.get("device_selected") or "cpu"),
            predictions=predictions,
            deployment_name=str(deployment.get("name") or ""),
            device_preference=str(deployment.get("device_preference") or "auto"),
        )

    infer_payload = {
        "onnx_relpath": source.get("onnx_relpath"),
        "metadata_relpath": metadata_relpath,
        "asset_relpath": storage_uri,
        "device_preference": deployment.get("device_preference", "auto"),
        "score_threshold": payload.score_threshold,
        "model_key": deployment.get("model_key"),
    }
    try:
        infer_response = await inference_client.infer_detection(infer_payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable") from exc
        raise api_error(status_code=502, code="inference_failed", message="Inference request failed") from exc
    except Exception as exc:
        raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable") from exc

    boxes_raw = infer_response.get("boxes")
    if not isinstance(boxes_raw, list):
        boxes_raw = []

    boxes: list[dict[str, Any]] = []
    for row in boxes_raw:
        if not isinstance(row, dict):
            continue
        class_index = row.get("class_index")
        score = row.get("score")
        bbox = row.get("bbox")
        if not isinstance(class_index, int) or class_index < 0 or class_index >= len(class_ids):
            raise api_error(
                status_code=409,
                code="deployment_output_dim_mismatch",
                message="Inference output does not match deployment class_ids",
            )
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
            continue
        class_id = class_ids[class_index]
        boxes.append(
            {
                "class_index": class_index,
                "class_id": class_id,
                "class_name": category_name_by_id[class_id],
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
                "bbox": [float(value) for value in bbox],
            }
        )

    return PredictBBoxResponse(
        asset_id=payload.asset_id,
        deployment_id=str(deployment.get("deployment_id")),
        task="bbox",
        device_selected=str(infer_response.get("device_selected") or "cpu"),
        boxes=boxes,
        deployment_name=str(deployment.get("name") or ""),
        device_preference=str(deployment.get("device_preference") or "auto"),
    )


@router.post("/projects/{project_id}/deployments/{deployment_id}/warmup")
async def warmup_deployment(project_id: str, deployment_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await require_project(db, project_id)
    deployment = _resolve_deployment_for_predict(project_id, deployment_id)
    deployment_task, _task_id = _require_supported_deployment(deployment)
    source = deployment.get("source")
    if not isinstance(source, dict):
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment source is invalid")

    infer_payload = {
        "onnx_relpath": source.get("onnx_relpath"),
        "metadata_relpath": source.get("metadata_relpath"),
        "device_preference": deployment.get("device_preference", "auto"),
        "model_key": deployment.get("model_key"),
    }
    try:
        if deployment_task == "classification":
            response = await inference_client.warmup_classification(infer_payload)
        else:
            response = await inference_client.warmup_detection(infer_payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable") from exc
        raise api_error(status_code=502, code="inference_failed", message="Inference request failed") from exc
    except Exception as exc:
        raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable") from exc
    return {
        "ok": True,
        "deployment_id": deployment_id,
        "device_selected": str(response.get("device_selected") or "cpu"),
    }
