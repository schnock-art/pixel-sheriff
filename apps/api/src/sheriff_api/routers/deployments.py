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
from sheriff_api.db.models import Asset, Category, TaskType
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.deployments import (
    DeploymentCreate,
    DeploymentCreateResponse,
    DeploymentItem,
    DeploymentListResponse,
    DeploymentPatch,
    PredictRequest,
    PredictResponse,
)
from sheriff_api.services.deployment_store import DeploymentStore
from sheriff_api.services.inference_client import InferenceClient
from sheriff_api.services.storage import LocalStorage

from .experiments.shared import experiment_store, require_project

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


def _require_classification_task(task_type: TaskType) -> None:
    if task_type in {TaskType.classification, TaskType.classification_single}:
        return
    raise api_error(
        status_code=409,
        code="deployment_task_mismatch",
        message="Deployments are supported only for classification projects",
    )


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
    project = await require_project(db, project_id)
    _require_classification_task(project.task_type)

    experiment = experiment_store.get(project_id, payload.source.experiment_id, metrics_limit=1, attempt=payload.source.attempt)
    if experiment is None:
        raise api_error(status_code=404, code="experiment_not_found", message="Experiment not found in project")

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
        task=payload.task,
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
    project = await require_project(db, project_id)
    _require_classification_task(project.task_type)
    deployment = _resolve_deployment_for_predict(project_id, payload.deployment_id)

    asset = await db.get(Asset, payload.asset_id)
    if asset is None or asset.project_id != project_id:
        raise api_error(status_code=404, code="asset_not_found", message="Asset not found in project")

    storage_uri = asset.metadata_json.get("storage_uri") if isinstance(asset.metadata_json, dict) else None
    if not isinstance(storage_uri, str) or not storage_uri:
        raise api_error(status_code=404, code="asset_path_missing", message="Asset file path missing")

    source = deployment.get("source")
    if not isinstance(source, dict):
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment source is invalid")
    metadata_relpath = source.get("metadata_relpath")
    if not isinstance(metadata_relpath, str):
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment metadata path is invalid")

    metadata = _read_json(storage.resolve(metadata_relpath))
    class_ids_raw = metadata.get("class_ids")
    if not isinstance(class_ids_raw, list) or not class_ids_raw:
        raise api_error(
            status_code=409,
            code="deployment_class_mapping_invalid",
            message="Deployment metadata is missing class_ids",
        )
    class_ids = [int(value) for value in class_ids_raw if isinstance(value, int)]
    if not class_ids:
        raise api_error(
            status_code=409,
            code="deployment_class_mapping_invalid",
            message="Deployment metadata class_ids are invalid",
        )

    categories = (
        await db.execute(select(Category.id, Category.name).where(Category.project_id == project_id))
    ).all()
    category_name_by_id = {int(row[0]): str(row[1]) for row in categories}
    if any(class_id not in category_name_by_id for class_id in class_ids):
        raise api_error(
            status_code=409,
            code="deployment_class_mapping_invalid",
            message="Deployment classes do not match project categories",
        )

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
        if not isinstance(class_index, int) or class_index < 0:
            continue
        if class_index >= len(class_ids):
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

    return PredictResponse(
        asset_id=payload.asset_id,
        deployment_id=str(deployment.get("deployment_id")),
        task="classification",
        device_selected=str(infer_response.get("device_selected") or "cpu"),
        predictions=predictions,
        deployment_name=str(deployment.get("name") or ""),
        device_preference=str(deployment.get("device_preference") or "auto"),
    )


@router.post("/projects/{project_id}/deployments/{deployment_id}/warmup")
async def warmup_deployment(project_id: str, deployment_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await require_project(db, project_id)
    _require_classification_task(project.task_type)
    deployment = _resolve_deployment_for_predict(project_id, deployment_id)
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
        response = await inference_client.warmup_classification(infer_payload)
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
