from __future__ import annotations

import asyncio
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import PrelabelSession, Task, TaskKind
from sheriff_api.schemas.prelabels import PrelabelConfigCreate
from sheriff_api.services.prelabel_adapters import PRELABEL_ADAPTER_REGISTRY, PrelabelAdapter
from sheriff_api.services.prelabel_common import deployment_store, inference_client, list_task_categories, normalize_prompts


FLORENCE_WARMUP_RETRY_DELAY_SECONDS = 0.5
FLORENCE_WARMUP_MAX_ATTEMPTS = 2


async def resolve_active_deployment(project_id: str, task_id: str) -> dict[str, Any]:
    listing = deployment_store.list(project_id)
    deployment_id = str(listing.get("active_deployment_id") or "").strip()
    if not deployment_id:
        raise ValueError("active_deployment_not_found")
    deployment = deployment_store.get(project_id, deployment_id)
    if not isinstance(deployment, dict):
        raise ValueError("active_deployment_not_found")
    if str(deployment.get("status") or "").strip().lower() == "archived":
        raise ValueError("active_deployment_not_found")
    if str(deployment.get("task") or "").strip().lower() != "bbox":
        raise ValueError("active_deployment_incompatible")
    deployment_task_id = str(deployment.get("task_id") or "").strip()
    if deployment_task_id and deployment_task_id != task_id:
        raise ValueError("active_deployment_incompatible")
    return deployment


async def resolve_prelabel_source_config(
    db: AsyncSession,
    *,
    project_id: str,
    task: Task,
    config: PrelabelConfigCreate,
) -> dict[str, Any]:
    if task.kind != TaskKind.bbox:
        raise ValueError("task_kind_unsupported")

    prompts = normalize_prompts(config.prompts)
    if config.source_type == "active_deployment":
        deployment = await resolve_active_deployment(project_id, task.id)
        return {
            "source_ref": str(deployment.get("deployment_id")),
            "source_label": str(deployment.get("name") or "").strip() or "Project model",
            "device_preference": str(deployment.get("device_preference") or "").strip() or "auto",
            "prompts": [],
            "deployment": deployment,
        }

    if not prompts:
        categories = await list_task_categories(db, project_id=project_id, task_id=task.id)
        prompts = [category.name for category in categories if isinstance(category.name, str) and category.name.strip()]
    return {
        "source_ref": "microsoft/Florence-2-base-ft",
        "source_label": "Florence-2",
        "device_preference": None,
        "prompts": prompts,
        "deployment": None,
    }


async def create_prelabel_session(
    db: AsyncSession,
    *,
    project_id: str,
    task: Task,
    sequence,
    config: PrelabelConfigCreate,
    live_mode: bool,
) -> PrelabelSession:
    resolved = await resolve_prelabel_source_config(db, project_id=project_id, task=task, config=config)

    session = PrelabelSession(
        project_id=project_id,
        task_id=task.id,
        sequence_id=sequence.id,
        source_type=config.source_type,
        source_ref=str(resolved["source_ref"]),
        prompts_json=list(resolved["prompts"]),
        sampling_mode=config.frame_sampling.mode,
        sampling_value=float(config.frame_sampling.value),
        confidence_threshold=float(config.confidence_threshold),
        max_detections_per_frame=int(config.max_detections_per_frame),
        live_mode=bool(live_mode),
        status="queued",
    )
    db.add(session)
    await db.flush()
    return session


async def warmup_prelabel_source(
    db: AsyncSession,
    *,
    project_id: str,
    task: Task,
    config: PrelabelConfigCreate,
) -> dict[str, Any]:
    resolved = await resolve_prelabel_source_config(db, project_id=project_id, task=task, config=config)
    if config.source_type == "active_deployment":
        deployment = resolved["deployment"]
        source = deployment.get("source") if isinstance(deployment, dict) else None
        if not isinstance(source, dict):
            raise RuntimeError("Deployment source is invalid")
        response = await inference_client.warmup_detection(
            {
                "onnx_relpath": source.get("onnx_relpath"),
                "metadata_relpath": source.get("metadata_relpath"),
                "device_preference": deployment.get("device_preference", "auto"),
                "model_key": deployment.get("model_key"),
            }
        )
        return {
            "ok": True,
            "source_type": config.source_type,
            "source_ref": str(resolved["source_ref"]),
            "source_label": str(resolved["source_label"]),
            "device_selected": str(response.get("device_selected") or "cpu"),
            "device_preference": resolved["device_preference"],
        }

    response = await warmup_florence_source(str(resolved["source_ref"]))
    return {
        "ok": True,
        "source_type": config.source_type,
        "source_ref": str(resolved["source_ref"]),
        "source_label": str(resolved["source_label"]),
        "device_selected": str(response.get("device_selected") or "cpu"),
        "device_preference": None,
    }


async def warmup_florence_source(model_name: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(FLORENCE_WARMUP_MAX_ATTEMPTS):
        try:
            return await inference_client.warmup_florence({"model_name": model_name})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 503 or attempt + 1 >= FLORENCE_WARMUP_MAX_ATTEMPTS:
                raise
            last_error = exc
        except httpx.HTTPError as exc:
            if attempt + 1 >= FLORENCE_WARMUP_MAX_ATTEMPTS:
                raise
            last_error = exc
        await asyncio.sleep(FLORENCE_WARMUP_RETRY_DELAY_SECONDS)
    assert last_error is not None
    raise last_error


async def build_adapter(
    *,
    project_id: str,
    session: PrelabelSession,
) -> PrelabelAdapter:
    if str(session.source_type) == "active_deployment":
        deployment = deployment_store.get(project_id, str(session.source_ref or ""))
        if not isinstance(deployment, dict):
            raise RuntimeError("Active deployment is unavailable")
        source = deployment.get("source")
        if not isinstance(source, dict):
            raise RuntimeError("Deployment source is invalid")
        metadata_relpath = str(source.get("metadata_relpath") or "").strip()
        onnx_relpath = str(source.get("onnx_relpath") or "").strip()
        if not metadata_relpath or not onnx_relpath:
            raise RuntimeError("Deployment artifacts are unavailable")
        adapter_factory = PRELABEL_ADAPTER_REGISTRY.get("active_deployment")
        if adapter_factory is None:
            raise RuntimeError("Prelabel adapter is unavailable")
        return adapter_factory(
            deployment=deployment,
            metadata_relpath=metadata_relpath,
            onnx_relpath=onnx_relpath,
            model_key=str(deployment.get("model_key") or "").strip() or None,
        )
    adapter_factory = PRELABEL_ADAPTER_REGISTRY.get(str(session.source_type))
    if adapter_factory is None:
        raise RuntimeError(f"Unsupported prelabel source: {session.source_type}")
    return adapter_factory(model_name=str(session.source_ref or "microsoft/Florence-2-base-ft"))
