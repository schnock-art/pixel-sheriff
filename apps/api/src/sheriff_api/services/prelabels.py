from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
import httpx
import logging
import math
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sheriff_api.config import get_settings
from sheriff_api.db.models import (
    Annotation,
    AnnotationStatus,
    Asset,
    AssetSequence,
    Category,
    PrelabelProposal,
    PrelabelSession,
    Task,
    TaskKind,
)
from sheriff_api.db.session import SessionLocal
from sheriff_api.schemas.prelabels import (
    PrelabelConfigCreate,
    PrelabelProposalRead,
    PrelabelSessionRead,
)
from sheriff_api.services.annotation_payload import normalize_annotation_payload
from sheriff_api.services.deployment_store import DeploymentStore
from sheriff_api.services.inference_client import InferenceClient
from sheriff_api.services.prelabel_adapters import (
    DetectionResult,
    PrelabelAdapter,
    PRELABEL_ADAPTER_REGISTRY,
)
from sheriff_api.services.prelabel_queue import PrelabelQueue


settings = get_settings()
deployment_store = DeploymentStore(settings.storage_root)
inference_client = InferenceClient(
    base_url=settings.trainer_inference_base_url,
    timeout_seconds=float(settings.trainer_inference_timeout_seconds),
)
logger = logging.getLogger(__name__)
_PRELABEL_DEBUG_DETECTIONS_LIMIT = 200
_FLORENCE_WARMUP_RETRY_DELAY_SECONDS = 0.5
_FLORENCE_WARMUP_MAX_ATTEMPTS = 2

_PRELABEL_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"human", "person", "people"}),
    frozenset({"glass", "glasses", "eyeglasses", "eye glasses", "spectacles", "specs", "eyewear"}),
    frozenset({"eye", "eyes", "eyeball", "eyeballs"}),
    frozenset({"mouth", "mouths", "lip", "lips"}),
    frozenset({"head", "face"}),
)
_PRELABEL_ALIAS_LOOKUP: dict[str, set[str]] = {}
for _group in _PRELABEL_ALIAS_GROUPS:
    for _value in _group:
        _PRELABEL_ALIAS_LOOKUP[_value] = set(_group - {_value})


def utc_now_dt() -> datetime:
    return datetime.utcnow()


def normalize_prompts(raw_prompts: list[str] | None) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for value in raw_prompts or []:
        normalized = " ".join(str(value or "").strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(normalized)
    return prompts


async def list_task_categories(db: AsyncSession, *, project_id: str, task_id: str) -> list[Category]:
    result = await db.execute(
        select(Category)
        .where(Category.project_id == project_id, Category.task_id == task_id)
        .order_by(Category.display_order, Category.id)
    )
    return list(result.scalars().all())


def prelabel_session_to_read(session: PrelabelSession) -> PrelabelSessionRead:
    source_label: str | None = None
    device_preference: str | None = None
    if str(session.source_type) == "active_deployment":
        deployment = deployment_store.get(session.project_id, str(session.source_ref or ""))
        if isinstance(deployment, dict):
            source_label = str(deployment.get("name") or "").strip() or "Project model"
            device_preference = str(deployment.get("device_preference") or "").strip() or None
        else:
            source_label = "Project model"
    else:
        source_label = "Florence-2"
    return PrelabelSessionRead(
        id=session.id,
        project_id=session.project_id,
        task_id=session.task_id,
        sequence_id=session.sequence_id,
        source_type=str(session.source_type),
        source_ref=session.source_ref,
        source_label=source_label,
        device_preference=device_preference,
        prompts=list(session.prompts_json or []),
        sampling_mode=str(session.sampling_mode),
        sampling_value=float(session.sampling_value),
        confidence_threshold=float(session.confidence_threshold),
        max_detections_per_frame=int(session.max_detections_per_frame),
        live_mode=bool(session.live_mode),
        status=str(session.status),
        input_closed_at=session.input_closed_at,
        enqueued_assets=int(session.enqueued_assets or 0),
        processed_assets=int(session.processed_assets or 0),
        generated_proposals=int(session.generated_proposals or 0),
        skipped_unmatched=int(session.skipped_unmatched or 0),
        error_message=session.error_message,
        debug_detections=_normalized_debug_detections(session.debug_detections_json),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def prelabel_proposal_to_read(proposal: PrelabelProposal) -> PrelabelProposalRead:
    return PrelabelProposalRead(
        id=proposal.id,
        session_id=proposal.session_id,
        asset_id=proposal.asset_id,
        project_id=proposal.project_id,
        task_id=proposal.task_id,
        category_id=proposal.category_id,
        label_text=proposal.label_text,
        prompt_text=proposal.prompt_text,
        confidence=float(proposal.confidence or 0.0),
        bbox=[float(value) for value in list(proposal.bbox_json or [])[:4]],
        status=str(proposal.status),
        reviewed_bbox=[float(value) for value in list(proposal.reviewed_bbox_json or [])[:4]]
        if isinstance(proposal.reviewed_bbox_json, list)
        else None,
        reviewed_category_id=proposal.reviewed_category_id,
        promoted_annotation_id=proposal.promoted_annotation_id,
        promoted_object_id=proposal.promoted_object_id,
        created_at=proposal.created_at,
        updated_at=proposal.updated_at,
    )


def sequence_pending_total_from_counts(counts_by_asset: dict[str, int]) -> int:
    return sum(int(value or 0) for value in counts_by_asset.values())


def _normalized_debug_detections(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    detections: list[dict[str, Any]] = []
    for row in raw_rows[:_PRELABEL_DEBUG_DETECTIONS_LIMIT]:
        if not isinstance(row, dict):
            continue
        asset_id = str(row.get("asset_id") or "").strip()
        label_text = str(row.get("label_text") or "").strip()
        bbox_xyxy = row.get("bbox_xyxy")
        status = str(row.get("status") or "").strip().lower()
        if not asset_id or not label_text:
            continue
        if status not in {"matched", "unmatched", "discarded"}:
            continue
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in bbox_xyxy):
            continue
        asset_frame_index = row.get("asset_frame_index")
        detections.append(
            {
                "asset_id": asset_id,
                "asset_frame_index": int(asset_frame_index) if isinstance(asset_frame_index, int) else None,
                "label_text": label_text,
                "resolved_category_id": str(row.get("resolved_category_id") or "").strip() or None,
                "resolved_category_name": str(row.get("resolved_category_name") or "").strip() or None,
                "confidence": float(row.get("confidence") or 0.0),
                "bbox_xyxy": [float(value) for value in bbox_xyxy],
                "status": status,
            }
        )
    return detections


def _append_debug_detection(
    session: PrelabelSession,
    *,
    asset: Asset,
    detection: DetectionResult,
    status: str,
    category: Category | None,
) -> None:
    debug_detections = _normalized_debug_detections(session.debug_detections_json)
    debug_detections.append(
        {
            "asset_id": asset.id,
            "asset_frame_index": int(asset.frame_index) if isinstance(asset.frame_index, int) else None,
            "label_text": str(detection.label_text or "").strip(),
            "resolved_category_id": category.id if category is not None else None,
            "resolved_category_name": category.name if category is not None else None,
            "confidence": float(detection.score),
            "bbox_xyxy": [float(value) for value in detection.bbox_xyxy],
            "status": status,
        }
    )
    session.debug_detections_json = debug_detections[-_PRELABEL_DEBUG_DETECTIONS_LIMIT:]


def _sampling_interval_frames(session: PrelabelSession, sequence_fps: float | None) -> int:
    raw_value = float(session.sampling_value or 1.0)
    if str(session.sampling_mode) == "every_n_seconds":
        fps = float(sequence_fps or 1.0)
        return max(1, int(round(raw_value * max(fps, 0.1))))
    return max(1, int(round(raw_value)))


def asset_matches_sampling(session: PrelabelSession, sequence: AssetSequence, asset: Asset) -> bool:
    frame_index = int(asset.frame_index or 0)
    interval_frames = _sampling_interval_frames(session, sequence.fps)
    return frame_index % interval_frames == 0


def _maybe_finalize_session(session: PrelabelSession) -> None:
    if str(session.status) in {"failed", "cancelled"}:
        return
    if session.input_closed_at is None:
        if int(session.enqueued_assets or 0) > 0:
            session.status = "running"
        return
    if int(session.processed_assets or 0) >= int(session.enqueued_assets or 0):
        session.status = "completed"
    elif int(session.enqueued_assets or 0) > 0:
        session.status = "running"


def _normalize_prelabel_label_key(value: str | None) -> str:
    chunks: list[str] = []
    current: list[str] = []
    for char in str(value or ""):
        if char.isalnum():
            current.append(char.lower())
            continue
        if current:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return " ".join(chunks)


def _inflection_alias_keys(normalized: str) -> set[str]:
    keys: set[str] = set()
    if not normalized:
        return keys

    if normalized.endswith("ies") and len(normalized) > 4:
        keys.add(f"{normalized[:-3]}y")
    if normalized.endswith("sses") and len(normalized) > 5:
        keys.add(normalized[:-2])
    elif normalized.endswith(("ches", "shes", "xes", "zes")) and len(normalized) > 4:
        keys.add(normalized[:-2])
    if normalized.endswith("s") and len(normalized) > 3 and not normalized.endswith("ss"):
        keys.add(normalized[:-1])

    if normalized.endswith("y") and len(normalized) > 1 and normalized[-2] not in {"a", "e", "i", "o", "u"}:
        keys.add(f"{normalized[:-1]}ies")
    elif normalized.endswith(("s", "x", "z", "ch", "sh")):
        keys.add(f"{normalized}es")
    else:
        keys.add(f"{normalized}s")
    return {key for key in keys if key and key != normalized}


def _category_exact_keys(value: str | None) -> list[str]:
    normalized = _normalize_prelabel_label_key(value)
    if not normalized:
        return []
    keys = [normalized]
    collapsed = normalized.replace(" ", "")
    if collapsed and collapsed != normalized:
        keys.append(collapsed)
    return keys


def _category_alias_keys(value: str | None) -> set[str]:
    normalized = _normalize_prelabel_label_key(value)
    if not normalized:
        return set()

    keys: set[str] = set()
    collapsed = normalized.replace(" ", "")
    if collapsed and collapsed != normalized:
        keys.add(collapsed)
    keys.update(_inflection_alias_keys(normalized))
    for alias in _PRELABEL_ALIAS_LOOKUP.get(normalized, set()):
        alias_key = _normalize_prelabel_label_key(alias)
        if not alias_key:
            continue
        keys.add(alias_key)
        collapsed_alias = alias_key.replace(" ", "")
        if collapsed_alias and collapsed_alias != alias_key:
            keys.add(collapsed_alias)
        keys.update(_inflection_alias_keys(alias_key))
    return keys - set(_category_exact_keys(value))


def _category_match_maps(categories: list[Category]) -> tuple[dict[str, Category], dict[str, Category]]:
    exact_mapping: dict[str, Category] = {}
    alias_candidates: dict[str, dict[str, Category]] = defaultdict(dict)
    for category in categories:
        for key in _category_exact_keys(category.name):
            if key not in exact_mapping:
                exact_mapping[key] = category
        for key in _category_alias_keys(category.name):
            alias_candidates[key][category.id] = category
    alias_mapping = {
        key: next(iter(categories_by_id.values()))
        for key, categories_by_id in alias_candidates.items()
        if len(categories_by_id) == 1 and key not in exact_mapping
    }
    return exact_mapping, alias_mapping


def _match_detection_category(
    *,
    label_text: str | None,
    exact_mapping: dict[str, Category],
    alias_mapping: dict[str, Category],
) -> Category | None:
    for key in _category_exact_keys(label_text):
        category = exact_mapping.get(key)
        if category is not None:
            return category
    for key in _category_alias_keys(label_text):
        category = alias_mapping.get(key)
        if category is not None:
            return category
    return None


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
    sequence: AssetSequence,
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

    response = await _warmup_florence_source(str(resolved["source_ref"]))
    return {
        "ok": True,
        "source_type": config.source_type,
        "source_ref": str(resolved["source_ref"]),
        "source_label": str(resolved["source_label"]),
        "device_selected": str(response.get("device_selected") or "cpu"),
        "device_preference": None,
    }


async def _warmup_florence_source(model_name: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(_FLORENCE_WARMUP_MAX_ATTEMPTS):
        try:
            return await inference_client.warmup_florence({"model_name": model_name})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 503 or attempt + 1 >= _FLORENCE_WARMUP_MAX_ATTEMPTS:
                raise
            last_error = exc
        except httpx.HTTPError as exc:
            if attempt + 1 >= _FLORENCE_WARMUP_MAX_ATTEMPTS:
                raise
            last_error = exc
        await asyncio.sleep(_FLORENCE_WARMUP_RETRY_DELAY_SECONDS)
    assert last_error is not None
    raise last_error


async def get_latest_sequence_prelabel_session(
    db: AsyncSession,
    *,
    sequence_id: str,
    task_id: str | None,
) -> PrelabelSession | None:
    stmt = select(PrelabelSession).where(PrelabelSession.sequence_id == sequence_id)
    if task_id:
        stmt = stmt.where(PrelabelSession.task_id == task_id)
    stmt = stmt.order_by(PrelabelSession.created_at.desc(), PrelabelSession.id.desc()).limit(1)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def pending_prelabel_counts_for_assets(
    db: AsyncSession,
    *,
    task_id: str | None,
    asset_ids: list[str],
) -> dict[str, int]:
    if not task_id or not asset_ids:
        return {}
    result = await db.execute(
        select(PrelabelProposal.asset_id, func.count(PrelabelProposal.id))
        .where(
            PrelabelProposal.task_id == task_id,
            PrelabelProposal.asset_id.in_(asset_ids),
            PrelabelProposal.status == "pending",
        )
        .group_by(PrelabelProposal.asset_id)
    )
    return {str(asset_id): int(count or 0) for asset_id, count in result.all()}


async def list_sequence_prelabel_sessions(
    db: AsyncSession,
    *,
    project_id: str,
    task_id: str,
    sequence_id: str,
) -> list[PrelabelSession]:
    result = await db.execute(
        select(PrelabelSession)
        .where(
            PrelabelSession.project_id == project_id,
            PrelabelSession.task_id == task_id,
            PrelabelSession.sequence_id == sequence_id,
        )
        .order_by(PrelabelSession.created_at.desc(), PrelabelSession.id.desc())
    )
    return list(result.scalars().all())


def _job_payload(*, session: PrelabelSession, asset: Asset) -> dict[str, Any]:
    return {
        "job_version": "1",
        "job_type": "prelabel_asset",
        "session_id": session.id,
        "asset_id": asset.id,
    }


async def enqueue_live_prelabel_jobs_for_asset(
    db: AsyncSession,
    *,
    sequence: AssetSequence,
    asset: Asset,
    queue: PrelabelQueue | None = None,
) -> list[str]:
    effective_queue = queue or PrelabelQueue()
    result = await db.execute(
        select(PrelabelSession)
        .where(
            PrelabelSession.sequence_id == sequence.id,
            PrelabelSession.task_id == sequence.task_id,
            PrelabelSession.live_mode.is_(True),
            PrelabelSession.input_closed_at.is_(None),
            PrelabelSession.status.in_(("queued", "running")),
        )
        .order_by(PrelabelSession.created_at.asc(), PrelabelSession.id.asc())
    )
    sessions = list(result.scalars().all())
    enqueued_ids: list[str] = []
    for session in sessions:
        if not asset_matches_sampling(session, sequence, asset):
            continue
        session.enqueued_assets = int(session.enqueued_assets or 0) + 1
        session.status = "running"
        await effective_queue.enqueue_asset_job(_job_payload(session=session, asset=asset))
        enqueued_ids.append(session.id)
    return enqueued_ids


async def enqueue_existing_sequence_assets_for_session(
    session_id: str,
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    queue: PrelabelQueue | None = None,
) -> dict[str, Any]:
    effective_session_factory = session_factory or SessionLocal
    effective_queue = queue or PrelabelQueue()
    async with effective_session_factory() as db:
        session = await db.get(PrelabelSession, session_id)
        if session is None:
            return {"session_id": session_id, "enqueued": 0, "status": "missing"}
        if str(session.status) in {"failed", "cancelled", "completed"}:
            return {"session_id": session_id, "enqueued": 0, "status": session.status}
        if session.input_closed_at is not None or int(session.enqueued_assets or 0) > 0:
            return {"session_id": session_id, "enqueued": int(session.enqueued_assets or 0), "status": session.status}

        sequence = await db.get(AssetSequence, session.sequence_id)
        if sequence is None:
            session.status = "failed"
            session.error_message = "Sequence not found"
            await db.commit()
            return {"session_id": session_id, "enqueued": 0, "status": "failed"}

        assets = list(
            (
                await db.execute(
                    select(Asset)
                    .where(Asset.sequence_id == sequence.id)
                    .order_by(Asset.frame_index.asc(), Asset.id.asc())
                )
            ).scalars().all()
        )
        enqueued = 0
        for asset in assets:
            if not asset_matches_sampling(session, sequence, asset):
                continue
            session.enqueued_assets = int(session.enqueued_assets or 0) + 1
            session.status = "running"
            await effective_queue.enqueue_asset_job(_job_payload(session=session, asset=asset))
            enqueued += 1
        session.input_closed_at = utc_now_dt()
        _maybe_finalize_session(session)
        await db.commit()
        return {"session_id": session.id, "enqueued": enqueued, "status": session.status}


async def close_prelabel_session_input(db: AsyncSession, session: PrelabelSession) -> None:
    if session.input_closed_at is None:
        session.input_closed_at = utc_now_dt()
    _maybe_finalize_session(session)


def _normalize_xywh_bbox(bbox: list[Any] | None) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in bbox):
        return None
    x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if width <= 0 or height <= 0:
        return None
    return [x, y, width, height]


def _bbox_xyxy_to_xywh(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int | None,
    height: int | None,
) -> list[float] | None:
    x1, y1, x2, y2 = bbox_xyxy
    if width is not None and width > 0:
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
    if height is not None and height > 0:
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
    left = min(x1, x2)
    top = min(y1, y2)
    box_width = max(x1, x2) - left
    box_height = max(y1, y2) - top
    if box_width <= 0 or box_height <= 0:
        return None
    return [left, top, box_width, box_height]


async def _build_adapter(
    db: AsyncSession,
    *,
    project_id: str,
    task_id: str,
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


async def process_prelabel_asset_job(
    payload: dict[str, Any],
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    effective_session_factory = session_factory or SessionLocal
    session_id = str(payload.get("session_id") or "").strip()
    asset_id = str(payload.get("asset_id") or "").strip()
    if not session_id or not asset_id:
        raise RuntimeError("session_id and asset_id are required")

    async with effective_session_factory() as db:
        session = await db.get(PrelabelSession, session_id)
        if session is None:
            raise RuntimeError("Prelabel session not found")
        if str(session.status) == "cancelled":
            return {"session_id": session_id, "asset_id": asset_id, "status": "cancelled"}
        asset = await db.get(Asset, asset_id)
        sequence = await db.get(AssetSequence, session.sequence_id)
        task = await db.get(Task, session.task_id)
        if asset is None or sequence is None or task is None:
            session.status = "failed"
            session.error_message = "Prelabel context is missing"
            await db.commit()
            raise RuntimeError("Prelabel context is missing")

        categories = await list_task_categories(db, project_id=session.project_id, task_id=session.task_id)
        category_exact_map, category_alias_map = _category_match_maps(categories)
        storage_uri = None
        if isinstance(asset.metadata_json, dict):
            storage_uri = asset.metadata_json.get("storage_uri")
        if not isinstance(storage_uri, str) or not storage_uri.strip():
            session.status = "failed"
            session.error_message = "Asset storage path is missing"
            await db.commit()
            raise RuntimeError("Asset storage path is missing")

        try:
            adapter = await _build_adapter(db, project_id=session.project_id, task_id=session.task_id, session=session)
            detections = await adapter.detect(
                asset_storage_uri=storage_uri,
                prompts=list(session.prompts_json or []),
                threshold=float(session.confidence_threshold),
                max_detections=int(session.max_detections_per_frame),
            )
            proposals_created = 0
            skipped_unmatched = 0
            for detection in detections:
                category = _match_detection_category(
                    label_text=detection.label_text,
                    exact_mapping=category_exact_map,
                    alias_mapping=category_alias_map,
                )
                if category is None:
                    _append_debug_detection(session, asset=asset, detection=detection, status="unmatched", category=None)
                    skipped_unmatched += 1
                    logger.info(
                        "Skipping unmatched prelabel detection",
                        extra={
                            "session_id": session.id,
                            "asset_id": asset.id,
                            "label_text": detection.label_text,
                        },
                    )
                    continue
                bbox_xywh = _bbox_xyxy_to_xywh(detection.bbox_xyxy, width=asset.width, height=asset.height)
                if bbox_xywh is None:
                    _append_debug_detection(session, asset=asset, detection=detection, status="discarded", category=category)
                    continue
                _append_debug_detection(session, asset=asset, detection=detection, status="matched", category=category)
                proposal = PrelabelProposal(
                    session_id=session.id,
                    asset_id=asset.id,
                    project_id=session.project_id,
                    task_id=session.task_id,
                    category_id=category.id,
                    label_text=category.name,
                    prompt_text=detection.label_text,
                    confidence=float(detection.score),
                    bbox_json=bbox_xywh,
                    status="pending",
                )
                db.add(proposal)
                proposals_created += 1

            session.processed_assets = int(session.processed_assets or 0) + 1
            session.generated_proposals = int(session.generated_proposals or 0) + proposals_created
            session.skipped_unmatched = int(session.skipped_unmatched or 0) + skipped_unmatched
            session.status = "running"
            _maybe_finalize_session(session)
            await db.commit()
            return {
                "session_id": session.id,
                "asset_id": asset.id,
                "status": session.status,
                "generated_proposals": proposals_created,
                "skipped_unmatched": skipped_unmatched,
            }
        except Exception as exc:
            await db.rollback()
            session = await db.get(PrelabelSession, session_id)
            if session is not None:
                session.status = "failed"
                session.error_message = str(exc) or "Prelabel inference failed"
                await db.commit()
            raise


async def mark_prelabel_session_failed(
    session_id: str,
    *,
    message: str,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> None:
    effective_session_factory = session_factory or SessionLocal
    async with effective_session_factory() as db:
        session = await db.get(PrelabelSession, session_id)
        if session is None:
            return
        session.status = "failed"
        session.error_message = message
        await db.commit()


def _proposal_scope_filter(
    *,
    session_id: str,
    asset_id: str | None,
    proposal_ids: list[str],
) -> tuple[list[Any], set[str]]:
    conditions: list[Any] = [PrelabelProposal.session_id == session_id]
    normalized_ids = {str(value).strip() for value in proposal_ids if str(value).strip()}
    if asset_id:
        conditions.append(PrelabelProposal.asset_id == asset_id)
    if normalized_ids:
        conditions.append(PrelabelProposal.id.in_(normalized_ids))
    return conditions, normalized_ids


async def list_session_proposals(
    db: AsyncSession,
    *,
    session_id: str,
    asset_id: str | None = None,
    status: str | None = None,
) -> list[PrelabelProposal]:
    stmt = select(PrelabelProposal).where(PrelabelProposal.session_id == session_id)
    if asset_id:
        stmt = stmt.where(PrelabelProposal.asset_id == asset_id)
    if status:
        stmt = stmt.where(PrelabelProposal.status == status)
    stmt = stmt.order_by(PrelabelProposal.created_at.asc(), PrelabelProposal.id.asc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


def _object_provenance_for_proposal(session: PrelabelSession, proposal: PrelabelProposal, *, decision: str) -> dict[str, Any]:
    return {
        "origin_kind": "ai_prelabel",
        "session_id": session.id,
        "proposal_id": proposal.id,
        "source_model": str(session.source_ref or session.source_type),
        "prompt_text": proposal.prompt_text,
        "confidence": float(proposal.confidence or 0.0),
        "review_decision": decision,
    }


def _annotation_payload_shell(
    *,
    asset: Asset,
    payload_json: dict[str, Any] | None,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    category_ids: list[str] = []
    for object_value in objects:
        category_id = object_value.get("category_id")
        if isinstance(category_id, str) and category_id not in category_ids:
            category_ids.append(category_id)
    primary_category_id = category_ids[0] if category_ids else None
    previous = payload_json if isinstance(payload_json, dict) else {}
    return {
        "version": "2.0",
        "type": "classification",
        "category_id": primary_category_id,
        "category_ids": category_ids,
        "classification": {
            "category_ids": category_ids,
            "primary_category_id": primary_category_id,
        },
        "objects": objects,
        "image_basis": {
            "width": int(asset.width),
            "height": int(asset.height),
        }
        if isinstance(asset.width, int) and asset.width > 0 and isinstance(asset.height, int) and asset.height > 0
        else previous.get("image_basis"),
        "coco": {
            "image_id": asset.id,
            "category_id": primary_category_id,
        },
        "source": str(previous.get("source") or "web-ui"),
    }


def _find_existing_object_for_proposal(objects: list[dict[str, Any]], proposal_id: str) -> dict[str, Any] | None:
    for object_value in objects:
        if not isinstance(object_value, dict):
            continue
        provenance = object_value.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if str(provenance.get("proposal_id") or "") == proposal_id:
            return object_value
    return None


async def _merge_proposals_for_asset(
    db: AsyncSession,
    *,
    session: PrelabelSession,
    asset: Asset,
    proposals: list[PrelabelProposal],
    categories: list[Category],
) -> str | None:
    allowed_category_ids = {category.id for category in categories}
    result = await db.execute(
        select(Annotation).where(
            Annotation.project_id == session.project_id,
            Annotation.task_id == session.task_id,
            Annotation.asset_id == asset.id,
        )
    )
    annotation = result.scalar_one_or_none()
    payload_json = annotation.payload_json if annotation and isinstance(annotation.payload_json, dict) else {}
    existing_objects = list(payload_json.get("objects") or []) if isinstance(payload_json.get("objects"), list) else []
    next_objects = [object_value for object_value in existing_objects if isinstance(object_value, dict)]

    for proposal in proposals:
        bbox = _normalize_xywh_bbox(
            proposal.reviewed_bbox_json if isinstance(proposal.reviewed_bbox_json, list) else proposal.bbox_json
        )
        category_id = proposal.reviewed_category_id or proposal.category_id
        if bbox is None or category_id not in allowed_category_ids:
            continue
        existing_object = _find_existing_object_for_proposal(next_objects, proposal.id)
        if existing_object is None:
            object_id = proposal.promoted_object_id or f"prelabel-{proposal.id}"
            next_objects.append(
                {
                    "id": object_id,
                    "kind": "bbox",
                    "category_id": category_id,
                    "bbox": bbox,
                    "provenance": _object_provenance_for_proposal(session, proposal, decision="accepted"),
                }
            )
            proposal.promoted_object_id = object_id
        else:
            proposal.promoted_object_id = str(existing_object.get("id") or proposal.promoted_object_id or "")

    normalized_payload = normalize_annotation_payload(
        _annotation_payload_shell(asset=asset, payload_json=payload_json, objects=next_objects),
        task_kind=TaskKind.bbox,
        label_mode=None,
        allowed_category_ids=allowed_category_ids,
        asset_width=asset.width,
        asset_height=asset.height,
    )

    if annotation is None:
        annotation = Annotation(
            project_id=session.project_id,
            asset_id=asset.id,
            task_id=session.task_id,
            status=AnnotationStatus.approved,
            payload_json=normalized_payload,
            annotated_by="ai-prelabel",
        )
        db.add(annotation)
        await db.flush()
    else:
        annotation.status = AnnotationStatus.approved
        annotation.payload_json = normalized_payload
        if not annotation.annotated_by:
            annotation.annotated_by = "ai-prelabel"
        await db.flush()

    for proposal in proposals:
        proposal.status = "accepted"
        proposal.reviewed_bbox_json = proposal.reviewed_bbox_json if isinstance(proposal.reviewed_bbox_json, list) else proposal.bbox_json
        proposal.reviewed_category_id = proposal.reviewed_category_id or proposal.category_id
        proposal.promoted_annotation_id = annotation.id
    return annotation.id


async def accept_prelabel_proposals(
    db: AsyncSession,
    *,
    session: PrelabelSession,
    asset_id: str | None,
    proposal_ids: list[str],
) -> list[str]:
    conditions, _normalized_ids = _proposal_scope_filter(session_id=session.id, asset_id=asset_id, proposal_ids=proposal_ids)
    stmt = (
        select(PrelabelProposal)
        .where(*conditions, PrelabelProposal.status == "pending")
        .order_by(PrelabelProposal.asset_id.asc(), PrelabelProposal.created_at.asc(), PrelabelProposal.id.asc())
    )
    proposals = list((await db.execute(stmt)).scalars().all())
    if not proposals:
        return []

    categories = await list_task_categories(db, project_id=session.project_id, task_id=session.task_id)
    asset_ids = sorted({proposal.asset_id for proposal in proposals})
    assets = list((await db.execute(select(Asset).where(Asset.id.in_(asset_ids)))).scalars().all())
    asset_by_id = {asset.id: asset for asset in assets}
    grouped: dict[str, list[PrelabelProposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal.asset_id].append(proposal)

    annotation_ids: list[str] = []
    for grouped_asset_id, grouped_proposals in grouped.items():
        asset = asset_by_id.get(grouped_asset_id)
        if asset is None:
            continue
        annotation_id = await _merge_proposals_for_asset(
            db,
            session=session,
            asset=asset,
            proposals=grouped_proposals,
            categories=categories,
        )
        if annotation_id:
            annotation_ids.append(annotation_id)
    await db.commit()
    return annotation_ids


async def reject_prelabel_proposals(
    db: AsyncSession,
    *,
    session: PrelabelSession,
    asset_id: str | None,
    proposal_ids: list[str],
) -> int:
    conditions, _normalized_ids = _proposal_scope_filter(session_id=session.id, asset_id=asset_id, proposal_ids=proposal_ids)
    proposals = list(
        (
            await db.execute(
                select(PrelabelProposal)
                .where(*conditions, PrelabelProposal.status == "pending")
                .order_by(PrelabelProposal.created_at.asc(), PrelabelProposal.id.asc())
            )
        ).scalars().all()
    )
    for proposal in proposals:
        proposal.status = "rejected"
    await db.commit()
    return len(proposals)


def _object_bbox_from_payload_object(object_value: dict[str, Any]) -> list[float] | None:
    raw_bbox = object_value.get("bbox")
    return _normalize_xywh_bbox(raw_bbox if isinstance(raw_bbox, list) else None)


def _object_review_decision(object_value: dict[str, Any], proposal: PrelabelProposal) -> str:
    proposal_bbox = _normalize_xywh_bbox(proposal.bbox_json)
    object_bbox = _object_bbox_from_payload_object(object_value)
    object_category_id = str(object_value.get("category_id") or "").strip()
    if proposal_bbox == object_bbox and object_category_id == proposal.category_id:
        return "accepted"
    return "edited"


async def sync_annotation_prelabel_proposals(
    db: AsyncSession,
    *,
    annotation: Annotation,
) -> None:
    payload_json = annotation.payload_json if isinstance(annotation.payload_json, dict) else {}
    objects = payload_json.get("objects")
    if not isinstance(objects, list):
        return

    proposal_ids: list[str] = []
    object_by_proposal_id: dict[str, dict[str, Any]] = {}
    for object_value in objects:
        if not isinstance(object_value, dict):
            continue
        provenance = object_value.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if str(provenance.get("origin_kind") or "") != "ai_prelabel":
            continue
        proposal_id = str(provenance.get("proposal_id") or "").strip()
        if not proposal_id:
            continue
        proposal_ids.append(proposal_id)
        object_by_proposal_id[proposal_id] = object_value

    if not proposal_ids:
        return

    proposals = list(
        (
            await db.execute(
                select(PrelabelProposal).where(
                    PrelabelProposal.id.in_(proposal_ids),
                    PrelabelProposal.asset_id == annotation.asset_id,
                    PrelabelProposal.task_id == annotation.task_id,
                    PrelabelProposal.project_id == annotation.project_id,
                )
            )
        ).scalars().all()
    )
    for proposal in proposals:
        object_value = object_by_proposal_id.get(proposal.id)
        if object_value is None:
            continue
        proposal.status = _object_review_decision(object_value, proposal)
        proposal.reviewed_bbox_json = _object_bbox_from_payload_object(object_value)
        proposal.reviewed_category_id = str(object_value.get("category_id") or "").strip() or proposal.category_id
        proposal.promoted_annotation_id = annotation.id
        proposal.promoted_object_id = str(object_value.get("id") or proposal.promoted_object_id or "")
