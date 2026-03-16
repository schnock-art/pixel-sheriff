from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import AssetSequence, Category, PrelabelProposal, PrelabelSession
from sheriff_api.schemas.prelabels import PrelabelProposalRead, PrelabelSessionRead
from sheriff_api.services.deployment_store import DeploymentStore
from sheriff_api.services.inference_client import InferenceClient
from sheriff_api.services.prelabel_matching import normalized_debug_detections


settings = get_settings()
deployment_store = DeploymentStore(settings.storage_root)
inference_client = InferenceClient(
    base_url=settings.trainer_inference_base_url,
    timeout_seconds=float(settings.trainer_inference_timeout_seconds),
)


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
        debug_detections=normalized_debug_detections(session.debug_detections_json),
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


def sampling_interval_frames(session: PrelabelSession, sequence_fps: float | None) -> int:
    raw_value = float(session.sampling_value or 1.0)
    if str(session.sampling_mode) == "every_n_seconds":
        fps = float(sequence_fps or 1.0)
        return max(1, int(round(raw_value * max(fps, 0.1))))
    return max(1, int(round(raw_value)))


def asset_matches_sampling(session: PrelabelSession, sequence: AssetSequence, asset) -> bool:
    frame_index = int(asset.frame_index or 0)
    interval_frames = sampling_interval_frames(session, sequence.fps)
    return frame_index % interval_frames == 0


def maybe_finalize_session(session: PrelabelSession) -> None:
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


async def close_prelabel_session_input(db: AsyncSession, session: PrelabelSession) -> None:
    if session.input_closed_at is None:
        session.input_closed_at = utc_now_dt()
    maybe_finalize_session(session)
