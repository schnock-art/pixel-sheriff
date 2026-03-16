from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sheriff_api.db.models import Asset, AssetSequence, PrelabelProposal, PrelabelSession, Task
from sheriff_api.db.session import SessionLocal
from sheriff_api.services.prelabel_common import (
    asset_matches_sampling,
    list_task_categories,
    maybe_finalize_session,
    utc_now_dt,
)
from sheriff_api.services.prelabel_matching import (
    append_debug_detection,
    bbox_xyxy_to_xywh,
    category_match_maps,
    match_detection_category,
)
from sheriff_api.services.prelabel_queue import PrelabelQueue
from sheriff_api.services.prelabel_sources import build_adapter


logger = logging.getLogger(__name__)


def job_payload(*, session: PrelabelSession, asset: Asset) -> dict[str, Any]:
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
        await effective_queue.enqueue_asset_job(job_payload(session=session, asset=asset))
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
            await effective_queue.enqueue_asset_job(job_payload(session=session, asset=asset))
            enqueued += 1
        session.input_closed_at = utc_now_dt()
        maybe_finalize_session(session)
        await db.commit()
        return {"session_id": session.id, "enqueued": enqueued, "status": session.status}


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
        category_exact_map, category_alias_map = category_match_maps(categories)
        storage_uri = None
        if isinstance(asset.metadata_json, dict):
            storage_uri = asset.metadata_json.get("storage_uri")
        if not isinstance(storage_uri, str) or not storage_uri.strip():
            session.status = "failed"
            session.error_message = "Asset storage path is missing"
            await db.commit()
            raise RuntimeError("Asset storage path is missing")

        try:
            adapter = await build_adapter(project_id=session.project_id, session=session)
            detections = await adapter.detect(
                asset_storage_uri=storage_uri,
                prompts=list(session.prompts_json or []),
                threshold=float(session.confidence_threshold),
                max_detections=int(session.max_detections_per_frame),
            )
            proposals_created = 0
            skipped_unmatched = 0
            for detection in detections:
                category = match_detection_category(
                    label_text=detection.label_text,
                    exact_mapping=category_exact_map,
                    alias_mapping=category_alias_map,
                )
                if category is None:
                    append_debug_detection(session, asset=asset, detection=detection, status="unmatched", category=None)
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
                bbox_xywh = bbox_xyxy_to_xywh(detection.bbox_xyxy, width=asset.width, height=asset.height)
                if bbox_xywh is None:
                    append_debug_detection(session, asset=asset, detection=detection, status="discarded", category=category)
                    continue
                append_debug_detection(session, asset=asset, detection=detection, status="matched", category=category)
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
            maybe_finalize_session(session)
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
