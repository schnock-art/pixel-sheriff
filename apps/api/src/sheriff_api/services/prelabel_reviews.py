from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, AnnotationStatus, Asset, Category, PrelabelProposal, PrelabelSession, TaskKind
from sheriff_api.services.annotation_payload import normalize_annotation_payload
from sheriff_api.services.prelabel_common import list_task_categories
from sheriff_api.services.prelabel_matching import normalize_xywh_bbox


def proposal_scope_filter(
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


def object_provenance_for_proposal(session: PrelabelSession, proposal: PrelabelProposal, *, decision: str) -> dict[str, Any]:
    return {
        "origin_kind": "ai_prelabel",
        "session_id": session.id,
        "proposal_id": proposal.id,
        "source_model": str(session.source_ref or session.source_type),
        "prompt_text": proposal.prompt_text,
        "confidence": float(proposal.confidence or 0.0),
        "review_decision": decision,
    }


def annotation_payload_shell(
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


def find_existing_object_for_proposal(objects: list[dict[str, Any]], proposal_id: str) -> dict[str, Any] | None:
    for object_value in objects:
        if not isinstance(object_value, dict):
            continue
        provenance = object_value.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if str(provenance.get("proposal_id") or "") == proposal_id:
            return object_value
    return None


async def merge_proposals_for_asset(
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
        bbox = normalize_xywh_bbox(
            proposal.reviewed_bbox_json if isinstance(proposal.reviewed_bbox_json, list) else proposal.bbox_json
        )
        category_id = proposal.reviewed_category_id or proposal.category_id
        if bbox is None or category_id not in allowed_category_ids:
            continue
        existing_object = find_existing_object_for_proposal(next_objects, proposal.id)
        if existing_object is None:
            object_id = proposal.promoted_object_id or f"prelabel-{proposal.id}"
            next_objects.append(
                {
                    "id": object_id,
                    "kind": "bbox",
                    "category_id": category_id,
                    "bbox": bbox,
                    "provenance": object_provenance_for_proposal(session, proposal, decision="accepted"),
                }
            )
            proposal.promoted_object_id = object_id
        else:
            proposal.promoted_object_id = str(existing_object.get("id") or proposal.promoted_object_id or "")

    normalized_payload = normalize_annotation_payload(
        annotation_payload_shell(asset=asset, payload_json=payload_json, objects=next_objects),
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
    conditions, _normalized_ids = proposal_scope_filter(session_id=session.id, asset_id=asset_id, proposal_ids=proposal_ids)
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
        annotation_id = await merge_proposals_for_asset(
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
    conditions, _normalized_ids = proposal_scope_filter(session_id=session.id, asset_id=asset_id, proposal_ids=proposal_ids)
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


def object_bbox_from_payload_object(object_value: dict[str, Any]) -> list[float] | None:
    raw_bbox = object_value.get("bbox")
    return normalize_xywh_bbox(raw_bbox if isinstance(raw_bbox, list) else None)


def object_review_decision(object_value: dict[str, Any], proposal: PrelabelProposal) -> str:
    proposal_bbox = normalize_xywh_bbox(proposal.bbox_json)
    object_bbox = object_bbox_from_payload_object(object_value)
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
        proposal.status = object_review_decision(object_value, proposal)
        proposal.reviewed_bbox_json = object_bbox_from_payload_object(object_value)
        proposal.reviewed_category_id = str(object_value.get("category_id") or "").strip() or proposal.category_id
        proposal.promoted_annotation_id = annotation.id
        proposal.promoted_object_id = str(object_value.get("id") or proposal.promoted_object_id or "")
