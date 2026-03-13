from __future__ import annotations

import httpx

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import AssetSequence, PrelabelSession, Project, Task
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.prelabels import (
    PrelabelCloseResponse,
    PrelabelConfigCreate,
    PrelabelProposalListResponse,
    PrelabelReviewAction,
    PrelabelReviewResponse,
    PrelabelSourceStatusRead,
    PrelabelSessionCreate,
    PrelabelSessionCreateResponse,
    PrelabelSessionListResponse,
)
from sheriff_api.services.prelabels import (
    accept_prelabel_proposals,
    close_prelabel_session_input,
    create_prelabel_session,
    enqueue_existing_sequence_assets_for_session,
    list_session_proposals,
    list_sequence_prelabel_sessions,
    prelabel_proposal_to_read,
    prelabel_session_to_read,
    reject_prelabel_proposals,
    utc_now_dt,
    warmup_prelabel_source,
)


router = APIRouter(tags=["prelabels"])


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status.HTTP_404_NOT_FOUND, code="project_not_found", message="Project not found")
    return project


async def _require_task(db: AsyncSession, project_id: str, task_id: str) -> Task:
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )
    return task


async def _require_sequence(db: AsyncSession, project_id: str, sequence_id: str) -> AssetSequence:
    sequence = await db.get(AssetSequence, sequence_id)
    if sequence is None or sequence.project_id != project_id:
        raise api_error(status.HTTP_404_NOT_FOUND, code="sequence_not_found", message="Sequence not found")
    return sequence


async def _require_session(db: AsyncSession, project_id: str, task_id: str, session_id: str) -> PrelabelSession:
    session = await db.get(PrelabelSession, session_id)
    if session is None or session.project_id != project_id or session.task_id != task_id:
        raise api_error(status.HTTP_404_NOT_FOUND, code="prelabel_session_not_found", message="Prelabel session not found")
    return session


@router.get("/projects/{project_id}/tasks/{task_id}/prelabels", response_model=PrelabelSessionListResponse)
async def list_prelabel_sessions(
    project_id: str,
    task_id: str,
    sequence_id: str,
    db: AsyncSession = Depends(get_db),
) -> PrelabelSessionListResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    sessions = await list_sequence_prelabel_sessions(db, project_id=project_id, task_id=task_id, sequence_id=sequence_id)
    return PrelabelSessionListResponse(items=[prelabel_session_to_read(session) for session in sessions])


@router.post("/projects/{project_id}/tasks/{task_id}/prelabels", response_model=PrelabelSessionCreateResponse)
async def create_session(
    project_id: str,
    task_id: str,
    payload: PrelabelSessionCreate,
    db: AsyncSession = Depends(get_db),
) -> PrelabelSessionCreateResponse:
    await _require_project(db, project_id)
    task = await _require_task(db, project_id, task_id)
    sequence = await _require_sequence(db, project_id, payload.sequence_id)
    if sequence.task_id and sequence.task_id != task_id:
        raise api_error(status.HTTP_409_CONFLICT, code="sequence_task_mismatch", message="Sequence belongs to a different task")

    try:
        session = await create_prelabel_session(
            db,
            project_id=project_id,
            task=task,
            sequence=sequence,
            config=payload,
            live_mode=False,
        )
    except ValueError as exc:
        code = str(exc)
        if code in {"active_deployment_not_found", "active_deployment_incompatible"}:
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Active deployment is unavailable for this task") from exc
        if code == "task_kind_unsupported":
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Prelabels are supported only for bbox tasks") from exc
        raise

    await db.commit()
    if sequence.status == "ready":
        await enqueue_existing_sequence_assets_for_session(session.id)
        await db.refresh(session)
    else:
        await db.refresh(session)
    return PrelabelSessionCreateResponse(session=prelabel_session_to_read(session))


@router.post("/projects/{project_id}/tasks/{task_id}/prelabels/source-status", response_model=PrelabelSourceStatusRead)
async def get_prelabel_source_status(
    project_id: str,
    task_id: str,
    payload: PrelabelConfigCreate,
    db: AsyncSession = Depends(get_db),
) -> PrelabelSourceStatusRead:
    await _require_project(db, project_id)
    task = await _require_task(db, project_id, task_id)
    try:
        status_payload = await warmup_prelabel_source(
            db,
            project_id=project_id,
            task=task,
            config=payload,
        )
    except ValueError as exc:
        code = str(exc)
        if code in {"active_deployment_not_found", "active_deployment_incompatible"}:
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Active deployment is unavailable for this task") from exc
        if code == "task_kind_unsupported":
            raise api_error(status.HTTP_409_CONFLICT, code=code, message="Prelabels are supported only for bbox tasks") from exc
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            raise api_error(status.HTTP_503_SERVICE_UNAVAILABLE, code="inference_unavailable", message="Inference service unavailable") from exc
        raise api_error(status.HTTP_502_BAD_GATEWAY, code="inference_failed", message="Inference request failed") from exc
    except Exception as exc:
        raise api_error(status.HTTP_503_SERVICE_UNAVAILABLE, code="inference_unavailable", message=str(exc) or "Inference unavailable") from exc
    return PrelabelSourceStatusRead.model_validate(status_payload)


@router.get("/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}", response_model=PrelabelCloseResponse)
async def get_session(
    project_id: str,
    task_id: str,
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> PrelabelCloseResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    session = await _require_session(db, project_id, task_id, session_id)
    return PrelabelCloseResponse(session=prelabel_session_to_read(session))


@router.get("/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/proposals", response_model=PrelabelProposalListResponse)
async def get_session_proposals(
    project_id: str,
    task_id: str,
    session_id: str,
    asset_id: str | None = None,
    status_filter: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> PrelabelProposalListResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    await _require_session(db, project_id, task_id, session_id)
    proposals = await list_session_proposals(db, session_id=session_id, asset_id=asset_id, status=status_filter)
    return PrelabelProposalListResponse(items=[prelabel_proposal_to_read(proposal) for proposal in proposals])


@router.post("/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/accept", response_model=PrelabelReviewResponse)
async def accept_session_proposals(
    project_id: str,
    task_id: str,
    session_id: str,
    payload: PrelabelReviewAction | None = None,
    db: AsyncSession = Depends(get_db),
) -> PrelabelReviewResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    session = await _require_session(db, project_id, task_id, session_id)
    body = payload or PrelabelReviewAction()
    annotation_ids = await accept_prelabel_proposals(
        db,
        session=session,
        asset_id=body.asset_id,
        proposal_ids=body.proposal_ids,
    )
    await db.refresh(session)
    return PrelabelReviewResponse(
        session=prelabel_session_to_read(session),
        updated=len(annotation_ids),
        annotation_ids=annotation_ids,
    )


@router.post("/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/reject", response_model=PrelabelReviewResponse)
async def reject_session_proposals(
    project_id: str,
    task_id: str,
    session_id: str,
    payload: PrelabelReviewAction | None = None,
    db: AsyncSession = Depends(get_db),
) -> PrelabelReviewResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    session = await _require_session(db, project_id, task_id, session_id)
    body = payload or PrelabelReviewAction()
    updated = await reject_prelabel_proposals(
        db,
        session=session,
        asset_id=body.asset_id,
        proposal_ids=body.proposal_ids,
    )
    await db.refresh(session)
    return PrelabelReviewResponse(session=prelabel_session_to_read(session), updated=updated, annotation_ids=[])


@router.post("/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/close-input", response_model=PrelabelCloseResponse)
async def close_session_input(
    project_id: str,
    task_id: str,
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> PrelabelCloseResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    session = await _require_session(db, project_id, task_id, session_id)
    await close_prelabel_session_input(db, session)
    await db.commit()
    await db.refresh(session)
    return PrelabelCloseResponse(session=prelabel_session_to_read(session))


@router.post("/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/cancel", response_model=PrelabelCloseResponse)
async def cancel_session(
    project_id: str,
    task_id: str,
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> PrelabelCloseResponse:
    await _require_project(db, project_id)
    await _require_task(db, project_id, task_id)
    session = await _require_session(db, project_id, task_id, session_id)
    session.status = "cancelled"
    if session.input_closed_at is None:
        session.input_closed_at = utc_now_dt()
    await db.commit()
    await db.refresh(session)
    return PrelabelCloseResponse(session=prelabel_session_to_read(session))
