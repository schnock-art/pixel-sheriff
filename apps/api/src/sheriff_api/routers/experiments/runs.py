from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import ProjectExperimentActionResponse

from .shared import (
    as_sse,
    collect_config_issues,
    ensure_dataset_export_zip,
    experiment_store,
    get_dataset_version,
    model_store,
    require_project,
    shared_architecture_family,
    train_queue,
    utc_now_iso,
)

router = APIRouter()


@router.post("/projects/{project_id}/experiments/{experiment_id}/start", response_model=ProjectExperimentActionResponse)
async def start_project_experiment(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentActionResponse:
    project = await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    status = str(current.get("status", "draft"))
    if status not in {"draft", "failed", "canceled"}:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment can only be started from draft, failed, or canceled state",
            details={"experiment_id": experiment_id, "status": status},
        )

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
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Experiment config validation failed",
            details={"issues": [{"path": "config_json", "message": "Experiment config is required"}]},
        )
    issues = collect_config_issues(config_json)
    if issues:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Experiment config validation failed",
            details={"issues": issues},
        )

    dataset_version_id = str(config_json.get("dataset_version_id") or "")
    dataset_version = await get_dataset_version(db, project_id, dataset_version_id)
    if dataset_version is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )

    dataset_export = await ensure_dataset_export_zip(db=db, project=project, dataset_version=dataset_version)
    model_config = model_record.get("config_json")
    if not isinstance(model_config, dict):
        raise api_error(
            status_code=422,
            code="model_config_invalid",
            message="Model config is not available",
            details={"project_id": project_id, "model_id": model_id},
        )

    model_family = shared_architecture_family(model_config)
    task = str(config_json.get("task") or "classification")
    job_id = str(uuid.uuid4())

    initialized = experiment_store.init_run_attempt(
        project_id=project_id,
        experiment_id=experiment_id,
        job_id=job_id,
        dataset_export=dataset_export,
        task=task,
        model_family=model_family,
    )
    if initialized is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt = initialized.get("current_run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        raise api_error(
            status_code=500,
            code="experiment_attempt_init_failed",
            message="Failed to initialize experiment run attempt",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    experiment_store.append_event(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        event={"type": "status", "status": "queued", "attempt": attempt, "job_id": job_id, "ts": utc_now_iso()},
    )

    job_payload = {
        "job_version": "1",
        "job_id": job_id,
        "job_type": "train",
        "attempt": attempt,
        "project_id": project_id,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "task": task,
        "model_config": model_config,
        "training_config": config_json,
        "dataset_export": dataset_export,
    }
    try:
        await train_queue.enqueue_train_job(job_payload)
    except Exception as exc:
        experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="failed", error=str(exc))
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            event={
                "type": "done",
                "status": "failed",
                "attempt": attempt,
                "ts": utc_now_iso(),
                "error_code": "train_queue_unavailable",
            },
        )
        raise api_error(
            status_code=503,
            code="train_queue_unavailable",
            message="Training queue is unavailable",
            details={"project_id": project_id, "experiment_id": experiment_id},
        ) from exc

    return ProjectExperimentActionResponse(ok=True, status="queued", attempt=attempt, job_id=job_id)


@router.post("/projects/{project_id}/experiments/{experiment_id}/cancel", response_model=ProjectExperimentActionResponse)
async def cancel_project_experiment(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentActionResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    status = str(current.get("status", "draft"))
    attempt = current.get("current_run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        raise api_error(
            status_code=409,
            code="experiment_state_invalid",
            message="Experiment has no active run to cancel",
            details={"experiment_id": experiment_id, "status": status},
        )

    if status == "queued":
        experiment_store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=True)
        experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="canceled")
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            event={"type": "done", "status": "canceled", "attempt": attempt, "ts": utc_now_iso()},
        )
        return ProjectExperimentActionResponse(ok=True, status="canceled", attempt=attempt)

    if status == "running":
        experiment_store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=True)
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            event={"type": "status", "status": "running", "attempt": attempt, "ts": utc_now_iso()},
        )
        return ProjectExperimentActionResponse(ok=True, status="running", attempt=attempt)

    raise api_error(
        status_code=409,
        code="experiment_state_invalid",
        message="Only queued or running experiments can be canceled",
        details={"experiment_id": experiment_id, "status": status},
    )


@router.get("/projects/{project_id}/experiments/{experiment_id}/events")
async def stream_project_experiment_events(
    project_id: str,
    experiment_id: str,
    from_line: int = 0,
    attempt: int | None = None,
    follow: bool = True,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    current_attempt = current.get("current_run_attempt")
    resolved_attempt = attempt if isinstance(attempt, int) and attempt >= 1 else current_attempt

    async def event_stream():
        if not isinstance(resolved_attempt, int) or resolved_attempt < 1:
            status = str(current.get("status", "draft"))
            yield as_sse({"line": 0, "attempt": None, "event": {"type": "status", "status": status}})
            if status in {"completed", "failed", "canceled", "draft"}:
                yield as_sse({"line": 0, "attempt": None, "event": {"type": "done", "status": status}})
            return

        cursor = max(0, int(from_line))
        done = False
        sent_snapshot = False
        while True:
            rows = experiment_store.read_events(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=resolved_attempt,
                from_line=cursor,
            )
            if rows:
                for row in rows:
                    cursor = int(row["line"])
                    event = row.get("event")
                    if isinstance(event, dict) and str(event.get("type")) == "done":
                        done = True
                    yield as_sse(row)
                if done:
                    break
                if not follow:
                    break
                continue

            status_row = experiment_store.get_status_row(project_id, experiment_id)
            status = str(status_row.get("status", "draft"))
            line_count = experiment_store.get_event_line_count(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=resolved_attempt,
            )
            if not sent_snapshot:
                sent_snapshot = True
                yield as_sse(
                    {
                        "line": cursor,
                        "attempt": resolved_attempt,
                        "event": {"type": "status", "status": status, "attempt": resolved_attempt},
                    }
                )
                if status in {"completed", "failed", "canceled", "draft"} and line_count <= cursor:
                    yield as_sse(
                        {
                            "line": cursor,
                            "attempt": resolved_attempt,
                            "event": {"type": "done", "status": status, "attempt": resolved_attempt},
                        }
                    )
                    break
                if not follow:
                    break
                continue
            if status in {"completed", "failed", "canceled"} and line_count <= cursor:
                yield as_sse(
                    {
                        "line": cursor,
                        "attempt": resolved_attempt,
                        "event": {"type": "done", "status": status, "attempt": resolved_attempt},
                    }
                )
                break

            if not follow:
                break
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.8)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
