from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import (
    ExperimentEvaluationResponse,
    ExperimentLogsChunkResponse,
    ExperimentRuntimeResponse,
    ExperimentSamplesResponse,
)

from .shared import (
    as_sample_item,
    experiment_store,
    filter_predictions,
    require_project,
)

router = APIRouter()


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/runtime",
    response_model=ExperimentRuntimeResponse,
)
async def get_project_experiment_runtime(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentRuntimeResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    loaded = experiment_store.read_runtime(project_id, experiment_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="runtime_not_found",
            message="Runtime not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt, payload = loaded
    response_payload = dict(payload)
    response_payload["attempt"] = attempt
    return ExperimentRuntimeResponse.model_validate(response_payload)


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/logs",
    response_model=ExperimentLogsChunkResponse,
)
async def get_project_experiment_logs(
    project_id: str,
    experiment_id: str,
    attempt: int | None = Query(default=None, ge=1),
    from_byte: int = Query(default=0, ge=0),
    max_bytes: int = Query(default=65536, ge=1, le=524288),
    db: AsyncSession = Depends(get_db),
) -> ExperimentLogsChunkResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    payload = experiment_store.read_training_log_chunk(
        project_id,
        experiment_id,
        attempt=attempt,
        from_byte=from_byte,
        max_bytes=max_bytes,
    )
    if payload is None:
        raise api_error(
            status_code=404,
            code="logs_not_found",
            message="Training logs not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return ExperimentLogsChunkResponse(
        attempt=int(payload.get("attempt", 0)),
        from_byte=int(payload.get("from_byte", 0)),
        to_byte=int(payload.get("to_byte", 0)),
        content=str(payload.get("content", "")),
    )


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/evaluation",
    response_model=ExperimentEvaluationResponse,
)
async def get_project_experiment_evaluation(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentEvaluationResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    loaded = experiment_store.read_evaluation(project_id, experiment_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="evaluation_not_found",
            message="Evaluation not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt, payload = loaded
    response_payload = dict(payload)
    response_payload["attempt"] = attempt
    return ExperimentEvaluationResponse.model_validate(response_payload)


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/samples",
    response_model=ExperimentSamplesResponse,
)
async def get_project_experiment_samples(
    project_id: str,
    experiment_id: str,
    mode: str = Query(default="misclassified"),
    true_class_index: int | None = Query(default=None, ge=0),
    pred_class_index: int | None = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> ExperimentSamplesResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"misclassified", "lowest_confidence_correct", "highest_confidence_wrong"}:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Unsupported samples mode",
            details={"mode": mode},
        )
    resolved_mode: Literal["misclassified", "lowest_confidence_correct", "highest_confidence_wrong"] = normalized_mode

    loaded_predictions = experiment_store.read_predictions(project_id, experiment_id)
    if loaded_predictions is not None:
        attempt, rows, _meta = loaded_predictions
        filtered_rows = filter_predictions(
            rows,
            mode=normalized_mode,
            true_class_index=true_class_index,
            pred_class_index=pred_class_index,
        )
        return ExperimentSamplesResponse(
            attempt=attempt,
            mode=resolved_mode,
            items=[as_sample_item(row) for row in filtered_rows[:limit]],
        )

    loaded_evaluation = experiment_store.read_evaluation(project_id, experiment_id)
    if loaded_evaluation is None:
        raise api_error(
            status_code=404,
            code="evaluation_not_found",
            message="Evaluation not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    attempt, evaluation_payload = loaded_evaluation
    sample_rows: list[dict[str, Any]] = []
    samples_block = evaluation_payload.get("samples")
    if isinstance(samples_block, dict):
        raw_items = samples_block.get(normalized_mode)
        if isinstance(raw_items, list):
            sample_rows = [row for row in raw_items if isinstance(row, dict)]
    filtered_rows = filter_predictions(
        sample_rows,
        mode=normalized_mode,
        true_class_index=true_class_index,
        pred_class_index=pred_class_index,
    )
    message = None
    if not filtered_rows:
        message = "No matching samples found for this filter."
    return ExperimentSamplesResponse(
        attempt=attempt,
        mode=resolved_mode,
        items=[as_sample_item(row) for row in filtered_rows[:limit]],
        message=message,
    )
