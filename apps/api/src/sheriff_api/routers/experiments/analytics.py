from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.session import get_db
from sheriff_api.schemas.experiments import (
    ExperimentAnalyticsBest,
    ExperimentAnalyticsItem,
    ProjectExperimentAnalyticsResponse,
)

from .shared import (
    experiment_store,
    extract_experiment_config,
    metric_objective_direction,
    model_store,
    require_project,
    safe_float,
    safe_int,
    series_row_value,
    utc_now_iso,
)

router = APIRouter()


@router.get("/projects/{project_id}/experiments/analytics", response_model=ProjectExperimentAnalyticsResponse)
async def project_experiments_analytics(
    project_id: str,
    max_points: int = Query(default=200, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
) -> ProjectExperimentAnalyticsResponse:
    await require_project(db, project_id)
    records = experiment_store.list_by_project(project_id)
    available_series: set[str] = set()
    items: list[ExperimentAnalyticsItem] = []
    for row in records:
        experiment_id = str(row.get("id") or "")
        if not experiment_id:
            continue
        status = str(row.get("status") or "draft")
        if status not in {"draft", "queued", "running", "completed", "failed", "canceled"}:
            status = "draft"
        resolved_status: Literal["draft", "queued", "running", "completed", "failed", "canceled"] = status
        config_json = row.get("config_json") if isinstance(row.get("config_json"), dict) else {}
        resolved_attempt = row.get("last_completed_attempt")
        if not isinstance(resolved_attempt, int) or resolved_attempt < 1:
            current_attempt = row.get("current_run_attempt")
            if isinstance(current_attempt, int) and current_attempt >= 1:
                resolved_attempt = current_attempt
            else:
                resolved_attempt = None
        metrics_rows = experiment_store.read_metrics(
            project_id,
            experiment_id,
            limit=max_points,
            attempt=resolved_attempt,
        )

        valid_rows: list[dict[str, Any]] = []
        for metric_row in metrics_rows:
            if not isinstance(metric_row, dict):
                continue
            epoch = safe_int(metric_row.get("epoch"))
            if epoch is None or epoch < 1:
                continue
            normalized_row = dict(metric_row)
            normalized_row["epoch"] = int(epoch)
            valid_rows.append(normalized_row)

        series: dict[str, Any] = {"epochs": [int(metric_row["epoch"]) for metric_row in valid_rows]}
        metric_keys: list[str] = []
        for metric_row in valid_rows:
            for key in metric_row.keys():
                if key in {"attempt", "created_at", "epoch"}:
                    continue
                if key not in metric_keys:
                    metric_keys.append(key)
        for key in metric_keys:
            values = [series_row_value(metric_row, key) for metric_row in valid_rows]
            if any(value is not None for value in values):
                series[key] = values
                available_series.add(key)

        summary_json = row.get("summary_json") if isinstance(row.get("summary_json"), dict) else {}
        best_metric_name = summary_json.get("best_metric_name")
        best_metric_value = safe_float(summary_json.get("best_metric_value"))
        best_epoch = safe_int(summary_json.get("best_epoch"))
        if not isinstance(best_metric_name, str):
            best_metric_name = None
        if best_metric_name is None and "val_accuracy" in series:
            best_metric_name = "val_accuracy"
            objective = metric_objective_direction(best_metric_name)
            candidates = [
                (epoch, value)
                for epoch, value in zip(series.get("epochs", []), series.get(best_metric_name, []))
                if isinstance(epoch, int) and isinstance(value, (int, float))
            ]
            if candidates:
                if objective == "min":
                    best_epoch, best_metric_value = min(candidates, key=lambda item: float(item[1]))
                else:
                    best_epoch, best_metric_value = max(candidates, key=lambda item: float(item[1]))

        final: dict[str, float | None] = {}
        if valid_rows:
            last_row = valid_rows[-1]
            for key in (
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_macro_precision",
                "val_macro_recall",
                "val_map",
                "val_iou",
                "epoch_seconds",
                "eta_seconds",
            ):
                value = safe_float(last_row.get(key))
                if value is not None:
                    final[key] = value

        model_id = str(row.get("model_id") or "")
        model_record = model_store.get(project_id, model_id) if model_id else None
        model_name = str(model_record.get("name")) if isinstance(model_record, dict) and model_record.get("name") else model_id
        updated_at = row.get("updated_at")
        if not isinstance(updated_at, str):
            updated_at = utc_now_iso()

        runtime_loaded = experiment_store.read_runtime(project_id, experiment_id)
        runtime_payload = runtime_loaded[1] if isinstance(runtime_loaded, tuple) else None
        runtime_summary: dict[str, Any] | None = None
        if isinstance(runtime_payload, dict):
            device_selected = runtime_payload.get("device_selected")
            if isinstance(device_selected, str) and device_selected.strip():
                runtime_summary = {"device_selected": device_selected.strip().lower()}

        items.append(
            ExperimentAnalyticsItem(
                experiment_id=experiment_id,
                name=str(row.get("name") or experiment_id),
                model_id=model_id,
                model_name=model_name,
                status=resolved_status,
                updated_at=updated_at,
                config=extract_experiment_config(config_json),
                best=ExperimentAnalyticsBest(
                    metric_name=best_metric_name,
                    metric_value=best_metric_value,
                    epoch=best_epoch,
                ),
                final=final,
                series=series,
                runtime=runtime_summary,
            )
        )

    return ProjectExperimentAnalyticsResponse(
        items=items,
        available_series=sorted(available_series),
    )
