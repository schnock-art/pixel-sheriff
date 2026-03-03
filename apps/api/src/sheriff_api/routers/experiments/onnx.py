from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.experiments import ExperimentOnnxResponse

from .shared import experiment_store, require_project

router = APIRouter()


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for row in value:
        if isinstance(row, int):
            result.append(int(row))
    return result


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for row in value:
        if isinstance(row, str):
            result.append(row)
    return result


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/onnx",
    response_model=ExperimentOnnxResponse,
)
async def get_project_experiment_onnx(
    project_id: str,
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentOnnxResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    latest = experiment_store.get_latest_onnx(project_id, experiment_id)
    if latest is None:
        raise api_error(
            status_code=404,
            code="onnx_not_found",
            message="ONNX export not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    attempt = int(latest.get("attempt") or 0)
    model_path = latest.get("model_path")
    metadata_path = latest.get("metadata_path")
    metadata = _load_metadata(metadata_path if isinstance(metadata_path, Path) else None)

    class_names = _as_str_list(metadata.get("class_names"))
    class_order = _as_str_list(metadata.get("class_order"))
    if not class_order:
        class_order = class_names
    status = str(metadata.get("status") or "")
    if status not in {"exported", "failed"}:
        status = "exported" if isinstance(model_path, Path) and model_path.exists() else "failed"

    model_url = None
    if isinstance(model_path, Path) and model_path.exists():
        model_url = f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download?file=model"

    return ExperimentOnnxResponse(
        attempt=attempt,
        status=status,
        model_onnx_url=model_url,
        metadata_url=f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download?file=metadata",
        input_shape=_as_int_list(metadata.get("input_shape")),
        class_names=class_names,
        class_order=class_order,
        preprocess=metadata.get("preprocess") if isinstance(metadata.get("preprocess"), dict) else {},
        validation=metadata.get("validation") if isinstance(metadata.get("validation"), dict) else None,
        error=str(metadata.get("error")) if isinstance(metadata.get("error"), str) and metadata.get("error") else None,
    )


@router.get("/projects/{project_id}/experiments/{experiment_id}/onnx/download")
async def download_project_experiment_onnx(
    project_id: str,
    experiment_id: str,
    file: Literal["model", "metadata"] = Query(default="model"),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    await require_project(db, project_id)
    current = experiment_store.get(project_id, experiment_id, metrics_limit=1)
    if current is None:
        raise api_error(
            status_code=404,
            code="experiment_not_found",
            message="Experiment not found in project",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    latest = experiment_store.get_latest_onnx(project_id, experiment_id)
    if latest is None:
        raise api_error(
            status_code=404,
            code="onnx_not_found",
            message="ONNX export not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )

    attempt = int(latest.get("attempt") or 0)
    model_path = latest.get("model_path")
    metadata_path = latest.get("metadata_path")
    if file == "model":
        if not isinstance(model_path, Path) or not model_path.exists() or not model_path.is_file():
            raise api_error(
                status_code=404,
                code="onnx_not_found",
                message="ONNX export not available for this experiment",
                details={"project_id": project_id, "experiment_id": experiment_id},
            )
        return FileResponse(
            path=model_path,
            media_type="application/octet-stream",
            filename=f"{experiment_id}-run{attempt}-model.onnx",
        )

    if not isinstance(metadata_path, Path) or not metadata_path.exists() or not metadata_path.is_file():
        raise api_error(
            status_code=404,
            code="onnx_not_found",
            message="ONNX export not available for this experiment",
            details={"project_id": project_id, "experiment_id": experiment_id},
        )
    return FileResponse(
        path=metadata_path,
        media_type="application/json",
        filename=f"{experiment_id}-run{attempt}-onnx.metadata.json",
    )
