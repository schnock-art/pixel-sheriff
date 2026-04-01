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
from .variants import VARIANT_FP32, VARIANT_KEYS, _list_variant_rows, _require_variant_attempt

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


def _resolve_variant_paths(
    *,
    project_id: str,
    experiment_id: str,
    current: dict[str, Any],
    requested_variant: str,
) -> tuple[int, str | None, str | None, Path | None, Path | None, list[str]]:
    attempt = _require_variant_attempt(current, project_id=project_id, experiment_id=experiment_id)
    listing = _list_variant_rows(project_id, experiment_id, attempt)
    variants = listing.get("variants")
    if not isinstance(variants, dict):
        variants = {}
    preferred_variant_key = listing.get("preferred_variant_key")
    resolved_variant = preferred_variant_key if requested_variant == "preferred" else requested_variant
    available_variants = [key for key in variants.keys() if key in VARIANT_KEYS]

    row = variants.get(resolved_variant) if isinstance(resolved_variant, str) else None
    if isinstance(row, dict):
        onnx = row.get("onnx")
        if not isinstance(onnx, dict):
            onnx = {}
        model_relpath = onnx.get("model_relpath")
        metadata_relpath = onnx.get("metadata_relpath")
        model_path = experiment_store._storage.resolve(model_relpath) if isinstance(model_relpath, str) and model_relpath else None  # type: ignore[attr-defined]
        metadata_path = experiment_store._storage.resolve(metadata_relpath) if isinstance(metadata_relpath, str) and metadata_relpath else None  # type: ignore[attr-defined]
        return attempt, resolved_variant if isinstance(resolved_variant, str) else None, preferred_variant_key, model_path, metadata_path, available_variants

    if resolved_variant in {None, VARIANT_FP32}:
        latest = experiment_store.get_latest_onnx(project_id, experiment_id)
        if latest is not None:
            return (
                int(latest.get("attempt") or attempt),
                VARIANT_FP32,
                preferred_variant_key or VARIANT_FP32,
                latest.get("model_path") if isinstance(latest.get("model_path"), Path) else None,
                latest.get("metadata_path") if isinstance(latest.get("metadata_path"), Path) else None,
                available_variants or [VARIANT_FP32],
            )

    raise api_error(
        status_code=404,
        code="onnx_not_found",
        message="Requested ONNX variant is not available for this experiment",
        details={"project_id": project_id, "experiment_id": experiment_id, "variant": requested_variant},
    )


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/onnx",
    response_model=ExperimentOnnxResponse,
)
async def get_project_experiment_onnx(
    project_id: str,
    experiment_id: str,
    variant: Literal["preferred", "fp32", "ptq_int8", "qat_int8"] = Query(default="preferred"),
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

    attempt, resolved_variant_key, preferred_variant_key, model_path, metadata_path, available_variants = _resolve_variant_paths(
        project_id=project_id,
        experiment_id=experiment_id,
        current=current,
        requested_variant=variant,
    )
    metadata = _load_metadata(metadata_path)

    class_names = _as_str_list(metadata.get("class_names"))
    class_order = _as_str_list(metadata.get("class_order"))
    if not class_order:
        class_order = class_names
    status = str(metadata.get("status") or "")
    if status not in {"exported", "failed"}:
        status = "exported" if isinstance(model_path, Path) and model_path.exists() else "failed"

    variant_query = ""
    if resolved_variant_key and not (variant == "preferred" and resolved_variant_key == VARIANT_FP32):
        variant_query = f"&variant={resolved_variant_key}"
    model_url = None
    if isinstance(model_path, Path) and model_path.exists():
        model_url = (
            f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download"
            f"?file=model{variant_query}"
        )

    return ExperimentOnnxResponse(
        attempt=attempt,
        status=status,
        variant_key=resolved_variant_key,
        preferred_variant_key=preferred_variant_key,
        model_onnx_url=model_url,
        metadata_url=(
            f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download"
            f"?file=metadata{variant_query}"
        ),
        input_shape=_as_int_list(metadata.get("input_shape")),
        class_names=class_names,
        class_order=class_order,
        preprocess=metadata.get("preprocess") if isinstance(metadata.get("preprocess"), dict) else {},
        validation=metadata.get("validation") if isinstance(metadata.get("validation"), dict) else None,
        error=str(metadata.get("error")) if isinstance(metadata.get("error"), str) and metadata.get("error") else None,
        available_variants=available_variants,
    )


@router.get("/projects/{project_id}/experiments/{experiment_id}/onnx/download")
async def download_project_experiment_onnx(
    project_id: str,
    experiment_id: str,
    file: Literal["model", "metadata"] = Query(default="model"),
    variant: Literal["preferred", "fp32", "ptq_int8", "qat_int8"] = Query(default="preferred"),
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

    attempt, resolved_variant_key, _preferred_variant_key, model_path, metadata_path, _available_variants = _resolve_variant_paths(
        project_id=project_id,
        experiment_id=experiment_id,
        current=current,
        requested_variant=variant,
    )
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
            filename=f"{experiment_id}-run{attempt}-{resolved_variant_key or 'preferred'}-model.onnx",
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
        filename=f"{experiment_id}-run{attempt}-{resolved_variant_key or 'preferred'}-onnx.metadata.json",
    )
