from __future__ import annotations

import json
import struct
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Category, Task, TaskKind
from sheriff_api.errors import api_error
from sheriff_api.schemas.prelabels import PrelabelConfigCreate
from sheriff_api.schemas.preview_inference import (
    PreviewInferenceBoxRead,
    PreviewInferenceDebugRead,
    PreviewInferencePredictionRead,
    PreviewInferenceResponse,
)
from sheriff_api.services.prelabel_common import deployment_store, inference_client, list_task_categories, normalize_prompts
from sheriff_api.services.prelabel_matching import bbox_xyxy_to_xywh, category_match_maps, match_detection_category
from sheriff_api.services.prelabel_sources import resolve_prelabel_source_config
from sheriff_api.services.storage import LocalStorage
from sheriff_api.config import get_settings


settings = get_settings()
storage = LocalStorage(settings.storage_root)

_JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _preview_storage_uri(project_id: str, *, filename: str | None, content_type: str | None) -> str:
    suffix = ""
    if isinstance(filename, str):
        dot_index = filename.rfind(".")
        if dot_index >= 0:
            suffix = filename[dot_index:].strip().lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        normalized_content_type = str(content_type or "").strip().lower()
        if normalized_content_type == "image/png":
            suffix = ".png"
        elif normalized_content_type == "image/webp":
            suffix = ".webp"
        else:
            suffix = ".jpg"
    return f"imports/{project_id}/preview/{uuid.uuid4().hex}{suffix}"


def _image_dimensions_from_bytes(content: bytes) -> tuple[int | None, int | None]:
    if len(content) >= 24 and content.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", content[16:24])
        return int(width), int(height)

    if len(content) >= 4 and content[:2] == b"\xff\xd8":
        index = 2
        while index + 1 < len(content):
            while index < len(content) and content[index] != 0xFF:
                index += 1
            while index < len(content) and content[index] == 0xFF:
                index += 1
            if index >= len(content):
                break
            marker = content[index]
            index += 1
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if index + 1 >= len(content):
                break
            segment_length = struct.unpack(">H", content[index:index + 2])[0]
            if segment_length < 2 or index + segment_length > len(content):
                break
            if marker in _JPEG_SOF_MARKERS and index + 7 < len(content):
                height, width = struct.unpack(">HH", content[index + 3:index + 7])
                return int(width), int(height)
            index += segment_length

    return None, None


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    resolved: list[str] = []
    for row in value:
        if isinstance(row, str) and row.strip():
            resolved.append(row.strip())
        elif isinstance(row, int):
            resolved.append(str(row))
    return resolved


def _normalize_label_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _map_category_names(values: list[str], *, categories: list[dict[str, str]]) -> list[str] | None:
    if not values:
        return None
    category_ids_by_name: dict[str, list[str]] = {}
    for category in categories:
        category_ids_by_name.setdefault(_normalize_label_name(category["name"]), []).append(category["id"])

    resolved: list[str] = []
    for value in values:
        matches = category_ids_by_name.get(_normalize_label_name(value), [])
        if len(matches) != 1:
            return None
        resolved.append(matches[0])
    return resolved


def _resolve_class_mapping(metadata: dict[str, Any], *, categories: list[dict[str, str]]) -> list[str]:
    category_name_by_id = {category["id"]: category["name"] for category in categories}

    for key in ("class_ids", "class_order"):
        values = _as_str_list(metadata.get(key))
        if values and all(class_id in category_name_by_id for class_id in values):
            return values

    for key in ("class_ids", "class_order", "class_names"):
        values = _as_str_list(metadata.get(key))
        resolved = _map_category_names(values, categories=categories)
        if resolved is not None:
            return resolved

    raw_class_ids = metadata.get("class_ids")
    raw_class_order = metadata.get("class_order")
    if (
        (isinstance(raw_class_ids, list) and any(isinstance(value, int) for value in raw_class_ids))
        or (isinstance(raw_class_order, list) and any(isinstance(value, int) for value in raw_class_order))
    ):
        raise api_error(
            status_code=409,
            code="deployment_legacy_metadata_incompatible",
            message="Deployment metadata uses legacy integer class_ids. Redeploy from a new dataset version.",
        )

    raise api_error(
        status_code=409,
        code="deployment_class_mapping_invalid",
        message="Deployment classes do not match project categories",
    )


def _read_deployment_metadata(metadata_relpath: str) -> dict[str, Any]:
    try:
        payload = json.loads(storage.resolve(metadata_relpath).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_compatible_deployment(project_id: str, *, task_id: str, task_kind: str, deployment_id: str | None) -> dict[str, Any]:
    listing = deployment_store.list(project_id)
    resolved_deployment_id = str(deployment_id or listing.get("active_deployment_id") or "").strip()
    if not resolved_deployment_id:
        raise api_error(status_code=409, code="no_active_deployment", message="No active deployment is configured")

    deployment = deployment_store.get(project_id, resolved_deployment_id)
    if deployment is None:
        raise api_error(status_code=404, code="deployment_not_found", message="Deployment not found")
    if str(deployment.get("status") or "").strip().lower() == "archived":
        raise api_error(status_code=409, code="deployment_archived", message="Deployment is archived")

    deployment_task = str(deployment.get("task") or "").strip().lower()
    if deployment_task != task_kind:
        code = "active_deployment_incompatible" if deployment_id is None else "deployment_incompatible"
        raise api_error(status_code=409, code=code, message="Deployment is incompatible with this task")

    deployment_task_id = str(deployment.get("task_id") or "").strip()
    if deployment_task_id and deployment_task_id != task_id:
        code = "active_deployment_incompatible" if deployment_id is None else "deployment_incompatible"
        raise api_error(status_code=409, code=code, message="Deployment is incompatible with this task")

    return deployment


async def _task_categories(project_id: str, task_id: str, db: AsyncSession) -> list[dict[str, str]]:
    categories = (
        await db.execute(
            select(Category.id, Category.name)
            .where(Category.project_id == project_id, Category.task_id == task_id)
            .order_by(Category.display_order, Category.id)
        )
    ).all()
    return [{"id": str(row[0]), "name": str(row[1])} for row in categories]


async def _run_classification_preview(
    *,
    db: AsyncSession,
    project_id: str,
    task: Task,
    storage_uri: str,
    deployment_id: str | None,
    top_k: int,
    preview_width: int | None,
    preview_height: int | None,
) -> PreviewInferenceResponse:
    if task.kind != TaskKind.classification:
        raise api_error(status_code=409, code="task_kind_unsupported", message="Classification preview requires a classification task")

    deployment = _resolve_compatible_deployment(
        project_id,
        task_id=task.id,
        task_kind="classification",
        deployment_id=deployment_id,
    )
    source = deployment.get("source")
    if not isinstance(source, dict):
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment source is invalid")

    metadata_relpath = str(source.get("metadata_relpath") or "").strip()
    onnx_relpath = str(source.get("onnx_relpath") or "").strip()
    if not metadata_relpath or not onnx_relpath:
        raise api_error(status_code=409, code="deployment_invalid", message="Deployment artifacts are unavailable")

    metadata = _read_deployment_metadata(metadata_relpath)
    categories = await _task_categories(project_id, task.id, db)
    class_ids = _resolve_class_mapping(metadata, categories=categories)
    category_name_by_id = {category["id"]: category["name"] for category in categories}

    response = await inference_client.infer_classification(
        {
            "onnx_relpath": onnx_relpath,
            "metadata_relpath": metadata_relpath,
            "asset_relpath": storage_uri,
            "device_preference": str(deployment.get("device_preference") or "auto"),
            "top_k": int(top_k),
            "model_key": deployment.get("model_key"),
        }
    )

    output_dim = response.get("output_dim")
    if isinstance(output_dim, int) and output_dim > len(class_ids):
        raise api_error(
            status_code=409,
            code="deployment_output_dim_mismatch",
            message="Inference output does not match deployment class_ids",
        )

    raw_predictions = list(response.get("predictions") or [])
    max_class_index = max(
        (int(row.get("class_index")) for row in raw_predictions if isinstance(row, dict) and isinstance(row.get("class_index"), int)),
        default=-1,
    )
    if max_class_index >= len(class_ids):
        raise api_error(
            status_code=409,
            code="deployment_output_dim_mismatch",
            message="Inference output does not match deployment class_ids",
        )

    predictions: list[PreviewInferencePredictionRead] = []
    for row in raw_predictions:
        if not isinstance(row, dict):
            continue
        class_index = row.get("class_index")
        if not isinstance(class_index, int) or class_index < 0 or class_index >= len(class_ids):
            continue
        class_id = class_ids[class_index]
        predictions.append(
            PreviewInferencePredictionRead(
                class_id=class_id,
                class_name=category_name_by_id.get(class_id, f"#{class_id}"),
                score=float(row.get("score") or 0.0),
            )
        )

    return PreviewInferenceResponse(
        task="classification",
        source_label=str(deployment.get("name") or "").strip() or "Active deployment",
        device_selected=str(response.get("device_selected") or "cpu"),
        preview_width=preview_width,
        preview_height=preview_height,
        predictions=predictions,
    )


async def _run_bbox_preview(
    *,
    db: AsyncSession,
    project_id: str,
    task: Task,
    storage_uri: str,
    config: PrelabelConfigCreate,
    preview_width: int | None,
    preview_height: int | None,
) -> PreviewInferenceResponse:
    if task.kind != TaskKind.bbox:
        raise api_error(status_code=409, code="task_kind_unsupported", message="Prelabels are supported only for bbox tasks")

    resolved = await resolve_prelabel_source_config(db, project_id=project_id, task=task, config=config)
    prompts = normalize_prompts(list(resolved.get("prompts") or []))
    source_label = str(resolved.get("source_label") or "").strip() or "AI preview"
    categories = await list_task_categories(db, project_id=project_id, task_id=task.id)
    exact_mapping, alias_mapping = category_match_maps(categories)

    response: dict[str, Any]
    if config.source_type == "active_deployment":
        deployment = resolved.get("deployment")
        if not isinstance(deployment, dict):
            raise api_error(status_code=409, code="active_deployment_not_found", message="Active deployment is unavailable for this task")
        source = deployment.get("source")
        if not isinstance(source, dict):
            raise api_error(status_code=409, code="deployment_invalid", message="Deployment source is invalid")
        response = await inference_client.infer_detection(
            {
                "onnx_relpath": source.get("onnx_relpath"),
                "metadata_relpath": source.get("metadata_relpath"),
                "asset_relpath": storage_uri,
                "device_preference": str(deployment.get("device_preference") or "auto"),
                "score_threshold": float(config.confidence_threshold),
                "model_key": deployment.get("model_key"),
            }
        )
        detections = [
            {
                "label_text": str(row.get("class_name") or "").strip(),
                "score": float(row.get("score") or 0.0),
                "bbox_xyxy": (
                    float(row["bbox"][0]),
                    float(row["bbox"][1]),
                    float(row["bbox"][0] + row["bbox"][2]),
                    float(row["bbox"][1] + row["bbox"][3]),
                ),
            }
            for row in list(response.get("boxes") or [])
            if isinstance(row, dict)
            and isinstance(row.get("bbox"), list)
            and len(row["bbox"]) == 4
            and all(isinstance(value, (int, float)) for value in row["bbox"])
        ]
    else:
        response = await inference_client.florence_detect(
            {
                "asset_relpath": storage_uri,
                "model_name": str(resolved.get("source_ref") or "microsoft/Florence-2-base-ft"),
                "prompts": prompts,
                "score_threshold": float(config.confidence_threshold),
                "max_detections": int(config.max_detections_per_frame),
            }
        )
        detections = [
            {
                "label_text": str(row.get("label_text") or "").strip(),
                "score": float(row.get("score") or 0.0),
                "bbox_xyxy": tuple(float(value) for value in row["bbox"]),
            }
            for row in list(response.get("boxes") or [])
            if isinstance(row, dict)
            and isinstance(row.get("bbox"), list)
            and len(row["bbox"]) == 4
            and all(isinstance(value, (int, float)) for value in row["bbox"])
        ]

    boxes: list[PreviewInferenceBoxRead] = []
    debug: list[PreviewInferenceDebugRead] = []
    for detection in detections[: int(config.max_detections_per_frame)]:
        bbox_xywh = bbox_xyxy_to_xywh(
            detection["bbox_xyxy"],
            width=preview_width,
            height=preview_height,
        )
        if bbox_xywh is None:
            debug.append(
                PreviewInferenceDebugRead(
                    label_text=detection["label_text"],
                    confidence=float(detection["score"]),
                    bbox=[],
                    status="discarded",
                )
            )
            continue
        category = match_detection_category(
            label_text=detection["label_text"],
            exact_mapping=exact_mapping,
            alias_mapping=alias_mapping,
        )
        matched = category is not None
        boxes.append(
            PreviewInferenceBoxRead(
                class_id=category.id if category is not None else None,
                class_name=category.name if category is not None else detection["label_text"],
                score=float(detection["score"]),
                bbox=bbox_xywh,
                matched=matched,
            )
        )
        if not matched:
            debug.append(
                PreviewInferenceDebugRead(
                    label_text=detection["label_text"],
                    confidence=float(detection["score"]),
                    bbox=bbox_xywh,
                    status="unmatched",
                )
            )

    return PreviewInferenceResponse(
        task="bbox",
        source_label=source_label,
        device_selected=str(response.get("device_selected") or "cpu"),
        preview_width=preview_width,
        preview_height=preview_height,
        boxes=boxes,
        debug=debug,
    )


async def run_preview_inference(
    *,
    db: AsyncSession,
    project_id: str,
    task: Task,
    task_kind: str,
    content: bytes,
    filename: str | None,
    content_type: str | None,
    prelabel_config_json: str | None = None,
    deployment_id: str | None = None,
    top_k: int = 5,
) -> PreviewInferenceResponse:
    if not content:
        raise api_error(status_code=400, code="uploaded_file_empty", message="Uploaded file is empty")

    normalized_task_kind = str(task_kind or "").strip().lower()
    if normalized_task_kind not in {"bbox", "classification"}:
        raise api_error(status_code=422, code="validation_error", message="task_kind must be 'bbox' or 'classification'")

    preview_width, preview_height = _image_dimensions_from_bytes(content)
    storage_uri = _preview_storage_uri(project_id, filename=filename, content_type=content_type)
    storage.write_bytes(storage_uri, content)
    try:
        if normalized_task_kind == "classification":
            return await _run_classification_preview(
                db=db,
                project_id=project_id,
                task=task,
                storage_uri=storage_uri,
                deployment_id=deployment_id,
                top_k=max(1, min(int(top_k), 100)),
                preview_width=preview_width,
                preview_height=preview_height,
            )

        if not isinstance(prelabel_config_json, str) or not prelabel_config_json.strip():
            raise api_error(status_code=422, code="validation_error", message="prelabel_config is required for bbox preview")

        try:
            config = PrelabelConfigCreate.model_validate_json(prelabel_config_json)
        except Exception as exc:
            raise api_error(status_code=422, code="validation_error", message="prelabel_config is invalid") from exc

        return await _run_bbox_preview(
            db=db,
            project_id=project_id,
            task=task,
            storage_uri=storage_uri,
            config=config,
            preview_width=preview_width,
            preview_height=preview_height,
        )
    finally:
        storage.delete_file(storage_uri)
