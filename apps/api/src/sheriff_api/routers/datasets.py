from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
import re
from typing import Any
import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, AnnotationStatus, Asset, Category, Project, Task, TaskKind, TaskLabelMode, TaskType
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.datasets import (
    DatasetPreviewRequest,
    DatasetPreviewResponse,
    DatasetSetActiveRequest,
    DatasetVersionAssetsResponse,
    DatasetVersionCreateRequest,
    DatasetVersionEnvelope,
    DatasetVersionExportResponse,
    DatasetVersionListResponse,
)
from sheriff_api.services.dataset_store import DatasetStore, DatasetStoreValidationError
from sheriff_api.services.exporter_coco import ExportValidationError, build_export_result
from sheriff_api.services.storage import LocalStorage

router = APIRouter(tags=["datasets"])
settings = get_settings()
dataset_store = DatasetStore(settings.storage_root)
storage = LocalStorage(settings.storage_root)

SPLIT_ORDER = ("train", "val", "test")


@dataclass
class AssetRow:
    asset: Asset
    annotation: Annotation | None
    status: str
    category_ids: list[str]
    primary_category_id: str | None
    has_objects: bool
    relative_path: str
    filename: str


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "class"


def _asset_relative_path(asset: Asset) -> str:
    metadata = asset.metadata_json if isinstance(asset.metadata_json, dict) else {}
    relative_path = metadata.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        return relative_path.replace("\\", "/").strip("/")
    original_filename = metadata.get("original_filename")
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.strip()
    return Path(asset.uri).name or asset.id


def _asset_filename(asset: Asset) -> str:
    metadata = asset.metadata_json if isinstance(asset.metadata_json, dict) else {}
    original_filename = metadata.get("original_filename")
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.strip()
    return Path(_asset_relative_path(asset)).name


def _annotation_category_ids(payload_json: dict[str, Any]) -> list[str]:
    category_ids: list[str] = []
    seen: set[str] = set()

    def push(value: Any) -> None:
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        category_ids.append(normalized)

    if isinstance(payload_json.get("category_ids"), list):
        for value in payload_json["category_ids"]:
            push(value)
    push(payload_json.get("category_id"))

    classification = payload_json.get("classification")
    if isinstance(classification, dict):
        if isinstance(classification.get("category_ids"), list):
            for value in classification["category_ids"]:
                push(value)
        push(classification.get("primary_category_id"))

    objects = payload_json.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            push(item.get("category_id"))

    return category_ids


def _annotation_primary_category_id(payload_json: dict[str, Any], category_ids: list[str]) -> str | None:
    classification = payload_json.get("classification")
    if isinstance(classification, dict):
        value = classification.get("primary_category_id")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = payload_json.get("category_id")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if category_ids:
        return category_ids[0]
    return None


def _annotation_has_objects(payload_json: dict[str, Any]) -> bool:
    objects = payload_json.get("objects")
    return isinstance(objects, list) and len(objects) > 0


def _folder_match(path: str, prefixes: list[str]) -> bool:
    if not prefixes:
        return False
    normalized_path = path.replace("\\", "/").strip("/")
    for prefix in prefixes:
        normalized_prefix = prefix.replace("\\", "/").strip("/")
        if not normalized_prefix:
            continue
        if normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}/"):
            return True
    return False


def _apply_filters(
    rows: list[AssetRow],
    *,
    mode: str,
    explicit_asset_ids: list[str],
    filters: dict[str, Any],
    task: str,
) -> list[AssetRow]:
    if mode == "explicit_asset_ids":
        explicit_set = {str(value) for value in explicit_asset_ids}
        return [row for row in rows if row.asset.id in explicit_set]

    include_labeled_only = bool(filters.get("include_labeled_only"))
    include_statuses = {
        str(value) for value in filters.get("include_statuses", []) if str(value) in {status.value for status in AnnotationStatus}
    }
    exclude_statuses = {
        str(value) for value in filters.get("exclude_statuses", []) if str(value) in {status.value for status in AnnotationStatus}
    }
    include_category_ids = {str(value) for value in filters.get("include_category_ids", []) if str(value).strip()}
    exclude_category_ids = {str(value) for value in filters.get("exclude_category_ids", []) if str(value).strip()}
    include_folder_paths = [str(value) for value in filters.get("include_folder_paths", []) if str(value).strip()]
    exclude_folder_paths = [str(value) for value in filters.get("exclude_folder_paths", []) if str(value).strip()]
    include_negative_images = filters.get("include_negative_images")

    selected: list[AssetRow] = []
    for row in rows:
        if include_labeled_only and row.status == AnnotationStatus.unlabeled.value:
            continue
        if include_statuses and row.status not in include_statuses:
            continue
        if exclude_statuses and row.status in exclude_statuses:
            continue
        if include_category_ids and not (include_category_ids & set(row.category_ids)):
            continue
        if exclude_category_ids and (exclude_category_ids & set(row.category_ids)):
            continue
        # Include folders are a restrictive baseline when provided.
        # Exclude folders always subtract from that baseline.
        # final_membership = included_set - excluded_set (exclude wins)
        if include_folder_paths and not _folder_match(row.relative_path, include_folder_paths):
            continue
        if exclude_folder_paths and _folder_match(row.relative_path, exclude_folder_paths):
            continue
        if task in {"bbox", "segmentation"} and include_negative_images is False and not row.has_objects:
            continue
        selected.append(row)
    return selected


def _validate_split_ratios(ratios: dict[str, Any]) -> tuple[float, float, float]:
    train = float(ratios.get("train", 0.8))
    val = float(ratios.get("val", 0.1))
    test = float(ratios.get("test", 0.1))
    if min(train, val, test) < 0:
        raise ValueError("Split ratios must be >= 0")
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to > 0")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    return train, val, test


def _allocate_counts(size: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    raw = {
        "train": size * ratios[0],
        "val": size * ratios[1],
        "test": size * ratios[2],
    }
    counts = {split: int(raw[split]) for split in SPLIT_ORDER}
    used = sum(counts.values())
    remainder = size - used
    order = sorted(SPLIT_ORDER, key=lambda split: (raw[split] - counts[split], split), reverse=True)
    for split in order[:max(0, remainder)]:
        counts[split] += 1
    return counts


def _random_split(asset_ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    shuffled = list(asset_ids)
    random.Random(seed).shuffle(shuffled)
    counts = _allocate_counts(len(shuffled), ratios)
    split_by_asset: dict[str, str] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        take = counts[split]
        for asset_id in shuffled[cursor : cursor + take]:
            split_by_asset[asset_id] = split
        cursor += take
    return split_by_asset


def _build_splits(
    rows: list[AssetRow],
    *,
    task: str,
    seed: int,
    ratios: tuple[float, float, float],
    stratify_enabled: bool,
    strict_stratify: bool,
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    asset_ids = [row.asset.id for row in rows]
    if not asset_ids:
        return {}, warnings

    if task != "classification" or not stratify_enabled:
        return _random_split(asset_ids, ratios, seed), warnings

    buckets: dict[str, list[str]] = {}
    nonzero_splits = [split for split, ratio in zip(SPLIT_ORDER, ratios) if ratio > 0]
    for row in rows:
        key = row.primary_category_id or "__missing__"
        buckets.setdefault(key, []).append(row.asset.id)

    impossible = any(len(items) < len(nonzero_splits) for items in buckets.values()) and len(nonzero_splits) > 1
    if impossible:
        message = "Stratified split is impossible for rare classes; using seeded random split fallback."
        if strict_stratify:
            raise RuntimeError("dataset_stratify_impossible")
        warnings.append(message)
        return _random_split(asset_ids, ratios, seed), warnings

    split_by_asset: dict[str, str] = {}
    for index, bucket in enumerate(sorted(buckets.keys())):
        bucket_asset_ids = list(buckets[bucket])
        random.Random(seed + index).shuffle(bucket_asset_ids)
        counts = _allocate_counts(len(bucket_asset_ids), ratios)
        cursor = 0
        for split in SPLIT_ORDER:
            take = counts[split]
            for asset_id in bucket_asset_ids[cursor : cursor + take]:
                split_by_asset[asset_id] = split
            cursor += take
    return split_by_asset, warnings


async def _load_asset_rows(db: AsyncSession, project_id: str, task_id: str) -> list[AssetRow]:
    assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id))).scalars().all())
    annotations = list(
        (await db.execute(select(Annotation).where(Annotation.project_id == project_id, Annotation.task_id == task_id))).scalars().all()
    )
    annotation_by_asset = {annotation.asset_id: annotation for annotation in annotations}

    rows: list[AssetRow] = []
    for asset in assets:
        annotation = annotation_by_asset.get(asset.id)
        payload_json = annotation.payload_json if annotation and isinstance(annotation.payload_json, dict) else {}
        category_ids = _annotation_category_ids(payload_json)
        rows.append(
            AssetRow(
                asset=asset,
                annotation=annotation,
                status=annotation.status.value if annotation else AnnotationStatus.unlabeled.value,
                category_ids=category_ids,
                primary_category_id=_annotation_primary_category_id(payload_json, category_ids),
                has_objects=_annotation_has_objects(payload_json),
                relative_path=_asset_relative_path(asset),
                filename=_asset_filename(asset),
            )
        )
    rows.sort(key=lambda row: (row.relative_path, row.asset.id))
    return rows


def _split_counts(split_by_asset: dict[str, str]) -> dict[str, int]:
    return {
        "train": sum(1 for value in split_by_asset.values() if value == "train"),
        "val": sum(1 for value in split_by_asset.values() if value == "val"),
        "test": sum(1 for value in split_by_asset.values() if value == "test"),
    }


def _class_counts(rows: list[AssetRow]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = row.primary_category_id or "__missing__"
        result[key] = int(result.get(key, 0)) + 1
    return result


def _to_selection_payload(mode: str, filters: dict[str, Any], explicit_asset_ids: list[str]) -> dict[str, Any]:
    def strip_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip_none(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [strip_none(item) for item in value]
        return value

    payload: dict[str, Any] = {"mode": mode}
    if mode == "explicit_asset_ids":
        payload["explicit"] = {"asset_ids": list(dict.fromkeys([str(value) for value in explicit_asset_ids]))}
    else:
        payload["filters"] = strip_none(copy.deepcopy(filters))
    return payload


async def _resolve_preview(
    *,
    db: AsyncSession,
    project_id: str,
    task_id: str,
    task_kind: str,
    mode: str,
    filters: dict[str, Any],
    explicit_asset_ids: list[str],
    seed: int,
    ratios: tuple[float, float, float],
    stratify_enabled: bool,
    strict_stratify: bool,
) -> tuple[list[AssetRow], dict[str, str], list[str]]:
    rows = await _load_asset_rows(db, project_id, task_id)
    selected = _apply_filters(rows, mode=mode, explicit_asset_ids=explicit_asset_ids, filters=filters, task=task_kind)
    split_by_asset, warnings = _build_splits(
        selected,
        task=task_kind,
        seed=seed,
        ratios=ratios,
        stratify_enabled=stratify_enabled,
        strict_stratify=strict_stratify,
    )
    return selected, split_by_asset, warnings


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


async def _require_task(db: AsyncSession, project_id: str, task_id: str) -> Task:
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )
    return task


def _task_type_for_task(task: Task) -> TaskType:
    if task.kind == TaskKind.classification:
        if task.label_mode == TaskLabelMode.multi_label:
            return TaskType.classification
        return TaskType.classification_single
    if task.kind == TaskKind.bbox:
        return TaskType.bbox
    return TaskType.segmentation


def _map_uuid_payload_to_coco_int(payload_json: dict[str, Any], class_to_coco: dict[str, int]) -> dict[str, Any]:
    payload = copy.deepcopy(payload_json)

    def map_one(value: Any) -> Any:
        if isinstance(value, str) and value in class_to_coco:
            return class_to_coco[value]
        return value

    def map_many(values: Any) -> Any:
        if not isinstance(values, list):
            return values
        mapped: list[int] = []
        for value in values:
            converted = map_one(value)
            if isinstance(converted, int):
                mapped.append(converted)
        return mapped

    payload["category_id"] = map_one(payload.get("category_id"))
    payload["category_ids"] = map_many(payload.get("category_ids"))

    classification = payload.get("classification")
    if isinstance(classification, dict):
        classification["primary_category_id"] = map_one(classification.get("primary_category_id"))
        classification["category_ids"] = map_many(classification.get("category_ids"))

    coco = payload.get("coco")
    if isinstance(coco, dict):
        coco["category_id"] = map_one(coco.get("category_id"))

    objects = payload.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            item["category_id"] = map_one(item.get("category_id"))
    return payload


async def _build_dataset_export(
    *,
    db: AsyncSession,
    project: Project,
    task: Task,
    dataset_version: dict[str, Any],
) -> tuple[str, str]:
    class_order = dataset_version.get("labels", {}).get("label_schema", {}).get("class_order")
    classes = dataset_version.get("labels", {}).get("label_schema", {}).get("classes")
    if not isinstance(class_order, list) or not isinstance(classes, list):
        raise api_error(
            status_code=422,
            code="dataset_split_invalid",
            message="Dataset version label snapshot is invalid",
            details={"dataset_version_id": dataset_version.get("dataset_version_id")},
        )

    class_to_coco: dict[str, int] = {}
    categories_for_export: list[dict[str, Any]] = []
    class_name_by_id: dict[str, str] = {}
    for class_row in classes:
        if not isinstance(class_row, dict):
            continue
        category_id = class_row.get("category_id")
        name = class_row.get("name")
        if isinstance(category_id, str) and isinstance(name, str):
            class_name_by_id[category_id] = name
    for index, category_id in enumerate(class_order):
        if not isinstance(category_id, str):
            continue
        class_to_coco[category_id] = index + 1
        categories_for_export.append(
            {
                "id": index + 1,
                "stable_id": category_id,
                "name": class_name_by_id.get(category_id, f"class_{index + 1}"),
                "display_order": index,
                "is_active": True,
            }
        )

    asset_ids = dataset_version.get("assets", {}).get("asset_ids")
    if not isinstance(asset_ids, list):
        asset_ids = []
    selected_asset_ids = {str(asset_id) for asset_id in asset_ids}

    assets = list((await db.execute(select(Asset).where(Asset.project_id == project.id))).scalars().all())
    annotations = list(
        (await db.execute(select(Annotation).where(Annotation.project_id == project.id, Annotation.task_id == task.id))).scalars().all()
    )
    selected_assets = [asset for asset in assets if asset.id in selected_asset_ids]
    selected_annotations = [annotation for annotation in annotations if annotation.asset_id in selected_asset_ids]

    try:
        manifest, _coco, content_hash, zip_bytes = build_export_result(
            project_id=project.id,
            project_name=project.name,
            task_type=_task_type_for_task(task),
            selection_criteria=dataset_version.get("selection", {}).get("filters", {}),
            categories=categories_for_export,
            assets=[
                {
                    "id": asset.id,
                    "uri": asset.uri,
                    "type": asset.type.value,
                    "width": asset.width,
                    "height": asset.height,
                    "checksum": asset.checksum,
                    "relative_path": (asset.metadata_json or {}).get("relative_path")
                    if isinstance(asset.metadata_json, dict)
                    else None,
                    "original_filename": (asset.metadata_json or {}).get("original_filename")
                    if isinstance(asset.metadata_json, dict)
                    else None,
                    "storage_uri": (asset.metadata_json or {}).get("storage_uri")
                    if isinstance(asset.metadata_json, dict)
                    else None,
                    "extension": Path(str((asset.metadata_json or {}).get("storage_uri", ""))).suffix.lower()
                    if isinstance(asset.metadata_json, dict)
                    else "",
                }
                for asset in selected_assets
            ],
            annotations=[
                {
                    "id": annotation.id,
                    "asset_id": annotation.asset_id,
                    "status": annotation.status.value,
                    "payload": _map_uuid_payload_to_coco_int(
                        annotation.payload_json if isinstance(annotation.payload_json, dict) else {},
                        class_to_coco,
                    ),
                    "created_at": annotation.created_at,
                    "updated_at": annotation.updated_at,
                    "annotated_by": annotation.annotated_by,
                }
                for annotation in selected_annotations
            ],
            load_asset_bytes=lambda asset: _load_asset_bytes(storage, asset),
        )
    except ExportValidationError as exc:
        raise api_error(status_code=422, code=exc.code, message=exc.message, details=exc.details) from exc

    storage_uri = f"exports/{project.id}/{content_hash}.zip"
    if not storage.resolve(storage_uri).exists():
        storage.write_bytes(storage_uri, zip_bytes)
    export_uri = f"/api/v1/projects/{project.id}/datasets/versions/{dataset_version['dataset_version_id']}/export/download"
    return content_hash, export_uri


def _load_asset_bytes(local_storage: LocalStorage, asset: dict[str, Any]) -> bytes | None:
    storage_uri = asset.get("storage_uri")
    if not isinstance(storage_uri, str) or not storage_uri:
        return None
    try:
        path = local_storage.resolve(storage_uri)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()


@router.get("/projects/{project_id}/datasets/versions", response_model=DatasetVersionListResponse)
async def list_dataset_versions(project_id: str, task_id: str | None = None) -> DatasetVersionListResponse:
    listed = dataset_store.list_versions(project_id, task_id=task_id)
    return DatasetVersionListResponse(
        active_dataset_version_id=listed["active_dataset_version_id"],
        items=[DatasetVersionEnvelope(**item) for item in listed["items"]],
    )


@router.post("/projects/{project_id}/datasets/versions/preview", response_model=DatasetPreviewResponse)
async def preview_dataset_version(
    project_id: str,
    payload: DatasetPreviewRequest,
    db: AsyncSession = Depends(get_db),
) -> DatasetPreviewResponse:
    await _require_project(db, project_id)
    task = await _require_task(db, project_id, payload.task_id)
    try:
        ratios = _validate_split_ratios(payload.split.ratios.model_dump())
    except ValueError as exc:
        raise api_error(status_code=422, code="dataset_split_invalid", message=str(exc)) from exc

    try:
        selected, split_by_asset, warnings = await _resolve_preview(
            db=db,
            project_id=project_id,
            task_id=payload.task_id,
            task_kind=task.kind.value,
            mode=payload.selection.mode,
            filters=payload.selection.filters.model_dump(),
            explicit_asset_ids=payload.selection.explicit_asset_ids,
            seed=payload.split.seed,
            ratios=ratios,
            stratify_enabled=payload.split.stratify.enabled,
            strict_stratify=payload.split.stratify.strict_stratify,
        )
    except RuntimeError as exc:
        if str(exc) == "dataset_stratify_impossible":
            raise api_error(
                status_code=422,
                code="dataset_stratify_impossible",
                message="Stratified split cannot be satisfied with current selection",
            ) from exc
        raise

    all_asset_ids = [row.asset.id for row in selected]
    if payload.strict_preview_cap and len(all_asset_ids) > payload.preview_cap:
        raise api_error(
            status_code=422,
            code="dataset_preview_too_large",
            message="Preview result exceeds cap",
            details={"cap": payload.preview_cap, "count": len(all_asset_ids)},
        )

    return DatasetPreviewResponse(
        asset_ids=all_asset_ids[: payload.preview_cap],
        sample_asset_ids=all_asset_ids[: min(120, payload.preview_cap)],
        counts={
            "total": len(all_asset_ids),
            "class_counts": _class_counts(selected),
            "split_counts": _split_counts(split_by_asset),
        },
        warnings=warnings,
    )


@router.post("/projects/{project_id}/datasets/versions", response_model=DatasetVersionEnvelope)
async def create_dataset_version(
    project_id: str,
    payload: DatasetVersionCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionEnvelope:
    project = await _require_project(db, project_id)
    task = await _require_task(db, project_id, payload.task_id)
    try:
        ratios = _validate_split_ratios(payload.split.ratios.model_dump())
    except ValueError as exc:
        raise api_error(status_code=422, code="dataset_split_invalid", message=str(exc)) from exc

    try:
        selected, split_by_asset, warnings = await _resolve_preview(
            db=db,
            project_id=project_id,
            task_id=payload.task_id,
            task_kind=task.kind.value,
            mode=payload.selection.mode,
            filters=payload.selection.filters.model_dump(),
            explicit_asset_ids=payload.selection.explicit_asset_ids,
            seed=payload.split.seed,
            ratios=ratios,
            stratify_enabled=payload.split.stratify.enabled,
            strict_stratify=payload.split.stratify.strict_stratify,
        )
    except RuntimeError as exc:
        if str(exc) == "dataset_stratify_impossible":
            raise api_error(
                status_code=422,
                code="dataset_stratify_impossible",
                message="Stratified split cannot be satisfied with current selection",
            ) from exc
        raise

    categories = list(
        (await db.execute(select(Category).where(Category.project_id == project_id, Category.task_id == payload.task_id))).scalars().all()
    )
    categories.sort(key=lambda category: (category.display_order, category.id))
    class_order = [category.id for category in categories]
    classes = [
        {
            "category_id": category.id,
            "name": category.name,
            "export_name": _slug(category.name),
            "is_active": category.is_active,
        }
        for category in categories
    ]
    selected_asset_ids = [row.asset.id for row in selected]
    split_items = [
        {"asset_id": asset_id, "split": split_by_asset.get(asset_id, "train")}
        for asset_id in selected_asset_ids
    ]
    dataset_version_payload = {
        "schema_version": "2.0",
        "dataset_version_id": str(uuid.uuid4()),
        "project_id": project_id,
        "task_id": payload.task_id,
        "name": payload.name,
        "task": task.kind.value,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"source_type": "project_assets_snapshot"},
        "assets": {
            "count": len(selected_asset_ids),
            "asset_ids": selected_asset_ids,
        },
        "labels": {
            "label_schema": {
                "version": "1.2",
                "class_order": class_order,
                "classes": classes,
                "rules": {"names_normalized": "lowercase_slug"},
            },
        },
        "selection": _to_selection_payload(
            payload.selection.mode,
            payload.selection.filters.model_dump(mode="json", exclude_none=True),
            payload.selection.explicit_asset_ids,
        ),
        "splits": {
            "strategy": "random_seeded",
            "seed": payload.split.seed,
            "ratios": payload.split.ratios.model_dump(),
            "stratify": {
                "enabled": payload.split.stratify.enabled,
                "by": payload.split.stratify.by,
            },
            "items": split_items,
        },
        "stats": {
            "asset_count": len(selected_asset_ids),
            "class_counts": _class_counts(selected),
            "split_counts": _split_counts(split_by_asset),
            "warnings": warnings,
        },
    }
    if isinstance(payload.description, str) and payload.description.strip():
        dataset_version_payload["description"] = payload.description
    if isinstance(payload.created_by, str) and payload.created_by.strip():
        dataset_version_payload["created_by"] = payload.created_by
    if task.kind == TaskKind.classification and task.label_mode is not None:
        dataset_version_payload["labels"]["label_mode"] = task.label_mode.value
    try:
        created = dataset_store.create_version(project_id, dataset_version_payload)
    except DatasetStoreValidationError as exc:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Dataset version validation failed",
            details={"issues": exc.issues},
        ) from exc

    current = dataset_store.list_versions(project_id)
    should_set_active = payload.set_active or not current.get("active_dataset_version_id")
    if should_set_active:
        dataset_store.set_active(project_id, created["dataset_version_id"])
    loaded = dataset_store.get_version(project_id, created["dataset_version_id"])
    return DatasetVersionEnvelope(**loaded)


@router.patch("/projects/{project_id}/datasets/active")
async def set_active_dataset_version(project_id: str, payload: DatasetSetActiveRequest) -> dict[str, Any]:
    try:
        dataset_store.set_active(project_id, payload.active_dataset_version_id)
    except KeyError as exc:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": payload.active_dataset_version_id},
        ) from exc
    return {"ok": True, "active_dataset_version_id": payload.active_dataset_version_id}


@router.get("/projects/{project_id}/datasets/versions/{dataset_version_id}", response_model=DatasetVersionEnvelope)
async def get_dataset_version(project_id: str, dataset_version_id: str) -> DatasetVersionEnvelope:
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    return DatasetVersionEnvelope(**loaded)


@router.get("/projects/{project_id}/datasets/versions/{dataset_version_id}/assets", response_model=DatasetVersionAssetsResponse)
async def list_dataset_version_assets(
    project_id: str,
    dataset_version_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=250),
    split: str | None = None,
    status: AnnotationStatus | None = None,
    class_id: str | None = None,
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionAssetsResponse:
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    version = loaded["version"]
    task_id = str(version.get("task_id") or "")
    if not task_id:
        raise api_error(
            status_code=422,
            code="dataset_version_invalid",
            message="Dataset version is missing task_id",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    version_asset_ids = version.get("assets", {}).get("asset_ids")
    split_items = version.get("splits", {}).get("items")
    if not isinstance(version_asset_ids, list):
        version_asset_ids = []
    split_by_asset = {}
    if isinstance(split_items, list):
        for item in split_items:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id")
            split_value = item.get("split")
            if isinstance(asset_id, str) and isinstance(split_value, str):
                split_by_asset[asset_id] = split_value

    rows = await _load_asset_rows(db, project_id, task_id)
    version_asset_id_set = {str(asset_id) for asset_id in version_asset_ids}
    # For saved dataset versions, membership comes only from stored version assets/splits.
    # Live DB rows are used only to enrich display fields such as status/labels/path.
    scoped = [row for row in rows if row.asset.id in version_asset_id_set]
    if split in {"train", "val", "test"}:
        scoped = [row for row in scoped if split_by_asset.get(row.asset.id) == split]
    if status is not None:
        scoped = [row for row in scoped if row.status == status.value]
    if isinstance(class_id, str) and class_id.strip():
        normalized = class_id.strip()
        scoped = [row for row in scoped if normalized in row.category_ids]
    if isinstance(search, str) and search.strip():
        needle = search.strip().lower()
        scoped = [
            row
            for row in scoped
            if needle in row.filename.lower() or needle in row.relative_path.lower() or needle in row.asset.id.lower()
        ]

    total = len(scoped)
    start = (page - 1) * page_size
    end = start + page_size
    paged = scoped[start:end]

    return DatasetVersionAssetsResponse(
        items=[
            {
                "asset_id": row.asset.id,
                "filename": row.filename,
                "relative_path": row.relative_path,
                "status": row.status,
                "split": split_by_asset.get(row.asset.id),
                "label_summary": {
                    "primary_category_id": row.primary_category_id,
                    "category_ids": row.category_ids,
                },
            }
            for row in paged
        ],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("/projects/{project_id}/datasets/versions/{dataset_version_id}/export", response_model=DatasetVersionExportResponse)
async def export_dataset_version(
    project_id: str,
    dataset_version_id: str,
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionExportResponse:
    project = await _require_project(db, project_id)
    loaded = dataset_store.get_version(project_id, dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )

    existing_artifact = dataset_store.get_export_artifact(project_id, dataset_version_id)
    if isinstance(existing_artifact, dict):
        existing_hash = existing_artifact.get("hash")
        if isinstance(existing_hash, str):
            relpath = f"exports/{project_id}/{existing_hash}.zip"
            if storage.resolve(relpath).exists():
                return DatasetVersionExportResponse(
                    dataset_version_id=dataset_version_id,
                    hash=existing_hash,
                    export_uri=str(existing_artifact.get("export_uri")),
                )

    version = loaded["version"]
    task_id = str(version.get("task_id") or "")
    if not task_id:
        raise api_error(
            status_code=422,
            code="dataset_version_invalid",
            message="Dataset version is missing task_id",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    task = await _require_task(db, project_id, task_id)

    content_hash, export_uri = await _build_dataset_export(
        db=db,
        project=project,
        task=task,
        dataset_version=version,
    )
    dataset_store.set_export_artifact(
        project_id,
        dataset_version_id,
        {
            "hash": content_hash,
            "export_uri": export_uri,
        },
    )
    return DatasetVersionExportResponse(
        dataset_version_id=dataset_version_id,
        hash=content_hash,
        export_uri=export_uri,
    )


@router.get("/projects/{project_id}/datasets/versions/{dataset_version_id}/export/download")
async def download_dataset_version_export(project_id: str, dataset_version_id: str) -> FileResponse:
    artifact = dataset_store.get_export_artifact(project_id, dataset_version_id)
    if not isinstance(artifact, dict):
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    content_hash = artifact.get("hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    relpath = f"exports/{project_id}/{content_hash}.zip"
    try:
        path = storage.resolve(relpath)
    except ValueError as exc:
        raise api_error(
            status_code=400,
            code="export_path_invalid",
            message="Invalid export path",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id, "reason": str(exc)},
        ) from exc
    if not path.exists() or not path.is_file():
        raise api_error(
            status_code=404,
            code="export_file_not_found",
            message="Export file not found",
            details={"project_id": project_id, "dataset_version_id": dataset_version_id},
        )
    return FileResponse(
        path=path,
        media_type="application/zip",
        filename=f"{project_id}-{dataset_version_id[:8]}-{content_hash[:8]}.zip",
    )
