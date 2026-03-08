from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, AnnotationStatus, Asset, Category


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


@dataclass
class CategorySnapshot:
    class_order: list[str]
    classes: list[dict[str, Any]]
    class_names_by_id: dict[str, str]
    warnings: list[str]


@dataclass
class PreviewResolution:
    selected_rows: list[AssetRow]
    split_by_asset: dict[str, str]
    warnings: list[str]


def slugify_label(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "class"


def asset_relative_path(asset: Asset) -> str:
    metadata = asset.metadata_json if isinstance(asset.metadata_json, dict) else {}
    relative_path = metadata.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        return relative_path.replace("\\", "/").strip("/")
    original_filename = metadata.get("original_filename")
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.strip()
    return Path(asset.uri).name or asset.id


def asset_filename(asset: Asset) -> str:
    metadata = asset.metadata_json if isinstance(asset.metadata_json, dict) else {}
    original_filename = metadata.get("original_filename")
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.strip()
    return Path(asset_relative_path(asset)).name


def annotation_category_ids(payload_json: dict[str, Any]) -> list[str]:
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


def annotation_primary_category_id(payload_json: dict[str, Any], category_ids: list[str]) -> str | None:
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


def annotation_has_objects(payload_json: dict[str, Any]) -> bool:
    objects = payload_json.get("objects")
    return isinstance(objects, list) and len(objects) > 0


def folder_match(path: str, prefixes: list[str]) -> bool:
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


def apply_selection_filters(
    rows: list[AssetRow],
    *,
    mode: str,
    explicit_asset_ids: list[str],
    filters: dict[str, Any],
    task_kind: str,
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
        if include_folder_paths and not folder_match(row.relative_path, include_folder_paths):
            continue
        if exclude_folder_paths and folder_match(row.relative_path, exclude_folder_paths):
            continue
        if task_kind in {"bbox", "segmentation"} and include_negative_images is False and not row.has_objects:
            continue
        selected.append(row)
    return selected


def validate_split_ratios(ratios: dict[str, Any]) -> tuple[float, float, float]:
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


def allocate_split_counts(size: int, ratios: tuple[float, float, float]) -> dict[str, int]:
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


def random_split(asset_ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    shuffled = list(asset_ids)
    random.Random(seed).shuffle(shuffled)
    counts = allocate_split_counts(len(shuffled), ratios)
    split_by_asset: dict[str, str] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        take = counts[split]
        for asset_id in shuffled[cursor : cursor + take]:
            split_by_asset[asset_id] = split
        cursor += take
    return split_by_asset


def build_split_plan(
    rows: list[AssetRow],
    *,
    task_kind: str,
    seed: int,
    ratios: tuple[float, float, float],
    stratify_enabled: bool,
    strict_stratify: bool,
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    asset_ids = [row.asset.id for row in rows]
    if not asset_ids:
        return {}, warnings

    if task_kind != "classification" or not stratify_enabled:
        return random_split(asset_ids, ratios, seed), warnings

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
        return random_split(asset_ids, ratios, seed), warnings

    split_by_asset: dict[str, str] = {}
    for index, bucket in enumerate(sorted(buckets.keys())):
        bucket_asset_ids = list(buckets[bucket])
        random.Random(seed + index).shuffle(bucket_asset_ids)
        counts = allocate_split_counts(len(bucket_asset_ids), ratios)
        cursor = 0
        for split in SPLIT_ORDER:
            take = counts[split]
            for asset_id in bucket_asset_ids[cursor : cursor + take]:
                split_by_asset[asset_id] = split
            cursor += take
    return split_by_asset, warnings


async def load_asset_rows(db: AsyncSession, project_id: str, task_id: str) -> list[AssetRow]:
    assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id))).scalars().all())
    annotations = list(
        (await db.execute(select(Annotation).where(Annotation.project_id == project_id, Annotation.task_id == task_id))).scalars().all()
    )
    annotation_by_asset = {annotation.asset_id: annotation for annotation in annotations}

    rows: list[AssetRow] = []
    for asset in assets:
        annotation = annotation_by_asset.get(asset.id)
        payload_json = annotation.payload_json if annotation and isinstance(annotation.payload_json, dict) else {}
        category_ids = annotation_category_ids(payload_json)
        rows.append(
            AssetRow(
                asset=asset,
                annotation=annotation,
                status=annotation.status.value if annotation else AnnotationStatus.unlabeled.value,
                category_ids=category_ids,
                primary_category_id=annotation_primary_category_id(payload_json, category_ids),
                has_objects=annotation_has_objects(payload_json),
                relative_path=asset_relative_path(asset),
                filename=asset_filename(asset),
            )
        )
    rows.sort(key=lambda row: (row.relative_path, row.asset.id))
    return rows


def split_counts(split_by_asset: dict[str, str]) -> dict[str, int]:
    return {
        "train": sum(1 for value in split_by_asset.values() if value == "train"),
        "val": sum(1 for value in split_by_asset.values() if value == "val"),
        "test": sum(1 for value in split_by_asset.values() if value == "test"),
    }


def class_counts(rows: list[AssetRow], *, task_kind: str | None = None) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        if task_kind in {"bbox", "segmentation"} and not row.has_objects and not row.category_ids:
            continue
        key = row.primary_category_id or "__missing__"
        result[key] = int(result.get(key, 0)) + 1
    return result


def deleted_category_name(category_id: str) -> str:
    short = category_id[:8] if isinstance(category_id, str) else "unknown"
    return f"Deleted category ({short})"


def referenced_category_ids(rows: list[AssetRow]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for category_id in row.category_ids:
            if not isinstance(category_id, str):
                continue
            normalized = category_id.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


async def build_category_snapshot(db: AsyncSession, project_id: str, task_id: str, rows: list[AssetRow]) -> CategorySnapshot:
    categories = list((await db.execute(select(Category).where(Category.project_id == project_id, Category.task_id == task_id))).scalars().all())
    categories.sort(key=lambda category: (category.display_order, category.id))

    class_order: list[str] = []
    classes: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for category in categories:
        if isinstance(category.id, str) and category.id.strip() and isinstance(category.name, str) and category.name.strip():
            mapping[category.id] = category.name
            class_order.append(category.id)
            classes.append(
                {
                    "category_id": category.id,
                    "name": category.name,
                    "export_name": slugify_label(category.name),
                    "is_active": category.is_active,
                }
            )

    warnings: list[str] = []
    known_ids = set(mapping)
    orphan_ids = sorted(category_id for category_id in referenced_category_ids(rows) if category_id not in known_ids)
    if orphan_ids:
        warnings.append("Some annotations reference deleted or missing categories; dataset snapshot kept them as inactive placeholders.")
        for category_id in orphan_ids:
            synthetic_name = deleted_category_name(category_id)
            mapping[category_id] = synthetic_name
            class_order.append(category_id)
            classes.append(
                {
                    "category_id": category_id,
                    "name": synthetic_name,
                    "export_name": slugify_label(f"deleted_{category_id}"),
                    "is_active": False,
                }
            )

    return CategorySnapshot(
        class_order=class_order,
        classes=classes,
        class_names_by_id=mapping,
        warnings=warnings,
    )


def sample_asset_item(row: AssetRow, split_by_asset: dict[str, str]) -> dict[str, Any]:
    return {
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


def to_selection_payload(mode: str, filters: dict[str, Any], explicit_asset_ids: list[str]) -> dict[str, Any]:
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


async def resolve_dataset_preview(
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
) -> PreviewResolution:
    rows = await load_asset_rows(db, project_id, task_id)
    selected = apply_selection_filters(
        rows,
        mode=mode,
        explicit_asset_ids=explicit_asset_ids,
        filters=filters,
        task_kind=task_kind,
    )
    split_by_asset, warnings = build_split_plan(
        selected,
        task_kind=task_kind,
        seed=seed,
        ratios=ratios,
        stratify_enabled=stratify_enabled,
        strict_stratify=strict_stratify,
    )
    return PreviewResolution(selected_rows=selected, split_by_asset=split_by_asset, warnings=warnings)
