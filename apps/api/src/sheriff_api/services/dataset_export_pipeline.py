from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, Category, Project
from sheriff_api.services.exporter_coco import build_export_result
from sheriff_api.services.storage import LocalStorage


@dataclass
class ExportInputs:
    categories: list[Category]
    assets: list[Asset]
    annotations: list[Annotation]


@dataclass
class BuiltExportBundle:
    manifest: dict[str, Any]
    content_hash: str
    zip_bytes: bytes


def status_filter_from_selection(selection_criteria: dict[str, Any]) -> set[str] | None:
    statuses_raw = selection_criteria.get("statuses")
    if isinstance(statuses_raw, list):
        normalized = {str(value) for value in statuses_raw if str(value).strip()}
        return normalized if normalized else None

    status_raw = selection_criteria.get("status")
    if isinstance(status_raw, str) and status_raw.strip():
        return {status_raw}

    return None


async def prepare_export_inputs(
    db: AsyncSession,
    project_id: str,
    selection_criteria: dict[str, Any],
) -> ExportInputs:
    categories = list((await db.execute(select(Category).where(Category.project_id == project_id))).scalars().all())
    all_assets = list((await db.execute(select(Asset).where(Asset.project_id == project_id))).scalars().all())
    all_annotations = list((await db.execute(select(Annotation).where(Annotation.project_id == project_id))).scalars().all())

    status_filter = status_filter_from_selection(selection_criteria)
    if status_filter is None:
        selected_annotations = all_annotations
        selected_assets = all_assets
    else:
        selected_annotations = [annotation for annotation in all_annotations if annotation.status.value in status_filter]
        selected_asset_ids = {annotation.asset_id for annotation in selected_annotations}
        selected_assets = [asset for asset in all_assets if asset.id in selected_asset_ids]

    return ExportInputs(categories=categories, assets=selected_assets, annotations=selected_annotations)


def build_export_bundle(
    *,
    project: Project,
    selection_criteria: dict[str, Any],
    inputs: ExportInputs,
    storage: LocalStorage,
) -> BuiltExportBundle:
    storage.ensure_project_dirs(project.id)
    manifest, _coco, content_hash, zip_bytes = build_export_result(
        project_id=project.id,
        project_name=project.name,
        task_type=project.task_type,
        selection_criteria=selection_criteria,
        categories=[
            {"id": category.id, "name": category.name, "display_order": category.display_order, "is_active": category.is_active}
            for category in inputs.categories
        ],
        assets=[
            {
                "id": asset.id,
                "uri": asset.uri,
                "type": asset.type.value,
                "width": asset.width,
                "height": asset.height,
                "checksum": asset.checksum,
                "relative_path": _asset_metadata(asset).get("relative_path"),
                "original_filename": _asset_metadata(asset).get("original_filename"),
                "storage_uri": _asset_metadata(asset).get("storage_uri"),
                "extension": _asset_extension(asset),
            }
            for asset in inputs.assets
        ],
        annotations=[
            {
                "id": annotation.id,
                "asset_id": annotation.asset_id,
                "status": annotation.status.value,
                "payload": annotation.payload_json,
                "created_at": annotation.created_at,
                "updated_at": annotation.updated_at,
                "annotated_by": annotation.annotated_by,
            }
            for annotation in inputs.annotations
        ],
        load_asset_bytes=lambda asset: _load_asset_bytes(storage, asset),
    )
    return BuiltExportBundle(manifest=manifest, content_hash=content_hash, zip_bytes=zip_bytes)


def _asset_metadata(asset: Asset) -> dict[str, Any]:
    metadata = asset.metadata_json
    if isinstance(metadata, dict):
        return metadata
    return {}


def _asset_extension(asset: Asset) -> str:
    storage_uri = _asset_metadata(asset).get("storage_uri")
    return Path(str(storage_uri or "")).suffix.lower()


def _load_asset_bytes(storage: LocalStorage, asset: dict[str, Any]) -> bytes | None:
    storage_uri = asset.get("storage_uri")
    if not isinstance(storage_uri, str) or not storage_uri:
        return None
    try:
        path = storage.resolve(storage_uri)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()
