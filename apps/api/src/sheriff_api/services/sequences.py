from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Annotation, Asset, AssetSequence, Folder
from sheriff_api.services.folders import ensure_folder_path, ensure_unique_folder_path, sanitize_folder_name
from sheriff_api.schemas.assets import AssetRead
from sheriff_api.schemas.folders import FolderRead
from sheriff_api.schemas.sequences import AssetSequenceRead, SequenceFrameAssetRead, SequenceStatusRead


def source_kind_for_sequence_source(source_type: str) -> str:
    return "webcam_frame" if source_type == "webcam" else "video_frame"


def asset_to_read(asset: Asset) -> AssetRead:
    return AssetRead(
        id=asset.id,
        project_id=asset.project_id,
        type=asset.type,
        folder_id=asset.folder_id,
        folder_path=asset.folder_path,
        file_name=asset.resolved_file_name,
        relative_path=asset.relative_path,
        sequence_id=asset.sequence_id,
        source_kind=asset.source_kind,
        frame_index=asset.frame_index,
        timestamp_seconds=asset.timestamp_seconds,
        uri=asset.uri,
        mime_type=asset.mime_type,
        width=asset.width,
        height=asset.height,
        checksum=asset.checksum,
        metadata_json=asset.metadata_json,
    )


def sequence_frame_to_read(asset: Asset, *, has_annotations: bool = False) -> SequenceFrameAssetRead:
    return SequenceFrameAssetRead(
        id=asset.id,
        file_name=asset.resolved_file_name,
        folder_id=asset.folder_id,
        folder_path=asset.folder_path,
        relative_path=asset.relative_path,
        source_kind=asset.source_kind,
        frame_index=asset.frame_index,
        timestamp_seconds=asset.timestamp_seconds,
        image_url=asset.uri,
        thumbnail_url=asset.uri,
        has_annotations=has_annotations,
    )


def sequence_to_read(
    sequence: AssetSequence,
    *,
    folder: Folder | None = None,
    assets: Iterable[Asset] = (),
    annotated_asset_ids: set[str] | None = None,
) -> AssetSequenceRead:
    folder_path = folder.path if folder is not None else sequence.folder.path if sequence.folder is not None else None
    annotations = annotated_asset_ids or set()
    ordered_assets = sorted(
        list(assets),
        key=lambda asset: (
            asset.frame_index if asset.frame_index is not None else 10**9,
            asset.resolved_file_name,
            asset.id,
        ),
    )
    return AssetSequenceRead(
        id=sequence.id,
        project_id=sequence.project_id,
        task_id=sequence.task_id,
        folder_id=sequence.folder_id,
        folder_path=folder_path,
        name=sequence.name,
        source_type=sequence.source_type,
        source_filename=sequence.source_filename,
        status=sequence.status,
        frame_count=sequence.frame_count,
        processed_frames=sequence.processed_frames,
        fps=sequence.fps,
        duration_seconds=sequence.duration_seconds,
        width=sequence.width,
        height=sequence.height,
        error_message=sequence.error_message,
        assets=[sequence_frame_to_read(asset, has_annotations=asset.id in annotations) for asset in ordered_assets],
    )


def sequence_status_to_read(sequence: AssetSequence) -> SequenceStatusRead:
    return SequenceStatusRead(
        id=sequence.id,
        status=sequence.status,
        frame_count=sequence.frame_count,
        processed_frames=sequence.processed_frames,
        error_message=sequence.error_message,
    )


def folder_to_read(
    folder: Folder,
    *,
    asset_count: int = 0,
    sequence: AssetSequence | None = None,
) -> FolderRead:
    return FolderRead(
        id=folder.id,
        project_id=folder.project_id,
        parent_id=folder.parent_id,
        name=folder.name,
        path=folder.path,
        asset_count=asset_count,
        sequence_id=sequence.id if sequence is not None else None,
        sequence_name=sequence.name if sequence is not None else None,
        sequence_source_type=sequence.source_type if sequence is not None else None,
        sequence_status=sequence.status if sequence is not None else None,
        sequence_frame_count=sequence.frame_count if sequence is not None else None,
        sequence_processed_frames=sequence.processed_frames if sequence is not None else None,
    )


async def refresh_sequence_counts(db: AsyncSession, sequence_id: str) -> AssetSequence | None:
    sequence = await db.get(AssetSequence, sequence_id)
    if sequence is None:
        return None

    result = await db.execute(
        select(
            func.count(Asset.id),
            func.max(Asset.frame_index),
        ).where(Asset.sequence_id == sequence_id)
    )
    count, _max_frame_index = result.one()
    count_int = int(count or 0)
    sequence.frame_count = count_int
    sequence.processed_frames = count_int
    return sequence


async def annotated_asset_ids_for_sequence(db: AsyncSession, *, task_id: str | None, asset_ids: list[str]) -> set[str]:
    if not task_id or not asset_ids:
        return set()
    result = await db.execute(
        select(Annotation.asset_id).where(
            Annotation.task_id == task_id,
            Annotation.asset_id.in_(asset_ids),
        )
    )
    return {str(value) for value in result.scalars().all()}


async def create_sequence_with_folder(
    db: AsyncSession,
    *,
    project_id: str,
    task_id: str | None,
    folder_id: str | None,
    folder_path: str | None = None,
    requested_name: str,
    source_type: str,
    source_filename: str | None = None,
    status: str,
    fps: float | None = None,
) -> tuple[Folder, AssetSequence]:
    if folder_id:
        folder = await db.get(Folder, folder_id)
        if folder is None or folder.project_id != project_id:
            raise ValueError("folder_not_found")
        existing_sequence = (
            await db.execute(select(AssetSequence).where(AssetSequence.folder_id == folder_id))
        ).scalar_one_or_none()
        if existing_sequence is not None:
            raise ValueError("folder_sequence_exists")
        asset_exists = (
            await db.execute(select(Asset.id).where(Asset.folder_id == folder_id).limit(1))
        ).scalar_one_or_none()
        if asset_exists is not None:
            raise ValueError("folder_not_empty")
    elif folder_path:
        folder = await ensure_folder_path(db, project_id, folder_path)
        if folder is None:
            raise ValueError("folder_create_failed")
        existing_sequence = (
            await db.execute(select(AssetSequence).where(AssetSequence.folder_id == folder.id))
        ).scalar_one_or_none()
        if existing_sequence is not None:
            raise ValueError("folder_sequence_exists")
        asset_exists = (
            await db.execute(select(Asset.id).where(Asset.folder_id == folder.id).limit(1))
        ).scalar_one_or_none()
        if asset_exists is not None:
            raise ValueError("folder_not_empty")
    else:
        folder_leaf = sanitize_folder_name(requested_name, fallback=source_type)
        folder_path = await ensure_unique_folder_path(db, project_id, folder_leaf)
        folder = await ensure_folder_path(db, project_id, folder_path)
        if folder is None:
            raise ValueError("folder_create_failed")

    sequence = AssetSequence(
        project_id=project_id,
        task_id=task_id,
        folder_id=folder.id,
        name=requested_name,
        source_type=source_type,
        source_filename=source_filename,
        status=status,
        frame_count=0,
        processed_frames=0,
        fps=fps,
    )
    db.add(sequence)
    await db.flush()
    return folder, sequence
