from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Asset, AssetType, Folder
from sheriff_api.services.folders import join_folder_and_file, sanitize_file_name
from sheriff_api.services.image_metadata import extract_image_dimensions
from sheriff_api.services.storage import LocalStorage


def build_asset_storage_uri(
    *,
    project_id: str,
    asset_id: str,
    file_name: str,
    sequence_id: str | None = None,
) -> str:
    safe_name = sanitize_file_name(file_name, fallback=f"{asset_id}{Path(file_name).suffix.lower()}")
    if isinstance(sequence_id, str) and sequence_id.strip():
        return f"assets/{project_id}/sequences/{sequence_id}/{safe_name}"
    suffix = Path(safe_name).suffix.lower()
    return f"assets/{project_id}/{asset_id}{suffix}"


def build_asset_record(
    *,
    project_id: str,
    content: bytes,
    file_name: str,
    mime_type: str,
    folder: Folder | None,
    original_filename: str | None = None,
    asset_type: AssetType = AssetType.image,
    sequence_id: str | None = None,
    sequence_name: str | None = None,
    source_kind: str = "image",
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
    asset_id: str | None = None,
) -> tuple[Asset, str]:
    generated_id = asset_id or str(uuid.uuid4())
    safe_file_name = sanitize_file_name(file_name, fallback=f"{generated_id}{Path(file_name).suffix.lower()}")
    relative_path = join_folder_and_file(folder.path if folder is not None else None, safe_file_name)
    storage_uri = build_asset_storage_uri(
        project_id=project_id,
        asset_id=generated_id,
        file_name=safe_file_name,
        sequence_id=sequence_id,
    )
    width, height = extract_image_dimensions(content)
    checksum = hashlib.sha256(content).hexdigest()
    metadata_json = {
        "storage_uri": storage_uri,
        "original_filename": original_filename or safe_file_name,
        "relative_path": relative_path,
        "size_bytes": len(content),
        "source_kind": source_kind,
    }
    if isinstance(sequence_id, str) and sequence_id.strip():
        metadata_json["sequence_id"] = sequence_id
    if isinstance(sequence_name, str) and sequence_name.strip():
        metadata_json["sequence_name"] = sequence_name
    if frame_index is not None:
        metadata_json["frame_index"] = frame_index
    if timestamp_seconds is not None:
        metadata_json["timestamp_seconds"] = timestamp_seconds

    asset = Asset(
        id=generated_id,
        project_id=project_id,
        type=asset_type,
        folder_id=folder.id if folder is not None else None,
        file_name=safe_file_name,
        sequence_id=sequence_id,
        source_kind=source_kind,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        uri=f"/api/v1/assets/{generated_id}/content",
        mime_type=mime_type or "application/octet-stream",
        width=width,
        height=height,
        checksum=checksum,
        metadata_json=metadata_json,
    )
    return asset, storage_uri


async def persist_asset_bytes(
    *,
    db: AsyncSession,
    storage: LocalStorage,
    project_id: str,
    content: bytes,
    file_name: str,
    mime_type: str,
    folder: Folder | None,
    original_filename: str | None = None,
    asset_type: AssetType = AssetType.image,
    sequence_id: str | None = None,
    sequence_name: str | None = None,
    source_kind: str = "image",
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
    commit: bool = True,
) -> Asset:
    asset, storage_uri = build_asset_record(
        project_id=project_id,
        content=content,
        file_name=file_name,
        mime_type=mime_type,
        folder=folder,
        original_filename=original_filename,
        asset_type=asset_type,
        sequence_id=sequence_id,
        sequence_name=sequence_name,
        source_kind=source_kind,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
    )

    wrote_file = False
    try:
        storage.write_bytes(storage_uri, content)
        wrote_file = True
        db.add(asset)
        if commit:
            await db.commit()
            await db.refresh(asset)
        else:
            await db.flush()
    except Exception:
        if commit:
            await db.rollback()
        if wrote_file:
            try:
                storage.delete_file(storage_uri)
            except ValueError:
                pass
        raise

    return asset
