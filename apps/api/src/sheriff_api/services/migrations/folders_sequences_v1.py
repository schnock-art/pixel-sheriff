from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sheriff_api.services.migrations.shared import (
    _add_column_if_missing,
    _coerce_json_dict,
    _legacy_asset_relative_path_for_migration,
    _utc_now_dt,
)


async def _ensure_folders_sequences_schema(conn: AsyncConnection) -> None:
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS folders (
                id VARCHAR NOT NULL PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                parent_id VARCHAR NULL,
                name VARCHAR NOT NULL,
                path VARCHAR NOT NULL,
                created_at DATETIME NULL,
                updated_at DATETIME NULL
            )
            """
        )
    )
    await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_folder_project_path ON folders (project_id, path)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_folders_project_id ON folders (project_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_folders_parent_id ON folders (parent_id)"))

    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS asset_sequences (
                id VARCHAR NOT NULL PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                task_id VARCHAR NULL,
                folder_id VARCHAR NULL,
                name VARCHAR NOT NULL,
                source_type VARCHAR NOT NULL,
                source_filename VARCHAR NULL,
                status VARCHAR NOT NULL,
                frame_count INTEGER NULL,
                processed_frames INTEGER NULL,
                fps FLOAT NULL,
                duration_seconds FLOAT NULL,
                width INTEGER NULL,
                height INTEGER NULL,
                error_message VARCHAR NULL,
                created_at DATETIME NULL,
                updated_at DATETIME NULL
            )
            """
        )
    )
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_asset_sequences_project_id ON asset_sequences (project_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_asset_sequences_task_id ON asset_sequences (task_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_asset_sequences_folder_id ON asset_sequences (folder_id)"))
    await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_asset_sequences_folder_id ON asset_sequences (folder_id)"))

    await _add_column_if_missing(conn, "assets", "folder_id", "folder_id VARCHAR NULL")
    await _add_column_if_missing(conn, "assets", "file_name", "file_name VARCHAR NULL")
    await _add_column_if_missing(conn, "assets", "sequence_id", "sequence_id VARCHAR NULL")
    await _add_column_if_missing(conn, "assets", "source_kind", "source_kind VARCHAR NOT NULL DEFAULT 'image'")
    await _add_column_if_missing(conn, "assets", "frame_index", "frame_index INTEGER NULL")
    await _add_column_if_missing(conn, "assets", "timestamp_seconds", "timestamp_seconds FLOAT NULL")
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_assets_folder_id ON assets (folder_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_assets_sequence_id ON assets (sequence_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_assets_frame_index ON assets (frame_index)"))


async def _ensure_folder_row_for_migration(
    conn: AsyncConnection,
    *,
    folder_insert,
    existing_by_project: dict[str, dict[str, str]],
    project_id: str,
    folder_path: str | None,
) -> str | None:
    normalized = str(folder_path or "").replace("\\", "/").strip("/")
    if not normalized:
        return None
    project_folders = existing_by_project.setdefault(project_id, {})
    if normalized in project_folders:
        return project_folders[normalized]

    parent_id: str | None = None
    cursor = ""
    for segment in normalized.split("/"):
        cursor = f"{cursor}/{segment}" if cursor else segment
        current_id = project_folders.get(cursor)
        if current_id is None:
            current_id = str(uuid.uuid4())
            await conn.execute(
                folder_insert,
                {
                    "id": current_id,
                    "project_id": project_id,
                    "parent_id": parent_id,
                    "name": segment,
                    "path": cursor,
                    "created_at": _utc_now_dt(),
                    "updated_at": _utc_now_dt(),
                },
            )
            project_folders[cursor] = current_id
        parent_id = current_id
    return parent_id


async def _backfill_folders_and_assets(conn: AsyncConnection) -> None:
    from sheriff_api.db.models import Asset, Folder

    folder_insert = Folder.__table__.insert()
    asset_update = (
        Asset.__table__
        .update()
        .where(Asset.id == bindparam("asset_id"))
        .values(
            folder_id=bindparam("folder_id"),
            file_name=bindparam("file_name"),
            source_kind=bindparam("source_kind"),
            metadata_json=bindparam("metadata_json"),
        )
    )

    folder_rows = (await conn.execute(text("SELECT id, project_id, path FROM folders"))).mappings().all()
    existing_by_project: dict[str, dict[str, str]] = {}
    for row in folder_rows:
        project_id = str(row["project_id"])
        path = str(row["path"])
        existing_by_project.setdefault(project_id, {})[path] = str(row["id"])

    asset_rows = (
        await conn.execute(
            text(
                """
                SELECT id, project_id, uri, metadata_json, folder_id, file_name, source_kind
                FROM assets
                """
            )
        )
    ).mappings().all()

    updates: list[dict[str, Any]] = []
    for row in asset_rows:
        asset_id = str(row["id"])
        project_id = str(row["project_id"])
        metadata_json = _coerce_json_dict(row["metadata_json"])
        legacy_relative_path = _legacy_asset_relative_path_for_migration(metadata_json, str(row["uri"] or ""), asset_id)
        normalized_path = legacy_relative_path.replace("\\", "/").strip("/")
        folder_path = str(Path(normalized_path).parent).replace("\\", "/").strip("/")
        if folder_path in {"", "."}:
            folder_path = None
        file_name = str(row["file_name"] or "").strip() or Path(normalized_path).name or asset_id
        folder_id = await _ensure_folder_row_for_migration(
            conn,
            folder_insert=folder_insert,
            existing_by_project=existing_by_project,
            project_id=project_id,
            folder_path=folder_path,
        )
        metadata_json["relative_path"] = f"{folder_path}/{file_name}" if folder_path else file_name
        if not isinstance(metadata_json.get("original_filename"), str) or not str(metadata_json.get("original_filename")).strip():
            metadata_json["original_filename"] = file_name
        updates.append(
            {
                "asset_id": asset_id,
                "folder_id": folder_id,
                "file_name": file_name,
                "source_kind": str(row["source_kind"] or "image"),
                "metadata_json": metadata_json,
            }
        )

    if updates:
        await conn.execute(asset_update, updates)


async def _apply_folders_sequences_migration(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await _ensure_folders_sequences_schema(conn)
        await _backfill_folders_and_assets(conn)
