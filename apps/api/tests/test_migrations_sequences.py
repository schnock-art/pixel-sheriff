from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.services.migrations import _apply_folders_sequences_migration


@pytest.mark.asyncio
async def test_folders_sequences_migration_backfills_folder_and_file_fields() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(
            text(
                """
                CREATE TABLE assets (
                    id VARCHAR PRIMARY KEY,
                    project_id VARCHAR NOT NULL,
                    type VARCHAR NOT NULL,
                    uri VARCHAR NOT NULL,
                    mime_type VARCHAR NOT NULL,
                    width INTEGER NULL,
                    height INTEGER NULL,
                    checksum VARCHAR NOT NULL,
                    metadata_json TEXT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO assets (id, project_id, type, uri, mime_type, width, height, checksum, metadata_json)
                VALUES (:id, :project_id, :type, :uri, :mime_type, :width, :height, :checksum, :metadata_json)
                """
            ),
            {
                "id": "asset-1",
                "project_id": "project-1",
                "type": "image",
                "uri": "/api/v1/assets/asset-1/content",
                "mime_type": "image/jpeg",
                "width": 640,
                "height": 480,
                "checksum": "a" * 64,
                "metadata_json": json.dumps({"relative_path": "legacy/train/cat.jpg", "original_filename": "cat.jpg"}),
            },
        )

    await _apply_folders_sequences_migration(engine)

    async with engine.begin() as conn:
        folders = (
            await conn.execute(text("SELECT path FROM folders WHERE project_id = :project_id ORDER BY path"), {"project_id": "project-1"})
        ).scalars().all()
        asset_row = (
            await conn.execute(
                text(
                    """
                    SELECT folder_id, file_name, source_kind, metadata_json
                    FROM assets
                    WHERE id = :asset_id
                    """
                ),
                {"asset_id": "asset-1"},
            )
        ).mappings().one()

    assert folders == ["legacy", "legacy/train"]
    assert asset_row["folder_id"]
    assert asset_row["file_name"] == "cat.jpg"
    assert asset_row["source_kind"] == "image"
    metadata = json.loads(asset_row["metadata_json"])
    assert metadata["relative_path"] == "legacy/train/cat.jpg"
    assert metadata["original_filename"] == "cat.jpg"
