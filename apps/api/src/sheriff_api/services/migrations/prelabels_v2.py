from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sheriff_api.services.migrations.shared import _add_column_if_missing


async def _ensure_prelabels_schema(conn: AsyncConnection) -> None:
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS prelabel_sessions (
                id VARCHAR NOT NULL PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                task_id VARCHAR NOT NULL,
                sequence_id VARCHAR NOT NULL,
                source_type VARCHAR NOT NULL,
                source_ref VARCHAR NULL,
                prompts_json JSON,
                sampling_mode VARCHAR NOT NULL,
                sampling_value FLOAT NOT NULL,
                confidence_threshold FLOAT NOT NULL,
                max_detections_per_frame INTEGER NOT NULL,
                live_mode BOOLEAN NOT NULL DEFAULT 0,
                status VARCHAR NOT NULL,
                input_closed_at DATETIME NULL,
                enqueued_assets INTEGER NOT NULL DEFAULT 0,
                processed_assets INTEGER NOT NULL DEFAULT 0,
                generated_proposals INTEGER NOT NULL DEFAULT 0,
                skipped_unmatched INTEGER NOT NULL DEFAULT 0,
                error_message VARCHAR NULL,
                debug_detections_json JSON,
                created_at DATETIME NULL,
                updated_at DATETIME NULL
            )
            """
        )
    )
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_sessions_project_id ON prelabel_sessions (project_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_sessions_task_id ON prelabel_sessions (task_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_sessions_sequence_id ON prelabel_sessions (sequence_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_sessions_status ON prelabel_sessions (status)"))
    await _add_column_if_missing(conn, "prelabel_sessions", "debug_detections_json", "debug_detections_json JSON")

    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS prelabel_proposals (
                id VARCHAR NOT NULL PRIMARY KEY,
                session_id VARCHAR NOT NULL,
                asset_id VARCHAR NOT NULL,
                project_id VARCHAR NOT NULL,
                task_id VARCHAR NOT NULL,
                category_id VARCHAR NOT NULL,
                label_text VARCHAR NOT NULL,
                prompt_text VARCHAR NULL,
                confidence FLOAT NOT NULL,
                bbox_json JSON,
                status VARCHAR NOT NULL,
                reviewed_bbox_json JSON NULL,
                reviewed_category_id VARCHAR NULL,
                promoted_annotation_id VARCHAR NULL,
                promoted_object_id VARCHAR NULL,
                created_at DATETIME NULL,
                updated_at DATETIME NULL
            )
            """
        )
    )
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_proposals_session_id ON prelabel_proposals (session_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_proposals_asset_id ON prelabel_proposals (asset_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_proposals_project_id ON prelabel_proposals (project_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_proposals_task_id ON prelabel_proposals (task_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_proposals_category_id ON prelabel_proposals (category_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_prelabel_proposals_status ON prelabel_proposals (status)"))


async def _apply_prelabels_migration(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await _ensure_prelabels_schema(conn)
