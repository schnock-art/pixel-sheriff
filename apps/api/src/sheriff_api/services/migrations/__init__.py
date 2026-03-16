from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from sheriff_api.services.migrations.folders_sequences_v1 import _apply_folders_sequences_migration
from sheriff_api.services.migrations.multi_task_projects_v1 import _apply_multi_task_projects_migration
from sheriff_api.services.migrations.prelabels_v2 import _apply_prelabels_migration
from sheriff_api.services.migrations.shared import (
    FOLDERS_SEQUENCES_MIGRATION_VERSION,
    MIGRATION_TABLE,
    MULTI_TASK_MIGRATION_VERSION,
    PRELABELS_MIGRATION_VERSION,
    _ensure_migration_table,
    _load_applied_migrations,
    _mark_migration_applied,
)


async def run_startup_migrations(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await _ensure_migration_table(conn)
        applied_versions = await _load_applied_migrations(conn)

    migration_steps = (
        (MULTI_TASK_MIGRATION_VERSION, _apply_multi_task_projects_migration),
        (FOLDERS_SEQUENCES_MIGRATION_VERSION, _apply_folders_sequences_migration),
        (PRELABELS_MIGRATION_VERSION, _apply_prelabels_migration),
    )
    for version, apply_migration in migration_steps:
        if version in applied_versions:
            continue
        await apply_migration(engine)
        async with engine.begin() as conn:
            await _mark_migration_applied(conn, version)
        applied_versions.add(version)


__all__ = [
    "FOLDERS_SEQUENCES_MIGRATION_VERSION",
    "MIGRATION_TABLE",
    "MULTI_TASK_MIGRATION_VERSION",
    "PRELABELS_MIGRATION_VERSION",
    "_apply_folders_sequences_migration",
    "_apply_multi_task_projects_migration",
    "_apply_prelabels_migration",
    "run_startup_migrations",
]
