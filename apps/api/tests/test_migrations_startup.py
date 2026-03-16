from __future__ import annotations

import pytest
from sqlalchemy import text

from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
import sheriff_api.services.migrations as migrations


async def _reset_migration_state() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text(f"DROP TABLE IF EXISTS {migrations.MIGRATION_TABLE}"))


@pytest.mark.asyncio
async def test_run_startup_migrations_applies_versions_in_order_and_records_them(monkeypatch: pytest.MonkeyPatch) -> None:
    await _reset_migration_state()

    applied: list[str] = []

    async def fake_multi_task(current_engine) -> None:
        assert current_engine is engine
        applied.append(migrations.MULTI_TASK_MIGRATION_VERSION)

    async def fake_folders_sequences(current_engine) -> None:
        assert current_engine is engine
        applied.append(migrations.FOLDERS_SEQUENCES_MIGRATION_VERSION)

    async def fake_prelabels(current_engine) -> None:
        assert current_engine is engine
        applied.append(migrations.PRELABELS_MIGRATION_VERSION)

    monkeypatch.setattr(migrations, "_apply_multi_task_projects_migration", fake_multi_task)
    monkeypatch.setattr(migrations, "_apply_folders_sequences_migration", fake_folders_sequences)
    monkeypatch.setattr(migrations, "_apply_prelabels_migration", fake_prelabels)

    await migrations.run_startup_migrations(engine)

    assert applied == [
        migrations.MULTI_TASK_MIGRATION_VERSION,
        migrations.FOLDERS_SEQUENCES_MIGRATION_VERSION,
        migrations.PRELABELS_MIGRATION_VERSION,
    ]

    async with engine.begin() as conn:
        recorded_versions = (
            await conn.execute(text(f"SELECT version FROM {migrations.MIGRATION_TABLE} ORDER BY version"))
        ).scalars().all()

    assert recorded_versions == sorted(applied)


@pytest.mark.asyncio
async def test_run_startup_migrations_skips_versions_that_are_already_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    await _reset_migration_state()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"""
                CREATE TABLE {migrations.MIGRATION_TABLE} (
                    version VARCHAR PRIMARY KEY,
                    applied_at VARCHAR NOT NULL
                )
                """
            )
        )
        await conn.execute(
            text(
                f"""
                INSERT INTO {migrations.MIGRATION_TABLE} (version, applied_at)
                VALUES (:version, :applied_at)
                """
            ),
            {"version": migrations.MULTI_TASK_MIGRATION_VERSION, "applied_at": "2026-03-16T00:00:00Z"},
        )

    applied: list[str] = []

    async def fake_apply(current_engine) -> None:
        assert current_engine is engine
        applied.append("called")

    monkeypatch.setattr(migrations, "_apply_multi_task_projects_migration", fake_apply)

    async def fake_folders_sequences(current_engine) -> None:
        assert current_engine is engine
        applied.append(migrations.FOLDERS_SEQUENCES_MIGRATION_VERSION)

    async def fake_prelabels(current_engine) -> None:
        assert current_engine is engine
        applied.append(migrations.PRELABELS_MIGRATION_VERSION)

    monkeypatch.setattr(migrations, "_apply_folders_sequences_migration", fake_folders_sequences)
    monkeypatch.setattr(migrations, "_apply_prelabels_migration", fake_prelabels)

    await migrations.run_startup_migrations(engine)

    assert applied == [
        migrations.FOLDERS_SEQUENCES_MIGRATION_VERSION,
        migrations.PRELABELS_MIGRATION_VERSION,
    ]

    async with engine.begin() as conn:
        recorded_versions = (
            await conn.execute(text(f"SELECT version FROM {migrations.MIGRATION_TABLE} ORDER BY version"))
        ).scalars().all()

    assert recorded_versions == sorted(
        [
            migrations.MULTI_TASK_MIGRATION_VERSION,
            migrations.FOLDERS_SEQUENCES_MIGRATION_VERSION,
            migrations.PRELABELS_MIGRATION_VERSION,
        ]
    )
