from __future__ import annotations

import copy
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sheriff_api.config import get_settings

logger = logging.getLogger(__name__)

MIGRATION_TABLE = "schema_migrations"
MULTI_TASK_MIGRATION_VERSION = "multi_task_projects_v1"


@dataclass
class TaskSnapshot:
    id: str
    project_id: str
    name: str
    kind: str
    label_mode: str | None
    created_at: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now_dt() -> datetime:
    # SQL timestamp columns are stored without timezone in this app.
    return datetime.utcnow()


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _is_integer_category_pk(sync_conn) -> bool:
    inspector = inspect(sync_conn)
    if not inspector.has_table("categories"):
        return False
    columns = inspector.get_columns("categories")
    id_column = next((column for column in columns if column.get("name") == "id"), None)
    if not isinstance(id_column, dict):
        return False
    return "int" in str(id_column.get("type", "")).lower()


async def _table_exists(conn: AsyncConnection, table_name: str) -> bool:
    return bool(await conn.run_sync(lambda sync_conn: inspect(sync_conn).has_table(table_name)))


async def _column_exists(conn: AsyncConnection, table_name: str, column_name: str) -> bool:
    def _exists(sync_conn) -> bool:
        inspector = inspect(sync_conn)
        if not inspector.has_table(table_name):
            return False
        return any(column.get("name") == column_name for column in inspector.get_columns(table_name))

    return bool(await conn.run_sync(_exists))


async def _unique_constraints(conn: AsyncConnection, table_name: str) -> list[dict[str, Any]]:
    def _read(sync_conn) -> list[dict[str, Any]]:
        inspector = inspect(sync_conn)
        if not inspector.has_table(table_name):
            return []
        return list(inspector.get_unique_constraints(table_name))

    return await conn.run_sync(_read)


async def _indexes(conn: AsyncConnection, table_name: str) -> list[dict[str, Any]]:
    def _read(sync_conn) -> list[dict[str, Any]]:
        inspector = inspect(sync_conn)
        if not inspector.has_table(table_name):
            return []
        return list(inspector.get_indexes(table_name))

    return await conn.run_sync(_read)


def _normalize_category_id(value: Any, mapping: dict[int, str]) -> tuple[Any, bool]:
    if isinstance(value, int):
        mapped = mapping.get(value, str(value))
        return mapped, True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value, False
        if stripped.isdigit():
            mapped = mapping.get(int(stripped), stripped)
            return mapped, mapped != value
        return stripped, stripped != value
    return value, False


def _normalize_category_ids(values: Any, mapping: dict[int, str]) -> tuple[list[str] | Any, bool]:
    if not isinstance(values, list):
        return values, False
    changed = False
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        next_value, value_changed = _normalize_category_id(value, mapping)
        changed = changed or value_changed
        if not isinstance(next_value, str):
            continue
        if not next_value or next_value in seen:
            continue
        seen.add(next_value)
        normalized.append(next_value)
    if normalized != values:
        changed = True
    return normalized, changed


def rewrite_annotation_payload_ids(payload_json: dict[str, Any], mapping: dict[int, str]) -> tuple[dict[str, Any], bool]:
    payload = copy.deepcopy(payload_json)
    changed = False

    next_category_id, field_changed = _normalize_category_id(payload.get("category_id"), mapping)
    if field_changed:
        payload["category_id"] = next_category_id
        changed = True

    next_category_ids, list_changed = _normalize_category_ids(payload.get("category_ids"), mapping)
    if list_changed:
        payload["category_ids"] = next_category_ids
        changed = True

    classification = payload.get("classification")
    if isinstance(classification, dict):
        next_classification_category_ids, class_list_changed = _normalize_category_ids(
            classification.get("category_ids"), mapping
        )
        if class_list_changed:
            classification["category_ids"] = next_classification_category_ids
            changed = True
        next_primary_id, primary_changed = _normalize_category_id(classification.get("primary_category_id"), mapping)
        if primary_changed:
            classification["primary_category_id"] = next_primary_id
            changed = True

    coco = payload.get("coco")
    if isinstance(coco, dict):
        next_coco_id, coco_changed = _normalize_category_id(coco.get("category_id"), mapping)
        if coco_changed:
            coco["category_id"] = next_coco_id
            changed = True

    objects = payload.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            next_object_category_id, object_changed = _normalize_category_id(item.get("category_id"), mapping)
            if object_changed:
                item["category_id"] = next_object_category_id
                changed = True

    return payload, changed


async def _ensure_migration_table(conn: AsyncConnection) -> None:
    await conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
                version VARCHAR PRIMARY KEY,
                applied_at VARCHAR NOT NULL
            )
            """
        )
    )


async def _load_applied_migrations(conn: AsyncConnection) -> set[str]:
    if not await _table_exists(conn, MIGRATION_TABLE):
        return set()
    rows = (await conn.execute(text(f"SELECT version FROM {MIGRATION_TABLE}"))).all()
    return {str(row[0]) for row in rows if isinstance(row[0], str)}


async def _mark_migration_applied(conn: AsyncConnection, version: str) -> None:
    await conn.execute(
        text(
            f"""
            INSERT INTO {MIGRATION_TABLE} (version, applied_at)
            VALUES (:version, :applied_at)
            """
        ),
        {"version": version, "applied_at": _utc_now_iso()},
    )


def _legacy_task_spec(task_type: str | None) -> tuple[str, str | None]:
    normalized = str(task_type or "classification_single").strip().lower()
    if normalized in {"classification", "classification_single"}:
        return "classification", "single_label"
    if normalized == "bbox":
        return "bbox", None
    if normalized == "segmentation":
        return "segmentation", None
    return "classification", "single_label"


def _task_name_for_kind(kind: str) -> str:
    if kind == "bbox":
        return "bbox"
    if kind == "segmentation":
        return "segmentation"
    return "classification"


def _training_task_for_kind(kind: str) -> str:
    if kind == "bbox":
        return "detection"
    if kind == "segmentation":
        return "segmentation"
    return "classification"


def _next_default_task_name(existing_names: set[str]) -> str:
    if "Default".lower() not in existing_names:
        return "Default"
    suffix = 2
    while True:
        candidate = f"Default ({suffix})"
        if candidate.lower() not in existing_names:
            return candidate
        suffix += 1


def _task_sort_key(row: TaskSnapshot) -> tuple[str, str]:
    return (str(row.created_at or ""), row.id)


def _normalize_task_kind(value: Any, fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"classification", "bbox", "segmentation"}:
        return normalized
    if normalized == "detection":
        return "bbox"
    if normalized == "classification_single":
        return "classification"
    return fallback


def _is_unique_for_columns(row: dict[str, Any], columns: tuple[str, ...]) -> bool:
    listed = tuple(str(col) for col in row.get("column_names") or [])
    return listed == columns


async def _run_legacy_category_uuid_migration(conn: AsyncConnection) -> None:
    if not await _table_exists(conn, "categories"):
        return
    if not await conn.run_sync(_is_integer_category_pk):
        return

    rows = (
        await conn.execute(
            text(
                "SELECT id, project_id, name, display_order, is_active FROM categories "
                "ORDER BY project_id, display_order, id"
            )
        )
    ).mappings().all()
    if not rows:
        return

    legacy_to_uuid = {int(row["id"]): str(uuid.uuid4()) for row in rows}

    await conn.execute(
        text(
            """
            CREATE TABLE categories_v2 (
                id VARCHAR NOT NULL PRIMARY KEY,
                legacy_int_id INTEGER,
                project_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                display_order INTEGER NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
            """
        )
    )
    for row in rows:
        await conn.execute(
            text(
                """
                INSERT INTO categories_v2 (id, legacy_int_id, project_id, name, display_order, is_active)
                VALUES (:id, :legacy_int_id, :project_id, :name, :display_order, :is_active)
                """
            ),
            {
                "id": legacy_to_uuid[int(row["id"])],
                "legacy_int_id": int(row["id"]),
                "project_id": str(row["project_id"]),
                "name": str(row["name"]),
                "display_order": int(row["display_order"] or 0),
                "is_active": bool(row["is_active"]),
            },
        )
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_categories_v2_project_id ON categories_v2 (project_id)"))
    await conn.execute(text("DROP TABLE categories"))
    await conn.execute(text("ALTER TABLE categories_v2 RENAME TO categories"))

    if not await _table_exists(conn, "annotations"):
        return

    annotation_rows = (await conn.execute(text("SELECT id, payload_json FROM annotations"))).mappings().all()
    for row in annotation_rows:
        annotation_id = str(row.get("id") or "")
        payload_raw = row.get("payload_json")
        if isinstance(payload_raw, dict):
            payload_json = payload_raw
        elif isinstance(payload_raw, str):
            try:
                parsed = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            payload_json = parsed
        else:
            continue

        rewritten_payload, changed = rewrite_annotation_payload_ids(payload_json, legacy_to_uuid)
        if not changed:
            continue
        payload_value = json.dumps(rewritten_payload, sort_keys=True)
        if conn.dialect.name == "postgresql":
            await conn.execute(
                text("UPDATE annotations SET payload_json = CAST(:payload_json AS JSON) WHERE id = :id"),
                {"payload_json": payload_value, "id": annotation_id},
            )
        else:
            await conn.execute(
                text("UPDATE annotations SET payload_json = :payload_json WHERE id = :id"),
                {"payload_json": payload_value, "id": annotation_id},
            )


async def _create_tasks_table_if_missing(conn: AsyncConnection) -> None:
    if await _table_exists(conn, "tasks"):
        return
    await conn.execute(
        text(
            """
            CREATE TABLE tasks (
                id VARCHAR NOT NULL PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                kind VARCHAR NOT NULL,
                label_mode VARCHAR NULL,
                name VARCHAR NOT NULL,
                created_at DATETIME NULL,
                updated_at DATETIME NULL,
                CONSTRAINT ck_task_kind_label_mode
                    CHECK ((kind = 'classification' AND label_mode IS NOT NULL) OR (kind != 'classification' AND label_mode IS NULL))
            )
            """
        )
    )
    await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_task_project_name ON tasks (project_id, name)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_project_id ON tasks (project_id)"))


async def _add_column_if_missing(conn: AsyncConnection, table: str, column: str, ddl_fragment: str) -> None:
    if await _column_exists(conn, table, column):
        return
    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl_fragment}"))


async def _rebuild_annotations_table_sqlite(conn: AsyncConnection) -> None:
    await conn.execute(text("PRAGMA foreign_keys=OFF"))
    try:
        await conn.execute(
            text(
                """
                CREATE TABLE annotations_v2 (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    asset_id VARCHAR NOT NULL,
                    project_id VARCHAR NOT NULL,
                    task_id VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    payload_json JSON,
                    annotated_by VARCHAR NULL,
                    created_at DATETIME NULL,
                    updated_at DATETIME NULL,
                    CONSTRAINT uq_annotation_asset_task UNIQUE (asset_id, task_id)
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO annotations_v2 (
                    id, asset_id, project_id, task_id, status, payload_json, annotated_by, created_at, updated_at
                )
                SELECT
                    id,
                    asset_id,
                    project_id,
                    COALESCE(task_id, (SELECT default_task_id FROM projects WHERE projects.id = annotations.project_id)),
                    status,
                    payload_json,
                    annotated_by,
                    created_at,
                    updated_at
                FROM annotations
                """
            )
        )
        await conn.execute(text("DROP TABLE annotations"))
        await conn.execute(text("ALTER TABLE annotations_v2 RENAME TO annotations"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_annotations_asset_id ON annotations (asset_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_annotations_project_id ON annotations (project_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_annotations_task_id ON annotations (task_id)"))
    finally:
        await conn.execute(text("PRAGMA foreign_keys=ON"))


async def _migrate_annotation_uniqueness(conn: AsyncConnection) -> None:
    uniques = await _unique_constraints(conn, "annotations")
    indexes = await _indexes(conn, "annotations")

    has_asset_task_unique = any(
        _is_unique_for_columns(row, ("asset_id", "task_id")) or _is_unique_for_columns(row, ("task_id", "asset_id"))
        for row in uniques
    ) or any(
        bool(row.get("unique"))
        and (
            _is_unique_for_columns(row, ("asset_id", "task_id"))
            or _is_unique_for_columns(row, ("task_id", "asset_id"))
        )
        for row in indexes
    )

    has_asset_only_unique = any(_is_unique_for_columns(row, ("asset_id",)) for row in uniques) or any(
        bool(row.get("unique")) and _is_unique_for_columns(row, ("asset_id",)) for row in indexes
    )

    dialect = conn.dialect.name
    if dialect == "sqlite":
        if has_asset_only_unique:
            await _rebuild_annotations_table_sqlite(conn)
        elif not has_asset_task_unique:
            await conn.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS uq_annotation_asset_task ON annotations (asset_id, task_id)")
            )
        return

    for row in uniques:
        if not _is_unique_for_columns(row, ("asset_id",)):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            continue
        await conn.execute(text(f"ALTER TABLE annotations DROP CONSTRAINT IF EXISTS {_quote_ident(name)}"))

    uniques = await _unique_constraints(conn, "annotations")
    has_asset_task_unique = any(
        _is_unique_for_columns(row, ("asset_id", "task_id")) or _is_unique_for_columns(row, ("task_id", "asset_id"))
        for row in uniques
    )
    if not has_asset_task_unique:
        await conn.execute(
            text("ALTER TABLE annotations ADD CONSTRAINT uq_annotation_asset_task UNIQUE (asset_id, task_id)")
        )


async def _load_task_snapshots(conn: AsyncConnection) -> tuple[dict[str, str], dict[str, dict[str, TaskSnapshot]]]:
    project_rows = (await conn.execute(text("SELECT id, default_task_id FROM projects"))).mappings().all()
    task_rows = (
        await conn.execute(text("SELECT id, project_id, name, kind, label_mode, created_at FROM tasks ORDER BY created_at, id"))
    ).mappings().all()

    default_task_by_project: dict[str, str] = {}
    for row in project_rows:
        project_id = str(row.get("id") or "")
        default_task_id = row.get("default_task_id")
        if project_id and isinstance(default_task_id, str) and default_task_id:
            default_task_by_project[project_id] = default_task_id

    tasks_by_project: dict[str, dict[str, TaskSnapshot]] = {}
    for row in task_rows:
        project_id = str(row.get("project_id") or "")
        task_id = str(row.get("id") or "")
        if not project_id or not task_id:
            continue
        snapshot = TaskSnapshot(
            id=task_id,
            project_id=project_id,
            name=str(row.get("name") or ""),
            kind=_normalize_task_kind(row.get("kind"), "classification"),
            label_mode=str(row.get("label_mode")) if isinstance(row.get("label_mode"), str) else None,
            created_at=str(row.get("created_at")) if row.get("created_at") is not None else None,
        )
        tasks_by_project.setdefault(project_id, {})[task_id] = snapshot
    return default_task_by_project, tasks_by_project


async def _run_multi_task_projects_db_migration(conn: AsyncConnection) -> tuple[dict[str, str], dict[str, dict[str, TaskSnapshot]]]:
    await _create_tasks_table_if_missing(conn)
    await _add_column_if_missing(conn, "projects", "default_task_id", "default_task_id VARCHAR")
    await _add_column_if_missing(conn, "categories", "task_id", "task_id VARCHAR")
    await _add_column_if_missing(conn, "annotations", "task_id", "task_id VARCHAR")

    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_project_id ON tasks (project_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_projects_default_task_id ON projects (default_task_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_categories_task_id ON categories (task_id)"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_annotations_task_id ON annotations (task_id)"))

    project_rows = (await conn.execute(text("SELECT id, task_type, default_task_id FROM projects ORDER BY created_at, id"))).mappings().all()
    task_rows = (
        await conn.execute(text("SELECT id, project_id, name, kind, label_mode, created_at FROM tasks ORDER BY created_at, id"))
    ).mappings().all()

    tasks_by_project: dict[str, list[TaskSnapshot]] = {}
    for row in task_rows:
        project_id = str(row.get("project_id") or "")
        task_id = str(row.get("id") or "")
        if not project_id or not task_id:
            continue
        tasks_by_project.setdefault(project_id, []).append(
            TaskSnapshot(
                id=task_id,
                project_id=project_id,
                name=str(row.get("name") or ""),
                kind=_normalize_task_kind(row.get("kind"), "classification"),
                label_mode=str(row.get("label_mode")) if isinstance(row.get("label_mode"), str) else None,
                created_at=str(row.get("created_at")) if row.get("created_at") is not None else None,
            )
        )

    for project_tasks in tasks_by_project.values():
        for task in project_tasks:
            if task.kind == "classification" and not task.label_mode:
                await conn.execute(
                    text("UPDATE tasks SET label_mode = :label_mode, updated_at = :updated_at WHERE id = :task_id"),
                    {"label_mode": "single_label", "updated_at": _utc_now_dt(), "task_id": task.id},
                )
                task.label_mode = "single_label"
            if task.kind != "classification" and task.label_mode is not None:
                await conn.execute(
                    text("UPDATE tasks SET label_mode = NULL, updated_at = :updated_at WHERE id = :task_id"),
                    {"updated_at": _utc_now_dt(), "task_id": task.id},
                )
                task.label_mode = None

    default_task_by_project: dict[str, str] = {}
    for row in project_rows:
        project_id = str(row.get("id") or "")
        if not project_id:
            continue

        existing = tasks_by_project.get(project_id, [])
        existing_by_id = {task.id: task for task in existing}
        existing_names = {task.name.strip().lower() for task in existing if task.name.strip()}

        requested_default = str(row.get("default_task_id") or "")
        selected_task: TaskSnapshot | None = existing_by_id.get(requested_default)
        if selected_task is None and existing:
            selected_task = sorted(existing, key=_task_sort_key)[0]

        if selected_task is None:
            kind, label_mode = _legacy_task_spec(str(row.get("task_type") or "classification_single"))
            task_name = _next_default_task_name(existing_names)
            task_id = str(uuid.uuid4())
            now_dt = _utc_now_dt()
            now_iso = now_dt.isoformat().replace("+00:00", "Z")
            await conn.execute(
                text(
                    """
                    INSERT INTO tasks (id, project_id, kind, label_mode, name, created_at, updated_at)
                    VALUES (:id, :project_id, :kind, :label_mode, :name, :created_at, :updated_at)
                    """
                ),
                {
                    "id": task_id,
                    "project_id": project_id,
                    "kind": kind,
                    "label_mode": label_mode,
                    "name": task_name,
                    "created_at": now_dt,
                    "updated_at": now_dt,
                },
            )
            selected_task = TaskSnapshot(
                id=task_id,
                project_id=project_id,
                name=task_name,
                kind=kind,
                label_mode=label_mode,
                created_at=now_iso,
            )
            tasks_by_project.setdefault(project_id, []).append(selected_task)
            task_rows.append(
                {
                    "id": task_id,
                    "project_id": project_id,
                    "name": task_name,
                    "kind": kind,
                    "label_mode": label_mode,
                    "created_at": now_iso,
                }
            )

        default_task_by_project[project_id] = selected_task.id
        if requested_default != selected_task.id:
            await conn.execute(
                text("UPDATE projects SET default_task_id = :default_task_id WHERE id = :project_id"),
                {"default_task_id": selected_task.id, "project_id": project_id},
            )

    await conn.execute(
        text(
            """
            UPDATE categories
            SET task_id = (
                SELECT projects.default_task_id
                FROM projects
                WHERE projects.id = categories.project_id
            )
            WHERE task_id IS NULL OR task_id = ''
            """
        )
    )
    await conn.execute(
        text(
            """
            UPDATE annotations
            SET task_id = (
                SELECT projects.default_task_id
                FROM projects
                WHERE projects.id = annotations.project_id
            )
            WHERE task_id IS NULL OR task_id = ''
            """
        )
    )

    if conn.dialect.name == "postgresql":
        # Keep nullable to allow project creation before default task linkage in one transaction.
        await conn.execute(text("ALTER TABLE projects ALTER COLUMN default_task_id DROP NOT NULL"))
        null_category_task_count = int(
            (await conn.execute(text("SELECT COUNT(*) FROM categories WHERE task_id IS NULL OR task_id = ''"))).scalar_one()
        )
        if null_category_task_count == 0:
            await conn.execute(text("ALTER TABLE categories ALTER COLUMN task_id SET NOT NULL"))
        null_annotation_task_count = int(
            (await conn.execute(text("SELECT COUNT(*) FROM annotations WHERE task_id IS NULL OR task_id = ''"))).scalar_one()
        )
        if null_annotation_task_count == 0:
            await conn.execute(text("ALTER TABLE annotations ALTER COLUMN task_id SET NOT NULL"))

    await _migrate_annotation_uniqueness(conn)
    return await _load_task_snapshots(conn)


def _ensure_backup(path: Path) -> None:
    backup_path = path.with_name(f"{path.name}.bak")
    if backup_path.exists():
        return
    backup_path.write_bytes(path.read_bytes())


def _write_json_atomic(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(path)


def _read_json_or_warn(path: Path, *, expected: str) -> Any | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping %s migration for %s due to parse/read error: %s", expected, path, exc)
        return None
    return loaded


def _resolve_task(
    candidate_task_id: str | None,
    *,
    default_task_id: str,
    tasks_by_id: dict[str, TaskSnapshot],
) -> TaskSnapshot:
    if isinstance(candidate_task_id, str):
        stripped = candidate_task_id.strip()
        if stripped and stripped in tasks_by_id:
            return tasks_by_id[stripped]
    if default_task_id in tasks_by_id:
        return tasks_by_id[default_task_id]
    return next(iter(tasks_by_id.values()))


def _migrate_datasets_file(
    path: Path,
    *,
    default_task_id: str,
    tasks_by_id: dict[str, TaskSnapshot],
) -> dict[str, str]:
    dataset_task_by_version: dict[str, str] = {}
    if not path.exists():
        return dataset_task_by_version

    loaded = _read_json_or_warn(path, expected="datasets")
    if not isinstance(loaded, dict):
        logger.warning("Skipping datasets migration for %s due to unexpected root type", path)
        return dataset_task_by_version
    items = loaded.get("items")
    if not isinstance(items, list):
        logger.warning("Skipping datasets migration for %s due to missing items list", path)
        return dataset_task_by_version

    changed = False
    for row in items:
        if not isinstance(row, dict):
            continue
        task = _resolve_task(
            str(row.get("task_id")) if isinstance(row.get("task_id"), str) else None,
            default_task_id=default_task_id,
            tasks_by_id=tasks_by_id,
        )
        if row.get("task_id") != task.id:
            row["task_id"] = task.id
            changed = True

        derived_task = _task_name_for_kind(task.kind)
        if row.get("task") != derived_task:
            row["task"] = derived_task
            changed = True

        if task.kind == "classification":
            labels = row.get("labels")
            if not isinstance(labels, dict):
                labels = {}
                row["labels"] = labels
                changed = True
            desired_label_mode = task.label_mode or "single_label"
            if labels.get("label_mode") != desired_label_mode:
                labels["label_mode"] = desired_label_mode
                changed = True

        dataset_version_id = row.get("dataset_version_id")
        if isinstance(dataset_version_id, str) and dataset_version_id:
            dataset_task_by_version[dataset_version_id] = task.id

    if not changed:
        return dataset_task_by_version

    try:
        _ensure_backup(path)
        _write_json_atomic(path, loaded)
    except OSError as exc:
        logger.warning("Failed writing migrated datasets file %s: %s", path, exc)
    return dataset_task_by_version


def _migrate_models_file(
    path: Path,
    *,
    default_task_id: str,
    tasks_by_id: dict[str, TaskSnapshot],
    dataset_task_by_version: dict[str, str],
) -> dict[str, str]:
    model_task_by_id: dict[str, str] = {}
    if not path.exists():
        return model_task_by_id

    loaded = _read_json_or_warn(path, expected="models")
    if not isinstance(loaded, list):
        logger.warning("Skipping models migration for %s due to unexpected root type", path)
        return model_task_by_id

    changed = False
    for row in loaded:
        if not isinstance(row, dict):
            continue
        config_json = row.get("config_json")
        if not isinstance(config_json, dict):
            config_json = {}
            row["config_json"] = config_json
            changed = True
        source_dataset = config_json.get("source_dataset")
        if not isinstance(source_dataset, dict):
            source_dataset = {}
            config_json["source_dataset"] = source_dataset
            changed = True

        source_manifest_id = source_dataset.get("manifest_id")
        inferred_task_id = dataset_task_by_version.get(source_manifest_id) if isinstance(source_manifest_id, str) else None
        task = _resolve_task(
            str(row.get("task_id")) if isinstance(row.get("task_id"), str) else inferred_task_id,
            default_task_id=default_task_id,
            tasks_by_id=tasks_by_id,
        )
        source_task_id = source_dataset.get("task_id")
        if not isinstance(source_task_id, str):
            source_task_id = None
        if source_task_id and source_task_id in tasks_by_id:
            task = tasks_by_id[source_task_id]

        if row.get("task_id") != task.id:
            row["task_id"] = task.id
            changed = True
        if source_dataset.get("task_id") != task.id:
            source_dataset["task_id"] = task.id
            changed = True

        desired_source_task = _training_task_for_kind(task.kind)
        if source_dataset.get("task") != desired_source_task:
            source_dataset["task"] = desired_source_task
            changed = True
        if task.kind == "classification":
            desired_label_mode = task.label_mode or "single_label"
            if source_dataset.get("label_mode") != desired_label_mode:
                source_dataset["label_mode"] = desired_label_mode
                changed = True
        elif "label_mode" in source_dataset:
            source_dataset.pop("label_mode", None)
            changed = True

        model_id = row.get("id")
        if isinstance(model_id, str) and model_id:
            model_task_by_id[model_id] = task.id

    if not changed:
        return model_task_by_id

    try:
        _ensure_backup(path)
        _write_json_atomic(path, loaded)
    except OSError as exc:
        logger.warning("Failed writing migrated models file %s: %s", path, exc)
    return model_task_by_id


def _migrate_experiment_config_file(
    path: Path,
    *,
    default_task_id: str,
    tasks_by_id: dict[str, TaskSnapshot],
    fallback_task_id: str | None,
) -> None:
    if not path.exists():
        return
    loaded = _read_json_or_warn(path, expected="experiment config")
    if not isinstance(loaded, dict):
        logger.warning("Skipping experiment config migration for %s due to unexpected root type", path)
        return

    task = _resolve_task(
        str(loaded.get("task_id")) if isinstance(loaded.get("task_id"), str) else fallback_task_id,
        default_task_id=default_task_id,
        tasks_by_id=tasks_by_id,
    )
    changed = False
    if loaded.get("task_id") != task.id:
        loaded["task_id"] = task.id
        changed = True
    desired_task = _training_task_for_kind(task.kind)
    if loaded.get("task") != desired_task:
        loaded["task"] = desired_task
        changed = True

    if not changed:
        return
    try:
        _ensure_backup(path)
        _write_json_atomic(path, loaded)
    except OSError as exc:
        logger.warning("Failed writing migrated experiment config file %s: %s", path, exc)


def _migrate_experiments_file(
    records_path: Path,
    *,
    experiments_dir: Path,
    default_task_id: str,
    tasks_by_id: dict[str, TaskSnapshot],
    dataset_task_by_version: dict[str, str],
    model_task_by_id: dict[str, str],
) -> dict[str, str]:
    experiment_task_by_id: dict[str, str] = {}
    if not records_path.exists():
        return experiment_task_by_id

    loaded = _read_json_or_warn(records_path, expected="experiments")
    if not isinstance(loaded, list):
        logger.warning("Skipping experiments migration for %s due to unexpected root type", records_path)
        return experiment_task_by_id

    changed = False
    for row in loaded:
        if not isinstance(row, dict):
            continue
        config_json = row.get("config_json")
        if not isinstance(config_json, dict):
            config_json = {}
            row["config_json"] = config_json
            changed = True

        dataset_version_id = config_json.get("dataset_version_id")
        model_id = row.get("model_id")
        inferred_task_id: str | None = None
        if isinstance(dataset_version_id, str):
            inferred_task_id = dataset_task_by_version.get(dataset_version_id)
        if not inferred_task_id and isinstance(model_id, str):
            inferred_task_id = model_task_by_id.get(model_id)

        task = _resolve_task(
            str(row.get("task_id")) if isinstance(row.get("task_id"), str) else inferred_task_id,
            default_task_id=default_task_id,
            tasks_by_id=tasks_by_id,
        )
        config_task_id = config_json.get("task_id")
        if isinstance(config_task_id, str) and config_task_id in tasks_by_id:
            task = tasks_by_id[config_task_id]

        if row.get("task_id") != task.id:
            row["task_id"] = task.id
            changed = True
        if config_json.get("task_id") != task.id:
            config_json["task_id"] = task.id
            changed = True

        desired_task = _training_task_for_kind(task.kind)
        if config_json.get("task") != desired_task:
            config_json["task"] = desired_task
            changed = True

        experiment_id = row.get("id")
        if isinstance(experiment_id, str) and experiment_id:
            experiment_task_by_id[experiment_id] = task.id

    if changed:
        try:
            _ensure_backup(records_path)
            _write_json_atomic(records_path, loaded)
        except OSError as exc:
            logger.warning("Failed writing migrated experiments file %s: %s", records_path, exc)

    for experiment_id, task_id in experiment_task_by_id.items():
        config_path = experiments_dir / experiment_id / "config.json"
        _migrate_experiment_config_file(
            config_path,
            default_task_id=default_task_id,
            tasks_by_id=tasks_by_id,
            fallback_task_id=task_id,
        )

    return experiment_task_by_id


def _migrate_deployments_file(
    path: Path,
    *,
    default_task_id: str,
    tasks_by_id: dict[str, TaskSnapshot],
    experiment_task_by_id: dict[str, str],
) -> None:
    if not path.exists():
        return
    loaded = _read_json_or_warn(path, expected="deployments")
    if not isinstance(loaded, dict):
        logger.warning("Skipping deployments migration for %s due to unexpected root type", path)
        return
    items = loaded.get("items")
    if not isinstance(items, list):
        logger.warning("Skipping deployments migration for %s due to missing items list", path)
        return

    changed = False
    for row in items:
        if not isinstance(row, dict):
            continue
        source = row.get("source")
        inferred_task_id: str | None = None
        if isinstance(source, dict):
            experiment_id = source.get("experiment_id")
            if isinstance(experiment_id, str):
                inferred_task_id = experiment_task_by_id.get(experiment_id)

        task = _resolve_task(
            str(row.get("task_id")) if isinstance(row.get("task_id"), str) else inferred_task_id,
            default_task_id=default_task_id,
            tasks_by_id=tasks_by_id,
        )
        if row.get("task_id") != task.id:
            row["task_id"] = task.id
            changed = True

        desired_task_name = _task_name_for_kind(task.kind)
        if row.get("task") != desired_task_name:
            row["task"] = desired_task_name
            changed = True

    if not changed:
        return
    try:
        _ensure_backup(path)
        _write_json_atomic(path, loaded)
    except OSError as exc:
        logger.warning("Failed writing migrated deployments file %s: %s", path, exc)


def _run_file_store_migrations(
    *,
    storage_root: Path,
    default_task_by_project: dict[str, str],
    tasks_by_project: dict[str, dict[str, TaskSnapshot]],
) -> None:
    for project_id, default_task_id in default_task_by_project.items():
        project_tasks = tasks_by_project.get(project_id) or {}
        if not project_tasks:
            logger.warning("Skipping file-store migration for project %s due to missing task snapshots", project_id)
            continue
        if default_task_id not in project_tasks:
            default_task_id = sorted(project_tasks.values(), key=_task_sort_key)[0].id

        datasets_path = storage_root / "datasets" / project_id / "datasets.json"
        dataset_task_by_version = _migrate_datasets_file(
            datasets_path,
            default_task_id=default_task_id,
            tasks_by_id=project_tasks,
        )

        models_path = storage_root / "models" / project_id / "records.json"
        model_task_by_id = _migrate_models_file(
            models_path,
            default_task_id=default_task_id,
            tasks_by_id=project_tasks,
            dataset_task_by_version=dataset_task_by_version,
        )

        experiments_project_dir = storage_root / "experiments" / project_id
        experiments_records_path = experiments_project_dir / "records.json"
        experiment_task_by_id = _migrate_experiments_file(
            experiments_records_path,
            experiments_dir=experiments_project_dir,
            default_task_id=default_task_id,
            tasks_by_id=project_tasks,
            dataset_task_by_version=dataset_task_by_version,
            model_task_by_id=model_task_by_id,
        )

        deployments_path = storage_root / "deployments" / project_id / "deployments.json"
        _migrate_deployments_file(
            deployments_path,
            default_task_id=default_task_id,
            tasks_by_id=project_tasks,
            experiment_task_by_id=experiment_task_by_id,
        )


async def _apply_multi_task_projects_migration(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await _run_legacy_category_uuid_migration(conn)
        default_task_by_project, tasks_by_project = await _run_multi_task_projects_db_migration(conn)

    try:
        settings = get_settings()
        storage_root = Path(settings.storage_root)
        _run_file_store_migrations(
            storage_root=storage_root,
            default_task_by_project=default_task_by_project,
            tasks_by_project=tasks_by_project,
        )
    except Exception as exc:
        logger.warning("File-store migration step failed with best-effort semantics: %s", exc)


async def run_startup_migrations(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await _ensure_migration_table(conn)
        applied_versions = await _load_applied_migrations(conn)

    if MULTI_TASK_MIGRATION_VERSION not in applied_versions:
        await _apply_multi_task_projects_migration(engine)
        async with engine.begin() as conn:
            await _mark_migration_applied(conn, MULTI_TASK_MIGRATION_VERSION)
