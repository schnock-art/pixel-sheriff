from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

MIGRATION_TABLE = "schema_migrations"
MULTI_TASK_MIGRATION_VERSION = "multi_task_projects_v1"
FOLDERS_SEQUENCES_MIGRATION_VERSION = "folders_sequences_v1"
PRELABELS_MIGRATION_VERSION = "prelabels_v2"


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
    if "default" not in existing_names:
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


async def _add_column_if_missing(conn: AsyncConnection, table: str, column: str, ddl_fragment: str) -> None:
    if await _column_exists(conn, table, column):
        return
    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl_fragment}"))


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


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def _legacy_asset_relative_path_for_migration(metadata_json: dict[str, Any], uri: str, asset_id: str) -> str:
    relative_path = metadata_json.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        return relative_path.replace("\\", "/").strip("/")
    original_filename = metadata_json.get("original_filename")
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.replace("\\", "/").strip("/")
    uri_path = str(uri or "").replace("\\", "/").strip()
    if uri_path:
        return Path(uri_path).name
    return asset_id
