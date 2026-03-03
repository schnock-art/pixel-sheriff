from __future__ import annotations

import copy
import uuid
from typing import Any

from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from sheriff_api.db.models import Annotation


def _is_integer_category_pk(sync_conn) -> bool:
    inspector = inspect(sync_conn)
    if not inspector.has_table("categories"):
        return False
    columns = inspector.get_columns("categories")
    id_column = next((column for column in columns if column.get("name") == "id"), None)
    if not isinstance(id_column, dict):
        return False
    return "int" in str(id_column.get("type", "")).lower()


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


async def run_startup_migrations(engine: AsyncEngine) -> None:
    legacy_to_uuid: dict[int, str] = {}

    async with engine.begin() as conn:
        if await conn.run_sync(_is_integer_category_pk):
            rows = (
                await conn.execute(
                    text(
                        "SELECT id, project_id, name, display_order, is_active FROM categories "
                        "ORDER BY project_id, display_order, id"
                    )
                )
            ).mappings().all()
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
                        "project_id": row["project_id"],
                        "name": row["name"],
                        "display_order": int(row["display_order"] or 0),
                        "is_active": bool(row["is_active"]),
                    },
                )
            await conn.execute(text("CREATE INDEX ix_categories_v2_project_id ON categories_v2 (project_id)"))
            await conn.execute(text("DROP TABLE categories"))
            await conn.execute(text("ALTER TABLE categories_v2 RENAME TO categories"))
        else:
            rows = (
                await conn.execute(
                    text(
                        "SELECT legacy_int_id, id FROM categories "
                        "WHERE legacy_int_id IS NOT NULL ORDER BY legacy_int_id"
                    )
                )
            ).mappings().all()
            legacy_to_uuid = {
                int(row["legacy_int_id"]): str(row["id"])
                for row in rows
                if row.get("legacy_int_id") is not None and isinstance(row.get("id"), str)
            }

    if not legacy_to_uuid:
        return

    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_maker() as session:
        annotations = list((await session.execute(select(Annotation))).scalars().all())
        changed_count = 0
        for annotation in annotations:
            payload_json = annotation.payload_json if isinstance(annotation.payload_json, dict) else {}
            rewritten_payload, changed = rewrite_annotation_payload_ids(payload_json, legacy_to_uuid)
            if not changed:
                continue
            annotation.payload_json = rewritten_payload
            changed_count += 1
        if changed_count > 0:
            await session.commit()
