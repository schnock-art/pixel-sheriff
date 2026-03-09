from __future__ import annotations

from pathlib import PurePosixPath

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.db.models import Folder


def sanitize_file_name(value: str | None, fallback: str = "upload") -> str:
    raw = str(value or "").replace("\\", "/").strip()
    name = PurePosixPath(raw).name.strip()
    if not name:
        return fallback
    if name in {".", ".."}:
        return fallback
    return name


def sanitize_folder_name(value: str | None, fallback: str = "folder") -> str:
    raw = str(value or "").replace("\\", "/").strip().strip(".")
    if not raw:
        return fallback
    pieces = [segment.strip() for segment in raw.split("/") if segment.strip() and segment.strip() not in {".", ".."}]
    if not pieces:
        return fallback
    return pieces[-1]


def normalize_relative_path(relative_path: str | None, fallback_filename: str | None = None) -> str:
    fallback = sanitize_file_name(fallback_filename)
    raw = str(relative_path or fallback).replace("\\", "/").strip().strip("/")
    if not raw:
        return fallback

    parts = [segment.strip() for segment in raw.split("/") if segment.strip()]
    if not parts:
        return fallback
    if any(segment in {".", ".."} for segment in parts):
        raise ValueError("Relative path cannot contain '.' or '..' segments")

    normalized = parts[:-1] + [sanitize_file_name(parts[-1], fallback=fallback)]
    return "/".join(part for part in normalized if part)


def split_relative_path(relative_path: str | None, fallback_filename: str | None = None) -> tuple[str | None, str]:
    normalized = normalize_relative_path(relative_path, fallback_filename)
    path = PurePosixPath(normalized)
    folder_path = str(path.parent).replace("\\", "/").strip("/")
    if folder_path in {"", "."}:
        folder_path = None
    return folder_path, path.name


def join_folder_and_file(folder_path: str | None, file_name: str) -> str:
    safe_file_name = sanitize_file_name(file_name)
    normalized_folder = str(folder_path or "").replace("\\", "/").strip("/")
    if normalized_folder:
        return f"{normalized_folder}/{safe_file_name}"
    return safe_file_name


async def list_project_folders(db: AsyncSession, project_id: str) -> list[Folder]:
    result = await db.execute(select(Folder).where(Folder.project_id == project_id))
    folders = list(result.scalars().all())
    folders.sort(key=lambda folder: (folder.path.count("/"), folder.path))
    return folders


async def ensure_folder_path(db: AsyncSession, project_id: str, folder_path: str | None) -> Folder | None:
    normalized_path = str(folder_path or "").replace("\\", "/").strip("/")
    if not normalized_path:
        return None

    existing = {
        folder.path: folder
        for folder in await list_project_folders(db, project_id)
    }
    if normalized_path in existing:
        return existing[normalized_path]

    parent: Folder | None = None
    cursor = ""
    created: Folder | None = None
    for segment in normalized_path.split("/"):
        cursor = f"{cursor}/{segment}" if cursor else segment
        current = existing.get(cursor)
        if current is None:
            current = Folder(
                project_id=project_id,
                parent_id=parent.id if parent is not None else None,
                name=segment,
                path=cursor,
            )
            db.add(current)
            await db.flush()
            existing[cursor] = current
        parent = current
        created = current
    return created


async def ensure_unique_folder_path(db: AsyncSession, project_id: str, requested_path: str) -> str:
    normalized = str(requested_path or "").replace("\\", "/").strip("/")
    if not normalized:
        normalized = "folder"

    existing_paths = {folder.path for folder in await list_project_folders(db, project_id)}
    if normalized not in existing_paths:
        return normalized

    path = PurePosixPath(normalized)
    parent = str(path.parent).replace("\\", "/").strip("/")
    stem = path.name or "folder"
    suffix = 2
    while True:
        candidate_leaf = f"{stem}_{suffix}"
        candidate = f"{parent}/{candidate_leaf}" if parent else candidate_leaf
        if candidate not in existing_paths:
            return candidate
        suffix += 1
