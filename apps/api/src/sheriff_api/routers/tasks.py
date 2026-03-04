from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Category, Project, Task, TaskKind, TaskLabelMode
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.tasks import TaskCreate, TaskRead

router = APIRouter(tags=["tasks"])
settings = get_settings()


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, (dict, list)):
        return raw
    return None


def _file_counts_for_task(project_id: str, task_id: str) -> dict[str, int]:
    root = Path(settings.storage_root)

    dataset_count = 0
    datasets_path = root / "datasets" / project_id / "datasets.json"
    datasets_doc = _read_json(datasets_path)
    if isinstance(datasets_doc, dict):
        for item in datasets_doc.get("items", []):
            if isinstance(item, dict) and str(item.get("task_id") or "") == task_id:
                dataset_count += 1

    model_count = 0
    models_path = root / "models" / project_id / "records.json"
    models_doc = _read_json(models_path)
    if isinstance(models_doc, list):
        for row in models_doc:
            if not isinstance(row, dict):
                continue
            if str(row.get("task_id") or "") == task_id:
                model_count += 1
                continue
            source_dataset = row.get("config_json", {}).get("source_dataset") if isinstance(row.get("config_json"), dict) else {}
            if isinstance(source_dataset, dict) and str(source_dataset.get("task_id") or "") == task_id:
                model_count += 1

    experiment_count = 0
    experiments_path = root / "experiments" / project_id / "records.json"
    experiments_doc = _read_json(experiments_path)
    if isinstance(experiments_doc, list):
        for row in experiments_doc:
            if not isinstance(row, dict):
                continue
            if str(row.get("task_id") or "") == task_id:
                experiment_count += 1
                continue
            config_json = row.get("config_json")
            if isinstance(config_json, dict) and str(config_json.get("task_id") or "") == task_id:
                experiment_count += 1

    deployment_count = 0
    deployments_path = root / "deployments" / project_id / "deployments.json"
    deployments_doc = _read_json(deployments_path)
    if isinstance(deployments_doc, dict):
        for row in deployments_doc.get("items", []):
            if isinstance(row, dict) and str(row.get("task_id") or "") == task_id:
                deployment_count += 1

    return {
        "datasets": dataset_count,
        "models": model_count,
        "experiments": experiment_count,
        "deployments": deployment_count,
    }


async def _require_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(status_code=404, code="project_not_found", message="Project not found")
    return project


def _task_read(task: Task, *, default_task_id: str | None) -> TaskRead:
    return TaskRead.model_validate(
        {
            "id": task.id,
            "project_id": task.project_id,
            "name": task.name,
            "kind": task.kind,
            "label_mode": task.label_mode,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "is_default": bool(default_task_id and task.id == default_task_id),
        }
    )


@router.post("/projects/{project_id}/tasks", response_model=TaskRead)
async def create_task(project_id: str, payload: TaskCreate, db: AsyncSession = Depends(get_db)) -> TaskRead:
    project = await _require_project(db, project_id)
    name = payload.name.strip()
    if not name:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="Task name is required",
            details={"project_id": project_id, "field": "name"},
        )

    label_mode = payload.label_mode
    if payload.kind == TaskKind.classification:
        if label_mode is None:
            label_mode = TaskLabelMode.single_label
    elif label_mode is not None:
        raise api_error(
            status_code=422,
            code="validation_error",
            message="label_mode is allowed only for classification tasks",
            details={"kind": payload.kind.value},
        )

    task = Task(
        project_id=project_id,
        name=name,
        kind=payload.kind,
        label_mode=label_mode,
    )
    db.add(task)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise api_error(
            status_code=409,
            code="task_name_conflict",
            message="Task name already exists in project",
            details={"project_id": project_id, "name": name},
        ) from exc
    await db.refresh(task)

    if not project.default_task_id:
        project.default_task_id = task.id
        await db.commit()
        await db.refresh(project)

    return _task_read(task, default_task_id=project.default_task_id)


@router.get("/projects/{project_id}/tasks", response_model=list[TaskRead])
async def list_tasks(project_id: str, db: AsyncSession = Depends(get_db)) -> list[TaskRead]:
    project = await _require_project(db, project_id)
    rows = list((await db.execute(select(Task).where(Task.project_id == project_id).order_by(Task.created_at, Task.id))).scalars().all())
    return [_task_read(task, default_task_id=project.default_task_id) for task in rows]


@router.get("/projects/{project_id}/tasks/{task_id}", response_model=TaskRead)
async def get_task(project_id: str, task_id: str, db: AsyncSession = Depends(get_db)) -> TaskRead:
    project = await _require_project(db, project_id)
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )
    return _task_read(task, default_task_id=project.default_task_id)


@router.delete("/projects/{project_id}/tasks/{task_id}")
async def delete_task(project_id: str, task_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _require_project(db, project_id)
    task = await db.get(Task, task_id)
    if task is None or task.project_id != project_id:
        raise api_error(
            status_code=404,
            code="task_not_found",
            message="Task not found in project",
            details={"project_id": project_id, "task_id": task_id},
        )

    task_rows = list((await db.execute(select(Task).where(Task.project_id == project_id))).scalars().all())
    if len(task_rows) <= 1:
        raise api_error(
            status_code=409,
            code="project_must_have_task",
            message="Project must have at least one task",
            details={"project_id": project_id, "task_id": task_id},
        )

    category_count = int((await db.execute(select(func.count(Category.id)).where(Category.task_id == task_id))).scalar_one())
    annotation_count = int((await db.execute(select(func.count(Annotation.id)).where(Annotation.task_id == task_id))).scalar_one())
    file_counts = _file_counts_for_task(project_id, task_id)
    details = {
        "project_id": project_id,
        "task_id": task_id,
        "references": {
            "categories": category_count,
            "annotations": annotation_count,
            **file_counts,
        },
    }
    if any(value > 0 for value in [category_count, annotation_count, *file_counts.values()]):
        raise api_error(
            status_code=409,
            code="task_not_empty",
            message="Task cannot be deleted while references exist",
            details=details,
        )

    is_default = project.default_task_id == task_id
    if is_default:
        remaining = sorted([row for row in task_rows if row.id != task_id], key=lambda row: (row.created_at, row.id))
        if not remaining:
            raise api_error(
                status_code=409,
                code="project_must_have_task",
                message="Project must have at least one task",
                details={"project_id": project_id, "task_id": task_id},
            )
        project.default_task_id = remaining[0].id

    await db.delete(task)
    await db.commit()
    return {"ok": True, "task_id": task_id}
