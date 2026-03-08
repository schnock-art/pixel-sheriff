import uuid

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Asset, Category, Project, Suggestion, Task, TaskKind, TaskLabelMode, TaskType
from sheriff_api.db.session import get_db
from sheriff_api.errors import api_error
from sheriff_api.schemas.projects import ProjectCreate, ProjectRead
from sheriff_api.services.storage import LocalStorage

router = APIRouter(prefix="/projects", tags=["projects"])
settings = get_settings()
storage = LocalStorage(settings.storage_root)


def _task_spec_from_project_task_type(task_type: TaskType) -> tuple[TaskKind, TaskLabelMode | None]:
    if task_type in {TaskType.classification, TaskType.classification_single}:
        if task_type == TaskType.classification:
            return TaskKind.classification, TaskLabelMode.multi_label
        return TaskKind.classification, TaskLabelMode.single_label
    if task_type == TaskType.bbox:
        return TaskKind.bbox, None
    return TaskKind.segmentation, None


def _project_task_type_from_default_task(default_task: Task | None, fallback: TaskType) -> TaskType:
    if default_task is None:
        return fallback
    if default_task.kind == TaskKind.classification:
        if default_task.label_mode == TaskLabelMode.multi_label:
            return TaskType.classification
        return TaskType.classification_single
    if default_task.kind == TaskKind.bbox:
        return TaskType.bbox
    return TaskType.segmentation


def _project_read(project: Project, default_task: Task | None) -> ProjectRead:
    return ProjectRead(
        id=project.id,
        name=project.name,
        task_type=_project_task_type_from_default_task(default_task, project.task_type),
        default_task_id=project.default_task_id,
        schema_version=project.schema_version,
    )


@router.post("", response_model=ProjectRead)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)) -> ProjectRead:
    project_id = str(uuid.uuid4())
    default_task_id = str(uuid.uuid4())
    kind, label_mode = _task_spec_from_project_task_type(payload.task_type)
    default_task = Task(
        id=default_task_id,
        project_id=project_id,
        name="Default",
        kind=kind,
        label_mode=label_mode,
    )
    project = Project(
        id=project_id,
        name=payload.name,
        task_type=payload.task_type,
        default_task_id=default_task_id,
    )

    db.add(project)
    db.add(default_task)
    await db.commit()
    await db.refresh(project)
    return _project_read(project, default_task)


@router.get("", response_model=list[ProjectRead])
async def list_projects(db: AsyncSession = Depends(get_db)) -> list[ProjectRead]:
    projects = list((await db.execute(select(Project))).scalars().all())
    default_task_ids = [project.default_task_id for project in projects if project.default_task_id]
    tasks_by_id: dict[str, Task] = {}
    if default_task_ids:
        task_rows = list((await db.execute(select(Task).where(Task.id.in_(default_task_ids)))).scalars().all())
        tasks_by_id = {row.id: row for row in task_rows}
    return [_project_read(project, tasks_by_id.get(project.default_task_id or "")) for project in projects]


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)) -> ProjectRead:
    project = await db.get(Project, project_id)
    if not project:
        raise api_error(
            status_code=404,
            code="project_not_found",
            message="Project not found",
            details={"project_id": project_id},
        )
    default_task = await db.get(Task, project.default_task_id) if project.default_task_id else None
    return _project_read(project, default_task)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    project = await db.get(Project, project_id)
    if project is None:
        raise api_error(
            status_code=404,
            code="project_not_found",
            message="Project not found",
            details={"project_id": project_id},
        )

    asset_rows = list(
        (
            await db.execute(
                select(Asset.id, Asset.metadata_json).where(Asset.project_id == project_id),
            )
        ).all()
    )
    asset_ids = [asset_id for asset_id, _metadata in asset_rows]
    storage_uris = [
        metadata.get("storage_uri")
        for _asset_id, metadata in asset_rows
        if isinstance(metadata, dict) and isinstance(metadata.get("storage_uri"), str)
    ]

    if asset_ids:
        await db.execute(delete(Suggestion).where(Suggestion.asset_id.in_(asset_ids)))
    await db.execute(delete(Annotation).where(Annotation.project_id == project_id))
    await db.execute(delete(Category).where(Category.project_id == project_id))
    await db.execute(delete(Task).where(Task.project_id == project_id))
    await db.execute(delete(Asset).where(Asset.project_id == project_id))
    await db.delete(project)
    await db.commit()

    for storage_uri in storage_uris:
        try:
            storage.delete_file(storage_uri)
        except ValueError:
            continue

    for relative_dir in (
        f"assets/{project_id}",
        f"exports/{project_id}",
        f"models/{project_id}",
        f"experiments/{project_id}",
        f"datasets/{project_id}",
    ):
        try:
            storage.delete_tree(relative_dir)
        except ValueError:
            continue

    return Response(status_code=status.HTTP_204_NO_CONTENT)
