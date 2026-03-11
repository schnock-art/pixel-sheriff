import enum
import uuid
from datetime import datetime
from pathlib import PurePosixPath

from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _legacy_relative_path(metadata_json: dict | None, uri: str, asset_id: str) -> str:
    metadata = metadata_json if isinstance(metadata_json, dict) else {}
    relative_path = metadata.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        return relative_path.replace("\\", "/").strip("/")
    original_filename = metadata.get("original_filename")
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.replace("\\", "/").strip("/")
    uri_path = str(uri or "").strip().replace("\\", "/")
    if uri_path:
        return PurePosixPath(uri_path).name
    return asset_id


class TaskType(str, enum.Enum):
    classification = "classification"
    classification_single = "classification_single"
    bbox = "bbox"
    segmentation = "segmentation"


class TaskKind(str, enum.Enum):
    classification = "classification"
    bbox = "bbox"
    segmentation = "segmentation"


class TaskLabelMode(str, enum.Enum):
    single_label = "single_label"
    multi_label = "multi_label"


class AnnotationStatus(str, enum.Enum):
    unlabeled = "unlabeled"
    labeled = "labeled"
    skipped = "skipped"
    needs_review = "needs_review"
    approved = "approved"


class AssetType(str, enum.Enum):
    image = "image"
    video = "video"
    frame = "frame"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    task_type: Mapped[TaskType] = mapped_column(Enum(TaskType, name="tasktype"), default=TaskType.classification_single)
    default_task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", name="fk_projects_default_task_id", use_alter=True),
        nullable=True,
        index=True,
    )
    schema_version: Mapped[str] = mapped_column(String, default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_task_project_name"),
        CheckConstraint(
            "(kind = 'classification' AND label_mode IS NOT NULL) OR (kind != 'classification' AND label_mode IS NULL)",
            name="ck_task_kind_label_mode",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", name="fk_tasks_project_id", ondelete="CASCADE"),
        index=True,
    )
    kind: Mapped[TaskKind] = mapped_column(Enum(TaskKind, name="taskkind"), nullable=False)
    label_mode: Mapped[TaskLabelMode | None] = mapped_column(Enum(TaskLabelMode, name="tasklabelmode"), nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    legacy_int_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Folder(Base):
    __tablename__ = "folders"
    __table_args__ = (UniqueConstraint("project_id", "path", name="uq_folder_project_path"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", name="fk_folders_project_id", ondelete="CASCADE"),
        index=True,
    )
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("folders.id", name="fk_folders_parent_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent: Mapped["Folder | None"] = relationship("Folder", remote_side="Folder.id")
    assets: Mapped[list["Asset"]] = relationship("Asset", back_populates="folder")
    sequence: Mapped["AssetSequence | None"] = relationship("AssetSequence", back_populates="folder", uselist=False)


class AssetSequence(Base):
    __tablename__ = "asset_sequences"
    __table_args__ = (UniqueConstraint("folder_id", name="uq_asset_sequences_folder_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", name="fk_asset_sequences_project_id", ondelete="CASCADE"),
        index=True,
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", name="fk_asset_sequences_task_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    folder_id: Mapped[str | None] = mapped_column(
        ForeignKey("folders.id", name="fk_asset_sequences_folder_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="processing")
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    folder: Mapped[Folder | None] = relationship("Folder", back_populates="sequence")
    assets: Mapped[list["Asset"]] = relationship("Asset", back_populates="sequence")


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    type: Mapped[AssetType] = mapped_column(Enum(AssetType), default=AssetType.image)
    folder_id: Mapped[str | None] = mapped_column(
        ForeignKey("folders.id", name="fk_assets_folder_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    file_name: Mapped[str | None] = mapped_column(String, nullable=True)
    sequence_id: Mapped[str | None] = mapped_column(
        ForeignKey("asset_sequences.id", name="fk_assets_sequence_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_kind: Mapped[str] = mapped_column(String, default="image")
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    folder: Mapped[Folder | None] = relationship("Folder", back_populates="assets")
    sequence: Mapped[AssetSequence | None] = relationship("AssetSequence", back_populates="assets")

    @property
    def resolved_file_name(self) -> str:
        if isinstance(self.file_name, str) and self.file_name.strip():
            return self.file_name
        return PurePosixPath(_legacy_relative_path(self.metadata_json, self.uri, self.id)).name

    @property
    def folder_path(self) -> str | None:
        folder = self.__dict__.get("folder")
        if folder is not None and isinstance(folder.path, str) and folder.path.strip():
            return folder.path
        rel = _legacy_relative_path(self.metadata_json, self.uri, self.id)
        parent = str(PurePosixPath(rel).parent).replace("\\", "/").strip("/")
        if parent in {"", "."}:
            return None
        return parent

    @property
    def relative_path(self) -> str:
        folder_path = self.folder_path
        file_name = self.resolved_file_name
        if folder_path:
            return f"{folder_path}/{file_name}"
        return file_name


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("asset_id", "task_id", name="uq_annotation_asset_task"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    status: Mapped[AnnotationStatus] = mapped_column(Enum(AnnotationStatus), default=AnnotationStatus.unlabeled)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    annotated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PrelabelSession(Base):
    __tablename__ = "prelabel_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", name="fk_prelabel_sessions_project_id", ondelete="CASCADE"),
        index=True,
    )
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", name="fk_prelabel_sessions_task_id", ondelete="CASCADE"),
        index=True,
    )
    sequence_id: Mapped[str] = mapped_column(
        ForeignKey("asset_sequences.id", name="fk_prelabel_sessions_sequence_id", ondelete="CASCADE"),
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    prompts_json: Mapped[list] = mapped_column(JSON, default=list)
    sampling_mode: Mapped[str] = mapped_column(String, nullable=False, default="every_n_frames")
    sampling_value: Mapped[float] = mapped_column(Float, nullable=False, default=15.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    max_detections_per_frame: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    live_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    input_closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    enqueued_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generated_proposals: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_unmatched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PrelabelProposal(Base):
    __tablename__ = "prelabel_proposals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("prelabel_sessions.id", name="fk_prelabel_proposals_session_id", ondelete="CASCADE"),
        index=True,
    )
    asset_id: Mapped[str] = mapped_column(
        ForeignKey("assets.id", name="fk_prelabel_proposals_asset_id", ondelete="CASCADE"),
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", name="fk_prelabel_proposals_project_id", ondelete="CASCADE"),
        index=True,
    )
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", name="fk_prelabel_proposals_task_id", ondelete="CASCADE"),
        index=True,
    )
    category_id: Mapped[str] = mapped_column(
        ForeignKey("categories.id", name="fk_prelabel_proposals_category_id", ondelete="CASCADE"),
        index=True,
    )
    label_text: Mapped[str] = mapped_column(String, nullable=False)
    prompt_text: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bbox_json: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    reviewed_bbox_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reviewed_category_id: Mapped[str | None] = mapped_column(String, nullable=True)
    promoted_annotation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    promoted_object_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    selection_criteria_json: Mapped[dict] = mapped_column(JSON, default=dict)
    manifest_json: Mapped[dict] = mapped_column(JSON, default=dict)
    export_uri: Mapped[str] = mapped_column(String, nullable=False)
    hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Suggestion(Base):
    __tablename__ = "suggestions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
