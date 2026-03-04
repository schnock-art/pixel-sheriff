from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from sheriff_api.db.models import AnnotationStatus


DatasetTask = Literal["classification", "bbox", "segmentation"]
SplitName = Literal["train", "val", "test"]


class DatasetSelectionFilters(BaseModel):
    include_labeled_only: bool = False
    include_statuses: list[AnnotationStatus] = Field(default_factory=list)
    exclude_statuses: list[AnnotationStatus] = Field(default_factory=list)
    include_category_ids: list[str] = Field(default_factory=list)
    exclude_category_ids: list[str] = Field(default_factory=list)
    include_folder_paths: list[str] = Field(default_factory=list)
    exclude_folder_paths: list[str] = Field(default_factory=list)
    include_negative_images: bool | None = None


class DatasetSelectionRequest(BaseModel):
    mode: Literal["filter_snapshot", "explicit_asset_ids"] = "filter_snapshot"
    filters: DatasetSelectionFilters = Field(default_factory=DatasetSelectionFilters)
    explicit_asset_ids: list[str] = Field(default_factory=list)


class DatasetSplitRatios(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class DatasetSplitStratify(BaseModel):
    enabled: bool = False
    by: Literal["label_primary", "label_multi_hot", "embedding_cluster"] = "label_primary"
    strict_stratify: bool = False


class DatasetSplitRequest(BaseModel):
    seed: int = 1337
    ratios: DatasetSplitRatios = Field(default_factory=DatasetSplitRatios)
    stratify: DatasetSplitStratify = Field(default_factory=DatasetSplitStratify)


class DatasetPreviewRequest(BaseModel):
    task: DatasetTask
    selection: DatasetSelectionRequest = Field(default_factory=DatasetSelectionRequest)
    split: DatasetSplitRequest = Field(default_factory=DatasetSplitRequest)
    strict_preview_cap: bool = False
    preview_cap: int = Field(default=5000, ge=1, le=50000)


class DatasetPreviewResponse(BaseModel):
    asset_ids: list[str] = Field(default_factory=list)
    sample_asset_ids: list[str] = Field(default_factory=list)
    counts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DatasetVersionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=4000)
    task: DatasetTask
    created_by: str | None = Field(default=None, max_length=200)
    selection: DatasetSelectionRequest = Field(default_factory=DatasetSelectionRequest)
    split: DatasetSplitRequest = Field(default_factory=DatasetSplitRequest)
    set_active: bool = False


class DatasetVersionEnvelope(BaseModel):
    version: dict[str, Any]
    is_archived: bool = False
    is_active: bool = False


class DatasetVersionListResponse(BaseModel):
    active_dataset_version_id: str | None = None
    items: list[DatasetVersionEnvelope] = Field(default_factory=list)


class DatasetSetActiveRequest(BaseModel):
    active_dataset_version_id: str | None = None


class DatasetVersionAssetItem(BaseModel):
    asset_id: str
    filename: str
    relative_path: str
    status: AnnotationStatus
    split: SplitName | None = None
    label_summary: dict[str, Any] = Field(default_factory=dict)


class DatasetVersionAssetsResponse(BaseModel):
    items: list[DatasetVersionAssetItem] = Field(default_factory=list)
    page: int
    page_size: int
    total: int


class DatasetVersionExportResponse(BaseModel):
    dataset_version_id: str
    hash: str
    export_uri: str
