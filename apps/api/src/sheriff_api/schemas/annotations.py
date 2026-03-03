from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sheriff_api.db.models import AnnotationStatus


class LegacyClassificationPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = None
    category_id: int | None = None
    category_ids: list[int] = Field(default_factory=list)
    coco: dict = Field(default_factory=dict)
    source: str | None = None


class GeometryBBoxObject(BaseModel):
    id: str
    kind: Literal["bbox"]
    category_id: int
    bbox: list[float]


class GeometryPolygonObject(BaseModel):
    id: str
    kind: Literal["polygon"]
    category_id: int
    segmentation: list[list[float]]


class AnnotationImageBasis(BaseModel):
    width: int
    height: int


class AnnotationClassificationV2(BaseModel):
    category_ids: list[int] = Field(default_factory=list)
    primary_category_id: int | None = None


class AnnotationPayloadV2(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str = "2.0"
    classification: AnnotationClassificationV2 = Field(default_factory=AnnotationClassificationV2)
    objects: list[GeometryBBoxObject | GeometryPolygonObject] = Field(default_factory=list)
    image_basis: AnnotationImageBasis | None = None
    source: str | None = None


class AnnotationUpsert(BaseModel):
    asset_id: str
    status: AnnotationStatus = AnnotationStatus.labeled
    payload_json: LegacyClassificationPayload | AnnotationPayloadV2
    annotated_by: str | None = None


class AnnotationRead(BaseModel):
    id: str
    asset_id: str
    project_id: str
    status: AnnotationStatus
    payload_json: dict
    annotated_by: str | None

    class Config:
        from_attributes = True
