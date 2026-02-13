1. Product Definition

A web-based labeling tool that supports:

Projects with a stable label taxonomy (categories)

Asset ingestion (images first; videos as frames)

Annotation workflow (label → review → export)

Dataset export with immutable label mapping for training/deploy consistency

2. Core Concepts
2.1 Project

Represents a labeling job.

Fields (logical):

project_id (uuid)

name

task_type (enum)

classification_single (v1)

future: classification_multi, bbox, polygon, keypoints

schema_version ("1.0.0")

created_at, updated_at

2.2 Category (Label taxonomy)

category_id (int or uuid; immutable)

project_id

name

display_order (int)

is_active (bool)

Rules:

category_id never changes

Reorder only updates display_order

Rename updates name but keeps category_id

2.3 Asset

A unit to label.

Types:

image

video

frame (derived from video, treated as image in UI)

Fields:

asset_id (uuid)

project_id

type

uri (storage path)

mime_type

width, height

checksum

metadata_json (includes provenance for frames)

created_at

2.4 Annotation

A label attached to an asset.

Fields:

annotation_id (uuid)

asset_id

project_id (denormalized for convenience)

status (enum: unlabeled, labeled, skipped, needs_review, approved)

payload_json (schema depends on task_type)

annotated_by (string/user_id optional)

created_at, updated_at

Payload (v1 classification single):

{
  "type": "classification",
  "category_ids": [2]
}


Future payloads:

Multi-label: category_ids: [2, 5, 9]

BBox:

{ "type":"bbox", "category_id":2, "bbox":[x,y,w,h] }


Polygon/segmentation:

{ "type":"polygon", "category_id":2, "segmentation":[[x1,y1,x2,y2,...]] }


COCO supports annotations like bbox/segmentation in its JSON schema, which is why it’s a good long-term export target.

2.5 Dataset Version (Export)

Represents a frozen snapshot of a project’s labels and selected assets.

Fields:

dataset_version_id (uuid)

project_id

created_at

selection_criteria_json (e.g. approved-only)

manifest_json

export_uri (zip location)

hash (content hash of manifest + annotations selection)

3. Storage
3.1 File storage (dev)

Local filesystem (e.g. ./data/assets)

Structure:

assets/{project_id}/{asset_id}/original

assets/{project_id}/{asset_id}/thumb.jpg

exports/{project_id}/{dataset_version_id}.zip

3.2 Database

Postgres recommended.

Tables:

projects

categories

assets

annotations

dataset_versions

Indexes:

(project_id, status) on annotations

(project_id) on everything

(asset_id) unique on annotations if you enforce 1 annotation per asset for v1

4. API Contracts (REST)

All endpoints are versioned: /api/v1/...

4.1 Projects

POST /projects

GET /projects

GET /projects/:id

4.2 Categories

POST /projects/:id/categories

GET /projects/:id/categories

PATCH /categories/:category_id (rename, reorder, deactivate)

4.3 Assets

POST /projects/:id/assets/upload (multipart)

GET /projects/:id/assets?status=...

GET /assets/:asset_id (metadata)

GET /assets/:asset_id/content (file stream)

GET /assets/:asset_id/thumb

4.4 Annotations

PUT /assets/:asset_id/annotation

GET /assets/:asset_id/annotation

PATCH /assets/:asset_id/annotation/status

4.5 Export

POST /projects/:id/exports (create dataset version)

GET /projects/:id/exports

GET /exports/:dataset_version_id/download

5. Export Format
5.1 manifest.json (required)
{
  "dataset_id": "uuid",
  "dataset_version_id": "uuid",
  "created_at": "2026-02-13T10:00:00+10:00",
  "task_type": "classification_single",
  "schema_version": "1.0.0",
  "categories": [
    { "id": 1, "name": "cat" },
    { "id": 2, "name": "dog" }
  ],
  "hash": "sha256:..."
}

5.2 annotations.json (COCO-style)

Top-level keys include (at minimum):

images: list of image records (id, file_name, width, height)

annotations: list of annotations referencing image ids

categories: list of categories (id, name)

This structure matches common COCO JSON conventions.

5.3 images/

All exported assets as image files. For videos, export the extracted frames.

6. Frontend Architecture (UI)

Routes:

/projects

/projects/:id/settings (task + categories)

/projects/:id/assets (grid + filters)

/projects/:id/label (labeling workstation)

/projects/:id/review (QA grid)

/projects/:id/exports (versions + download)

Key components:

AssetGrid

LabelingWorkstation

CategoryPanel

KeyboardShortcuts

ReviewGrid

ExportDialog

State rules:

Optimistic UI for annotation saves

Debounced autosave

Local “dirty” indicator until server confirms

7. Consistency & “Label Drift” Prevention

Training reads categories from manifest.json only.

Deployment embeds hash and categories snapshot.

If model’s embedded hash != dataset export hash → hard fail (refuse deploy).

8. Testing Strategy

Unit: category immutability + reorder semantics

Unit: exporter produces deterministic hashes

Integration: upload → label → export → download zip

UI smoke: hotkeys + autosave