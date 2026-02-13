Goal

Build a fast, reliable labeling UI that exports training-ready datasets with immutable label definitions and versioned exports, ensuring label consistency across labeling → training → deployment.

Non-goals (v1)

Model-assisted labeling

Bounding boxes / polygons / keypoints

Active learning / automatic sampling

Multi-user roles/permissions beyond “who labeled it”

These are planned later, but v1 is structured to add them without breaking the dataset contract.

Guiding Principles

Dataset contract first: UI and backend must produce exports with a stable schema.

Immutable category IDs: label reorder/rename must not break training class order.

Version everything: every export is a dataset version with a manifest hash.

Boring is good: keep DB and API simple, avoid premature microservices.

Deliverables

Labeling web app (classification-first)

Backend API for assets + annotations + export

Exporter that produces:

images/ (frames/images)

annotations.json (COCO-style)

manifest.json (dataset metadata + categories + schema_version)

Milestones
M0 — Repo + scaffolding (Day 0–1)

Monorepo layout (frontend, backend, shared)

Docker compose for local dev (db + api + web)

Basic CI: lint + typecheck + unit tests

Acceptance

docker compose up starts web + api + db

Health endpoints return OK

M1 — Project setup + Categories (v1 cornerstone) (Day 1–3)

Create Project

Categories editor:

Add category

Rename category

Reorder categories

Disable/delete (soft delete)

Category IDs are immutable

IDs never change once created

Reorder only affects UI display order

Project config includes task_type = classification_single

Acceptance

Category list is stored and returned with stable id

Reordering does not change ids

Project has a “schema_version” field set to 1.0.0

M2 — Asset ingestion (images first) (Day 3–5)

Upload images (drag/drop, multi-select)

Store:

asset_id

file location (local/dev storage ok)

width/height

mime type

checksum

Asset list + filters:

All / Unlabeled / In Review / Done

Acceptance

Upload 100+ images reliably

Asset grid loads quickly with thumbnails

Filters work and are correct

M3 — Labeling UI (single-label classification) (Day 5–8)

Labeling screen layout:

Left: queue (next/prev, filters)

Center: viewer (image) + overlay layer (even if empty in v1)

Right: label panel (categories)

Hotkeys:

1..9 assigns category by display order

Arrow keys: prev/next

S: skip (sets status = skipped)

Autosave + optimistic UI

Annotation states:

unlabeled

labeled

skipped

needs_review

approved

Acceptance

Labeling 500 images feels smooth

No lost labels on refresh

Status counters match DB

M4 — Review Mode + QA (Day 8–10)

Review grid:

filter by category

filter by status

show label counts

Bulk actions:

mark approved

set needs_review

Simple issues panel:

missing labels

“skipped” count

Acceptance

Reviewer can validate a dataset quickly

Stats are accurate

M5 — Export v1 (COCO-style + manifest) (Day 10–12)

Export “Dataset Version”:

Snapshot categories

Select assets by status (default: approved only)

Generate:

images/ (copied or hardlinked)

annotations.json (COCO-ish: images, annotations, categories)

manifest.json (dataset_id, version, schema_version, task_type, categories, hash)

Zip + downloadable link

COCO JSON uses the canonical sections like images, annotations, categories.

Acceptance

Export is reproducible (same selection => same hash)

Training script can read categories/order from manifest.json

M6 — Video ingestion (frames as assets) (Day 12–16)

Upload video

Server extracts frames:

fps sampling or “N frames per video”

store provenance: video_id, timestamp_ms, frame_index

Frames are labeled exactly like images

Acceptance

Upload mp4, see extracted frames

Export includes frames as normal images + provenance metadata

Post-v1 roadmap (planned)
P1 — Multi-label classification

category_ids: number[] in annotation payload

UI: toggles/checklist + hotkeys

P2 — Bounding boxes

Viewer overlay tools

Export: COCO bbox fields

P3 — Segmentation/polygons

Polygon tool + mask import

Export: COCO segmentation fields

P4 — Model-assisted labeling

ONNX inference service

Suggestions + “accept” workflow

Active sampling