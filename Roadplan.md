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



MAL extension plan (designed so v1 doesn’t get rewritten)
MAL goal

Show model suggestions in the UI, let user accept/adjust quickly, and log outcomes so you can improve the model and sampling.

Core idea: add “suggestions” as first-class objects

Don’t stuff suggestions into annotations. Suggestions are proposals that can be accepted or rejected.

New entity: prediction_suggestions

Minimal fields:

suggestion_id

asset_id

project_id

model_id (or model_version_id)

payload_json (same shape as annotation payload)

score / confidence

created_at

status: pending | accepted | rejected | superseded

Payload examples:

{ "type": "classification", "category_ids": [2], "scores": {"2": 0.91, "1": 0.06} }

New entity: models

model_id

project_id

name

task_type

labels_hash (must match dataset manifest categories hash)

artifact_uri (onnx path)

created_at

Rule: if labels_hash mismatches the project categories snapshot/hash, the UI must warn and the API should block suggestions.

MAL v1 (simple, high leverage)
Flow

User opens asset in labeling UI

UI calls: GET /assets/:id/suggestions?model=active

UI displays:

top suggested label

confidence bar

top-k optional

Hotkey to accept suggestion (e.g. A)

When accepted: UI writes normal annotation via existing endpoint

API endpoints

POST /projects/:id/models (register ONNX + metadata)

GET /projects/:id/models (list)

POST /models/:id/suggest (enqueue batch suggestion job OR inline for single)

GET /assets/:asset_id/suggestions?model_id=...

Worker jobs

inference_suggest.py:

loads ONNX once per worker process

runs inference for a batch of assets

writes prediction_suggestions

MAL v2 (batch + queue integration)
Add “Suggest next N”

UI button: “Generate suggestions for unlabeled”

API: POST /projects/:id/suggestions:batch with criteria

Worker runs inference over assets and stores suggestions

UI queue shows “suggested” badge and lets you accept with 1 keypress

MAL v3 (active learning sampling)

Add a sampler job that selects which assets should be labeled next:

entropy / margin sampling

disagreement across ensemble models

coverage balancing across videos/time

New table: sampling_runs (optional), or just store in assets.metadata_json.

MAL v4 (geometry tasks later)

Because your annotation payloads are typed, you can extend:

bbox suggestions

polygon suggestions
…and the UI overlay tools start using the same payload shapes.