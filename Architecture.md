# Pixel Sheriff Architecture

## 1) System Overview

Pixel Sheriff is a local-first annotation platform:
- Web UI (`apps/web`, Next.js)
- API (`apps/api`, FastAPI + SQLAlchemy)
- Postgres metadata store
- Redis placeholder for async workflows
- Filesystem-backed asset storage (`data/`)

Primary flow:
1. Select/create project
2. Import images from local folder
3. Upload bytes to API storage
4. Annotate in edit mode (single or multi-label)
5. Submit staged annotations
6. Export metadata snapshot (zip generation pending)

## 2) Runtime Topology

Docker Compose services:
- `web` -> host `WEB_PORT` (current example: `3010`)
- `api` -> host `API_PORT` (current example: `8010`)
- `db` -> host `POSTGRES_PORT` (`5433`)
- `redis` -> host `REDIS_PORT` (`6380`)

Web/API communication:
- Browser currently calls API using `NEXT_PUBLIC_API_BASE_URL` (set in compose to `http://localhost:${API_PORT}`)
- Next rewrite is also configured for `/api/v1/* -> INTERNAL_API_BASE_URL` and can be used as fallback strategy

CORS:
- Explicit origins from env + localhost/127.0.0.1 regex in API middleware

## 3) Backend Architecture (`apps/api`)

### App composition
- FastAPI app in `apps/api/src/sheriff_api/main.py`
- Startup creates DB tables
- Routers mounted under `/api/v1`

### Data model
From `apps/api/src/sheriff_api/db/models.py`:
- `Project`
- `Category`
- `Asset`
- `Annotation`
- `DatasetVersion`
- `Model` and `Suggestion` (MAL placeholders)

Important behavior:
- At most one annotation row per asset (`annotations.asset_id` unique)
- Category IDs remain stable; patch updates mutable fields only

### Storage
`LocalStorage` in `apps/api/src/sheriff_api/services/storage.py`:
- Ensures project directories
- Safe path resolution (no storage-root escape)
- Writes uploaded bytes to disk

Upload endpoint stores:
- asset record with `uri=/api/v1/assets/{asset_id}/content`
- metadata with original filename and relative path provenance

## 4) Web Architecture (`apps/web`)

Main page: `apps/web/src/app/page.tsx`

Key UI areas:
- Left: datasets + file tree
- Center: viewer + pagination/navigation
- Right: label panel

Features currently wired:
- Import modal (existing/new project + target folder name)
- Hierarchical file tree generated from stored relative paths
- Edit mode with staged annotation changes
- Batch submit from staged edits
- Multi-label toggle (edit mode only)
- Label management mode (add/rename/reorder/activate/deactivate)
- Arrow-key image navigation

Core components:
- `AssetGrid`
- `Filters`
- `Viewer`
- `LabelPanel`
- `Pagination`

Hooks:
- `useProject`
- `useAssets`
- `useLabels`

API client:
- `apps/web/src/lib/api.ts`

## 5) Annotation Model in UI/API

Submission payloads use classification shape with COCO-oriented fields:
- `type: "classification"`
- `category_id` (primary)
- `category_ids` (supports multi-label)
- `coco: { image_id, category_id }`

Status values:
- `unlabeled`, `labeled`, `skipped`, `needs_review`, `approved`

## 6) Import Pipeline

1. User picks a local folder
2. UI opens import modal:
   - existing/new project
   - project name or dropdown
   - target folder name
3. UI uploads each image to:
   - `POST /api/v1/projects/{project_id}/assets/upload`
4. API stores file in `STORAGE_ROOT/assets/{project_id}/...`
5. API serves file via:
   - `GET /api/v1/assets/{asset_id}/content`
6. Viewer loads resolved `asset.uri`

## 7) Export Path

Current:
- Export endpoint creates dataset version metadata + manifest hash

Pending:
- Zip build and downloadable artifact endpoint

## 8) Known Gaps

- Upload endpoint does not currently validate project existence before writing
- Image dimensions are not extracted/populated on upload
- No dedicated review UI yet
- No auth/multi-user access controls
- Worker/MAL flow still placeholder
