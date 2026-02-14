# Pixel Sheriff Architecture

## 1. System Overview

Pixel Sheriff is a local-first image annotation system with three active runtime layers:

- Web app (`apps/web`, Next.js App Router)
- API (`apps/api`, FastAPI + SQLAlchemy async)
- Local infra services (Postgres + Redis via Docker Compose)

Primary workflow:

1. Select or create a project.
2. Import a local folder into a project/folder destination.
3. API persists image bytes to local storage and records asset metadata in Postgres.
4. Annotate in classification mode (single-label or project-wide multi-label).
5. Submit staged edits (or submit single-image edits directly).
6. Generate and download a COCO-style export zip.

## 2. Runtime Topology

Compose services and default host ports:

- `web` -> `WEB_PORT` (default `3010`)
- `api` -> `API_PORT` (default `8010`)
- `db` -> `POSTGRES_PORT` (default `5433`)
- `redis` -> `REDIS_PORT` (default `6380`)

Persistence:

- DB data in `postgres_data` volume
- Binary assets + export zips in repo-local `./data` (mounted to `/app/data` in API container)

Network behavior:

- Browser API calls use `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:${API_PORT}`)
- Next rewrite proxy remains configured in `apps/web/next.config.js`
- API CORS accepts configured origins plus localhost/127.0.0.1 regex

## 3. Backend Architecture (`apps/api`)

### App Composition

- Entry: `apps/api/src/sheriff_api/main.py`
- DB schema initialization runs on startup (`Base.metadata.create_all`)
- Routers mounted under `/api/v1`

### Data Model

Defined in `apps/api/src/sheriff_api/db/models.py`:

- Core: `Project`, `Category`, `Asset`, `Annotation`, `DatasetVersion`
- Placeholder MAL domain: `Model`, `Suggestion`

Key invariants:

- One annotation row per asset (`annotations.asset_id` unique)
- Category IDs are stable identities; label edits mutate name/order/active only

### Storage

`LocalStorage` (`apps/api/src/sheriff_api/services/storage.py`) provides:

- per-project directory setup
- safe path resolution constrained to storage root
- byte writes and reads for uploaded assets/exports

Upload endpoint (`/projects/{project_id}/assets/upload`) writes files as:

- `assets/{project_id}/{asset_uuid}{extension}`

Upload metadata stored in `assets.metadata_json`:

- `storage_uri`
- `original_filename`
- `relative_path`
- `size_bytes`

### Export Builder

`apps/api/src/sheriff_api/services/exporter_coco.py` builds deterministic export artifacts:

- `manifest.json`
- `annotations.json`
- `images/...` (for assets with local bytes available)

Export files are persisted at:

- `exports/{project_id}/{content_hash}.zip`

## 4. Implemented API Surface

Health:

- `GET /api/v1/health`

Projects:

- `POST /api/v1/projects`
- `GET /api/v1/projects`
- `GET /api/v1/projects/{project_id}`

Categories:

- `POST /api/v1/projects/{project_id}/categories`
- `GET /api/v1/projects/{project_id}/categories`
- `PATCH /api/v1/categories/{category_id}`

Assets:

- `POST /api/v1/projects/{project_id}/assets`
- `GET /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `GET /api/v1/assets/{asset_id}/content`

Annotations:

- `POST /api/v1/projects/{project_id}/annotations` (upsert)
- `GET /api/v1/projects/{project_id}/annotations`

Exports:

- `POST /api/v1/projects/{project_id}/exports`
- `GET /api/v1/projects/{project_id}/exports`
- `GET /api/v1/projects/{project_id}/exports/{content_hash}/download`

MAL placeholders:

- `POST /api/v1/models`
- `GET /api/v1/models`
- `GET /api/v1/assets/{asset_id}/suggestions`
- `POST /api/v1/projects/{project_id}/suggestions/batch`

## 5. Web Architecture (`apps/web`)

Main workspace:

- `apps/web/src/app/page.tsx`

UI structure:

- Left: project list + hierarchical file tree
- Center: viewer canvas + adaptive pagination + skip/nav controls
- Right: label panel (label selection + manage mode + edit/submit)

### Frontend Data/State

Custom hooks:

- `useProject`: project listing
- `useAssets`: assets + annotations for active project
- `useLabels`: categories for active project

Local persisted setting:

- project multi-label map in `localStorage` (`pixel-sheriff:project-multilabel:v1`)

### Implemented UX Behaviors

- Import dialog supports:
  - existing vs new project target
  - optional existing folder/subfolder target
  - editable folder path
- Import progress panel shows:
  - percent + completed/total files
  - bytes processed / total
  - upload speed + file rate
  - elapsed + ETA
  - uploaded/failed/remaining counts
- File tree behavior:
  - deterministic hierarchy from `relative_path`
  - per-folder expand/collapse + global collapse/expand
  - folder-scope review queue filtering
  - labeled/unlabeled indicators for files and folders
- Viewer behavior:
  - black letterbox with image `contain`
  - bounded responsive viewport height
  - keyboard `ArrowLeft`/`ArrowRight`
  - skip controls: `-10`, `-5`, `<`, `>`, `+5`, `+10`
- Pagination behavior:
  - width-adaptive page-token window
  - `First`/`Last` chips
  - labeled/unlabeled color status
- Label panel behavior:
  - manage mode for create/rename/reorder/activate/deactivate
  - project multi-label toggle only editable in manage mode
  - edit mode stages multi-asset changes
  - submit commits staged changes in batch

## 6. Annotation Payload Contract

Classification payload includes:

- `type: "classification"`
- `category_id` (primary label)
- `category_ids` (multi-label-compatible)
- `category_name`
- `coco` object with at least `image_id` and `category_id`

Supported statuses:

- `unlabeled`
- `labeled`
- `skipped`
- `needs_review`
- `approved`

## 7. Known Gaps

- Upload endpoint still does not validate project existence before writing bytes
- Image metadata extraction (`width`/`height`) is not populated on upload
- Review/QA moderation workflow not implemented
- Bounding box and segmentation tooling not implemented
- Auth/multi-user permissions not implemented
- MAL routes are placeholders only
