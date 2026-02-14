# Pixel Sheriff Architecture

## 1. Overview

Pixel Sheriff is a local-first annotation system:
- Web app (`apps/web`)
- API (`apps/api`)
- Postgres metadata store
- Redis placeholder
- Filesystem storage mount (`data/`)

Operational flow:
1. Pick/create project.
2. Import local images to a project folder.
3. Persist images in API storage.
4. Annotate in edit mode (single or multi-label).
5. Submit staged annotations.
6. Generate and download export artifact (`.zip`).

## 2. Runtime Topology

Compose services:
- `web` on `WEB_PORT` (default `3010`)
- `api` on `API_PORT` (default `8010`)
- `db` on `POSTGRES_PORT` (default `5433`)
- `redis` on `REDIS_PORT` (default `6380`)

Networking:
- Web currently calls API directly using `NEXT_PUBLIC_API_BASE_URL` (`http://localhost:${API_PORT}`).
- Next rewrite proxy (`/api/v1/* -> INTERNAL_API_BASE_URL`) is also configured as fallback.
- API CORS supports configured origins plus localhost/127.0.0.1 regex.

## 3. Backend (`apps/api`)

### App Composition
- FastAPI app in `apps/api/src/sheriff_api/main.py`
- Startup creates database tables
- Routers mounted under `/api/v1`

### Data Model
Defined in `apps/api/src/sheriff_api/db/models.py`:
- `Project`, `Category`, `Asset`, `Annotation`, `DatasetVersion`
- `Model`, `Suggestion` (MAL placeholders)

Key constraints:
- One annotation row per asset (`annotations.asset_id` unique)
- Category identity stable (ID immutable; only mutable fields patchable)

### Storage Layer
`LocalStorage` in `apps/api/src/sheriff_api/services/storage.py`:
- Creates per-project storage directories
- Resolves and validates paths within storage root
- Writes uploaded bytes to disk

Upload metadata captures:
- `original_filename`
- `relative_path`
- `storage_uri`
- `size_bytes`

## 4. API Surface (Implemented)

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

## 5. Web (`apps/web`)

Main screen: `apps/web/src/app/page.tsx`

Layout:
- Left: dataset list + hierarchical file tree
- Center: viewer + pagination + nav
- Right: label panel

Implemented behaviors:
- Import modal:
  - existing/new project destination
  - editable target folder name
- Hierarchical folder/file tree generation from asset relative paths
- Viewer letterbox rendering (`contain` on black background)
- Arrow key image navigation
- Label management mode:
  - add labels
  - rename/reorder/activate/deactivate
- Edit mode:
  - stage per-asset edits
  - batch submit staged annotations
- Multi-label toggle in edit mode

## 6. Annotation Payload Shape

Classification payload currently includes:
- `type: "classification"`
- `category_id` (primary)
- `category_ids` (multi-label-capable)
- `category_name`
- `coco` mapping (`image_id`, `category_id`)

Statuses:
- `unlabeled`, `labeled`, `skipped`, `needs_review`, `approved`

## 7. Export Path

Current implementation:
- Export endpoint builds a deterministic COCO-style artifact and creates a dataset version record.
- Artifact is written under storage root at `exports/{project_id}/{hash}.zip`.
- Download endpoint serves the artifact by `{project_id}` and `{content_hash}`.
- Bundle structure:
  - `manifest.json`
  - `annotations.json`
  - `images/...` (for assets that have local stored bytes)

## 8. Known Gaps

- Upload endpoint does not yet validate project existence before write
- Image dimensions (`width/height`) not extracted on upload
- Review/QA UI not implemented
- Auth/multi-user permissions not implemented
- MAL functionality remains placeholder
