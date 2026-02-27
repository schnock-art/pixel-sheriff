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
4. Annotate in classification, bounding-box, or polygon-segmentation mode.
   - active mode is locked per project by `project.task_type`
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
- Global exception handlers normalize API failures into a structured envelope for UI consumption

### Data Model

Defined in `apps/api/src/sheriff_api/db/models.py`:

- Core: `Project`, `Category`, `Asset`, `Annotation`, `DatasetVersion`
- Placeholder MAL domain: `Model`, `Suggestion`

Key invariants:

- One annotation row per asset (`annotations.asset_id` unique)
- Category IDs are stable identities; label edits mutate name/order/active only
- Project task mode governs accepted annotation payload shape:
  - `classification` / `classification_single`: no geometry objects
  - `bbox`: bbox objects only
  - `segmentation`: polygon objects only

### Storage

`LocalStorage` (`apps/api/src/sheriff_api/services/storage.py`) provides:

- per-project directory setup
- safe path resolution constrained to storage root
- byte writes and reads for uploaded assets/exports
- file and subtree deletion helpers for cleanup workflows

Upload endpoint (`/projects/{project_id}/assets/upload`) writes files as:

- `assets/{project_id}/{asset_uuid}{extension}`

Upload metadata stored in `assets.metadata_json`:

- `storage_uri`
- `original_filename`
- `relative_path`
- `size_bytes`

Additional upload-enriched fields persisted on `assets` rows:

- `width` and `height` (when dimensions can be inferred from uploaded bytes)

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
- `DELETE /api/v1/projects/{project_id}`

Project `task_type` values supported by API:

- `classification` (legacy compatibility)
- `classification_single`
- `bbox`
- `segmentation`

Categories:

- `POST /api/v1/projects/{project_id}/categories`
- `GET /api/v1/projects/{project_id}/categories`
- `PATCH /api/v1/categories/{category_id}`

Assets:

- `POST /api/v1/projects/{project_id}/assets`
- `GET /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
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

Error response contract:

- non-2xx API responses use:
  - `error.code` (stable machine-readable code)
  - `error.message` (human-readable message)
  - `error.details` (context including `request_path` and `request_method`)
- validation failures use `error.code = "validation_error"` and include `details.issues`

## 5. Web Architecture (`apps/web`)

Main workspace:

- `apps/web/src/app/page.tsx` (composition/wiring)

UI structure:

- Left: project list + hierarchical file tree
- Center: viewer canvas + adaptive pagination + skip/nav controls
- Right: label panel (label selection + manage mode + edit/submit)

### Frontend Data/State

Custom hooks:

- `useProject`: project listing
- `useAssets`: assets + annotations for active project
- `useLabels`: categories for active project
- `useImportWorkflow`: import dialog, import progress, validation state, and remembered defaults/folder-option loading
- `useDeleteWorkflow`: single/bulk/folder/project delete flows
- `useAnnotationWorkflow`: staged vs direct submit, selection state, and submit gating

Local persisted setting:

- project multi-label map in `localStorage` (`pixel-sheriff:project-multilabel:v1`)

Workspace pure helpers (`apps/web/src/lib/workspace/*`):

- `tree.*`: relative-path normalization, folder tree construction, folder chain helpers
- `pagination.*`: width-aware chip capacity and page-token window generation
- `annotationState.*`: draft vs committed selection-state comparison and submit eligibility rules
- `hotkeys.*`: keyboard shortcut parsing/routing for navigation and label selection
- `deleteState.*`: pure selection/pruning helpers for bulk and folder-scope delete flows
- `annotationSubmission.*`: payload construction helpers for single/staged annotation submit paths
- `geometry.*`: image/viewport coordinate transforms, geometry math, and hit-testing helpers
- `importDialog.*`: import form validation and default-resolution helpers for mode/project/folder destination
- `importFiles.*`: image candidate filtering and import relative-path construction helpers
- `annotationWorkflowSelection.*`: current-asset selection resolution across pending vs committed annotation state
- `classColors.*`: deterministic class-to-color mapping for label chips and geometry overlays

### Implemented UX Behaviors

- Import dialog supports:
  - existing vs new project target
  - task-type selection when creating a new project
  - optional existing folder/subfolder target
  - editable folder path
  - inline field validation hints/errors
  - remembered defaults across sessions for mode/project/folder destination
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
  - explicit staged/dirty badges for assets/folders with pending edits
  - folder/subfolder delete from tree
  - bulk delete mode (multi-select image removal within current scope)
- Viewer behavior:
  - black letterbox with image `contain`
  - bounded responsive viewport height
  - keyboard `ArrowLeft`/`ArrowRight`
  - skip controls: `-10`, `-5`, `<`, `>`, `+5`, `+10`
  - geometry overlay for bbox/polygon rendering and drawing interactions
- Pagination behavior:
  - width-adaptive page-token window
  - `First`/`Last` chips
  - labeled/unlabeled color status
  - staged/dirty indicators for pages with pending edits
- Label panel behavior:
  - manage mode for create/rename/reorder/activate/deactivate
  - project multi-label toggle only editable in manage mode
  - classification mode includes explicit `Clear Selected Labels`
  - classification mode surfaces assigned-label summary for current image
  - edit mode stages multi-asset changes
  - staged edits persist when switching assets and restore when returning to an asset with pending state
  - submit commits staged changes in batch
  - number-key shortcuts (`1..9`, top row and numpad) map to active label order
  - mode tabs switch between `labels`, `bbox`, and `segmentation`
  - mode tabs/actions are locked by project task mode
  - in bbox/seg modes, class buttons assign the selected geometry object category
  - selected geometry object can be deleted from panel or keyboard (`Delete`)
- Geometry authoring behavior:
  - bbox mode: drag to draw, click to select, drag selected box to move, drag corner/edge handles to resize, `Esc` to cancel draft
  - segmentation mode (polygon v1): click to add points, close near start-point, by double-click, or with `Enter`; `Esc` to cancel draft
  - geometry edits join existing pending/edit-mode submit workflow
  - inline draft-status warnings are shown while geometry is uncommitted
- Feedback behavior:
  - auto-dismiss toast message for success/error summaries
  - delete summaries include removed image and annotation counts

## 6. Annotation Payload Contract

Annotation payload is normalized to a backward-compatible v2 shape:

- `version: "2.0"`
- legacy-compatible classification fields:
  - `type: "classification"`
  - `category_id` (primary label)
  - `category_ids` (multi-label-compatible)
- v2 classification block:
  - `classification.category_ids`
  - `classification.primary_category_id`
- geometry object list:
  - bbox object: `{ id, kind: "bbox", category_id, bbox: [x, y, width, height] }`
  - polygon object: `{ id, kind: "polygon", category_id, segmentation: [[x1, y1, ...]] }`
- optional `image_basis` (`width`/`height`) for deterministic geometry bounds
- `coco` compatibility block with `image_id` / `category_id`

Annotation payload normalization enforces project task mode compatibility before persistence and returns stable structured error codes on mismatch.

Supported statuses:

- `unlabeled`
- `labeled`
- `skipped`
- `needs_review`
- `approved`

## 7. Test Coverage

- Web test suite uses Node's built-in test runner (`apps/web/tests/*.test.js`).
- Coverage includes:
  - import -> label -> submit workflow composition checks
  - edit-mode staged state persistence regression checks
  - helper-level regressions for hotkeys, delete flows, tree/pagination, import defaults, annotation state transitions, and geometry math helpers
- API test suite uses `pytest` with `httpx` ASGI client fixtures in `apps/api/tests`.
- API coverage includes geometry validation and COCO export geometry record assertions.

## 8. Known Gaps

- Review/QA moderation workflow not implemented
- Geometry tooling polish pending (no polygon vertex dragging/editing yet)
- Auth/multi-user permissions not implemented
- MAL routes are placeholders only
- Active bug investigation: intermittent annotation submit `404` can occur in stale project/asset submit contexts
