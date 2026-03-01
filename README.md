# pixel-sheriff

Local-first CV annotation platform.

## What Changed Since Last Milestone

- Import UX was unified into one dialog with existing/new project targets plus optional existing folder/subfolder destination.
- Import now shows live progress (counts, bytes, speed, ETA) and clearer per-file failure diagnostics.
- Viewer/navigation improved with bounded responsive viewport height and skip controls (`-10`, `-5`, `<`, `>`, `+5`, `+10`).
- Pagination now adapts to available width and supports `First`/`Last` quick jumps.
- Export flow is fully wired from UI to API zip generation and direct download.
- Delete workflows were added:
  - project delete
  - single-image delete
  - multi-image delete
  - folder/subfolder delete (tree subtree scope)
- Delete actions now show confirmation summaries in an auto-dismiss toast with counts.
- Keyboard labeling shortcuts now support number keys (`1..9`) including numpad keys.
- Workspace internals were refactored:
  - root route now redirects to project-scoped routes
  - datasets workspace moved to `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`
  - project shell layout now handles project selector, top tabs, and project status bar
  - annotation, import, and delete workflows remain in dedicated hooks
  - tree/pagination/annotation-state logic moved into pure workspace helpers with unit tests
- Review-state visibility improved with explicit staged/dirty indicators in tree rows and pagination chips.
- Import dialog UX was upgraded with inline validation hints and remembered defaults for mode/project/folder destination.
- Testing coverage expanded with integration/regression suites for:
  - import -> label -> submit workflow
  - edit-mode staged state persistence across asset switches
- API upload now derives image `width`/`height` when detectable from uploaded bytes.
- API error responses now use a structured shape (`error.code`, `error.message`, `error.details`) for better UI diagnostics.
- Bounding box and polygon segmentation annotation tools are now wired end-to-end (draw/select/delete/submit/export).
- COCO export now includes geometry records (`bbox`, `segmentation`, `area`) for object annotations.
- Project-level task mode is now enforced (`classification_single`, `bbox`, `segmentation`) and selected during new-project import.
- Labels and geometry overlays now use deterministic class-based colors.
- Bounding-box interaction now supports move + resize (corner and edge-midpoint handles), plus inline draft warnings.
- Polygon closing is now more forgiving (`near-start`, double-click, or `Enter`) with draft-status guidance.
- Classification mode now includes explicit "Clear Selected Labels" and an assigned-label summary line.
- Export contract upgraded to v1.2:
  - `manifest.json` now includes explicit `tasks`, `label_schema`, `splits`, `training_defaults`, and `stats`
  - COCO companion file is now `coco_instances.json`
  - COCO and manifest now use the same canonical UUID asset IDs (`image_id == asset_id`)
  - class names are normalized to lowercase slug in model-facing/export fields
  - detection/segmentation exports now support explicit negative-image policy (`include_negative_images`)
  - detection COCO annotations now omit `segmentation` instead of emitting empty lists
- Project-scoped Phase 1 UI refactor is now implemented:
  - route structure under `/projects/{project_id}/...`
  - top navigation tabs: `Datasets`, `Models`, `Experiments`, `Deploy` (disabled placeholder)
  - project selector dropdown with `+ Create Project` modal
  - datasets `Build Model` CTA creates a project-scoped model draft from latest manifest and opens model detail
  - unsaved draft guard for project/tab/model navigation
- Model Builder v1.0 edit flow is now implemented:
  - editable steps for Input, Backbone, Outputs (embedding aux), and Export
  - AJV-based client-side schema validation on every draft edit
  - save gating by `isDirty && isValid`
  - `PUT /api/v1/projects/{project_id}/models/{model_id}` persists schema-valid config updates

## Stack

- `apps/web`: Next.js labeling UI
- `apps/api`: FastAPI backend
- `apps/worker`: worker scaffold/placeholders
- Postgres + Redis via Docker Compose
- Local filesystem storage under `./data`

## Current Features

- Project-scoped workspace shell with route-based navigation:
  - `/` redirects to `/projects`
  - `/projects` resolves to the last/first project or shows create-project empty state
  - project routes:
    - `/projects/{project_id}/datasets`
    - `/projects/{project_id}/models`
    - `/projects/{project_id}/models/new`
    - `/projects/{project_id}/models/{model_id}`
    - `/projects/{project_id}/experiments`
    - `/projects/{project_id}/experiments/{experiment_id}`
  - top shell project selector + tabs (`Datasets`, `Models`, `Experiments`, disabled `Deploy`)
  - workspace status line (`images labeled`, `classes`, `models`, `experiments`)
- Models pages support project-scoped create/list/detail plus editable model config drafting/saving
- Model detail page includes:
  - editable controls for Input, Backbone, Outputs (embedding aux), and Export
  - live model summary updates as draft fields change
  - AJV (`ajv` + `ajv-formats`) validation against `ModelConfig v1.0`
  - Save enablement only when draft has changes and passes schema validation
  - unsaved changes indicator + guarded navigation integration
- Experiments pages remain placeholders (no project-scoped training backend yet)
- Local folder import with one modal:
  - import into existing or new project
  - select task mode when creating a new project (`classification_single`, `bbox`, `segmentation`)
  - optional existing folder/subfolder target for existing projects
  - editable destination folder name
  - inline validation hints/errors for project and folder fields
  - remembered import defaults across sessions (mode + target project + per-project folder preference)
- Import progress UI with:
  - completed/total count
  - uploaded/failed/remaining counts
  - bytes processed
  - throughput + ETA
- Robust import diagnostics:
  - extension fallback when MIME is missing
  - per-file read/network/API failure details
- Automatic asset/tree refresh after import (no page reload)
- Persistent upload storage + streamed image serving from API
- Upload metadata enrichment on API:
  - persisted `width`/`height` when image dimensions can be inferred
  - preserved `relative_path` (or filename fallback) for tree/export consistency
- Structured API errors with machine-readable codes:
  - response envelope: `error.code`, `error.message`, `error.details`
  - validation failures include per-field issues
- Hierarchical file tree (folders/subfolders/files), with:
  - preserved hierarchy ordering
  - per-folder expand/collapse
  - collapse all / expand all
  - folder-scope filtering (review a subtree only)
  - labeled/unlabeled status dots on folders and files
  - explicit staged/dirty badges for pending annotation edits
  - folder/subfolder delete actions
  - bulk delete mode with in-scope multi-select
- Viewer:
  - aspect-ratio-preserving black letterbox (`object-fit: contain`)
  - responsive bounded viewport height
  - keyboard navigation (`ArrowLeft` / `ArrowRight`)
  - skip navigation controls (`-10`, `-5`, `<`, `>`, `+5`, `+10`)
- Adaptive pagination:
  - dynamically sized page-chip window based on available width
  - `First` / `Last` chips
  - labeled/unlabeled chip coloring
  - staged/dirty page badges for pending edits
- Labels:
  - create labels
  - manage labels (rename/reorder/activate/deactivate)
  - project-scoped multi-label toggle (editable only in Manage Labels mode)
  - deterministic class-based colors for label rows/chips
  - clear selected labels action in classification mode
  - assigned-label summary (`Assigned: ...`) for immediate visual confirmation
- Annotation flow:
  - edit mode staging
  - staged edits persist while navigating between assets until submitted or reset
  - batch submit staged annotations
  - direct single-submit path when not staging
  - unsaved-draft leave guard on project/tab/model navigation
  - status values: `unlabeled`, `labeled`, `skipped`, `needs_review`, `approved`
  - keyboard label selection by class index (`1..9`, top row and numpad)
- Geometry tools:
  - Bounding box mode: draw by drag, select existing boxes, move by drag, resize via 8 handles (corners + edge midpoints), delete selected (`Delete`), cancel draft (`Esc`)
  - Segmentation mode (polygon v1): click to add vertices, close polygon near start-point, by double-click, or with `Enter`; delete selected and cancel draft (`Esc`)
  - project task mode locks available annotation tabs/actions
  - Geometry class assignment uses the same project label set
  - Geometry edits participate in the same pending/edit-mode submit workflow as classification edits
  - Inline draft warnings clarify when geometry is not committed yet
- Dataset export (manifest v1.2 + COCO companion):
  - export record creation/listing
  - deterministic zip artifact with `manifest.json`, `coco_instances.json`, and `assets/`
  - one-click download from web UI
  - canonical join key is UUID `asset_id` across manifest and COCO
  - classification exports keep `coco_instances.json` with empty `annotations`
  - detection/segmentation exports include geometry records with validated `bbox`/`segmentation` and computed `area`
  - configurable detection/segmentation negative-image policy through `selection_criteria_json.include_negative_images` (`true` by default)

## Run Locally

1. Copy env:

```bash
cp .env.example .env
```

2. Start:

```bash
docker compose up --build
```

3. Open (`.env.example` defaults):

- Web: `http://localhost:3010`
- API base: `http://localhost:8010/api/v1`
- API docs: `http://localhost:8010/docs`

## Useful Commands

- Start/rebuild: `docker compose up --build`
- Stop: `docker compose down`
- Logs: `docker compose logs -f web api`
- Status: `docker compose ps`
- Web tests: `cd apps/web && npm test`
- Web build check: `cd apps/web && npm run build`
- API tests (container):
  - `docker compose cp apps/api/tests api:/app/tests`
  - `docker compose exec api python -m pytest /app/tests -q`

`docker compose logs ...` only reads logs; it does not start containers.

## API Surface (Implemented)

- `GET /api/v1/health`
- `GET/POST /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `DELETE /api/v1/projects/{project_id}`
- `GET/POST /api/v1/projects/{project_id}/categories`
- `PATCH /api/v1/categories/{category_id}`
- `GET/POST /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
- `GET /api/v1/assets/{asset_id}/content`
- `GET/POST /api/v1/projects/{project_id}/annotations`
- `GET/POST /api/v1/projects/{project_id}/exports`
- `GET /api/v1/projects/{project_id}/exports/{content_hash}/download`
- `GET/POST /api/v1/projects/{project_id}/models`
- `GET /api/v1/projects/{project_id}/models/{model_id}`
- `PUT /api/v1/projects/{project_id}/models/{model_id}`
- `GET/POST /api/v1/models` (placeholder MAL surface)
- `GET /api/v1/assets/{asset_id}/suggestions` (placeholder)
- `POST /api/v1/projects/{project_id}/suggestions/batch` (placeholder)

## Runtime Notes

- Web uses `NEXT_PUBLIC_API_BASE_URL` for browser calls.
- Next.js rewrite proxy (`/api/v1/* -> INTERNAL_API_BASE_URL`) remains available.
- Default local ports are conflict-resistant and configurable in `.env`.

## Known Gaps

- Review/QA workflow
- Geometry tooling polish (no polygon vertex dragging/edit yet)
- Experiment/Deploy sections are still UI placeholders in the project shell
- Model training execution is not implemented yet (`Train Model` remains disabled)
- MAL implementation beyond placeholder endpoints
- Shared-asset reference mode (upload-once/link-many)

## Active Known Issue

- Intermittent annotation submit `404` can still occur in stale project/asset contexts (for example after project/asset churn while staged state exists). Investigation is ongoing.

## Troubleshooting

- Import `NetworkError`:
  - `curl http://localhost:8010/api/v1/health`
  - `docker compose logs --tail=200 api web`
  - `docker compose up --build web api`

- Local read failures (`AbortError`/`NotReadableError`):
  - Files are often cloud placeholders (for example OneDrive "online-only")
  - Move/sync images to a true local directory and re-import

- Annotation submit `404`:
  - Usually means the selected project/asset pair no longer matches server state.
  - Refresh the page, reselect the project, and verify the asset is still present in the tree before submitting again.
