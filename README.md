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

## Stack

- `apps/web`: Next.js labeling UI
- `apps/api`: FastAPI backend
- `apps/worker`: worker scaffold/placeholders
- Postgres + Redis via Docker Compose
- Local filesystem storage under `./data`

## Current Features

- Dataset/project browsing with search
- Local folder import with one modal:
  - import into existing or new project
  - optional existing folder/subfolder target for existing projects
  - editable destination folder name
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
- Hierarchical file tree (folders/subfolders/files), with:
  - preserved hierarchy ordering
  - per-folder expand/collapse
  - collapse all / expand all
  - folder-scope filtering (review a subtree only)
  - labeled/unlabeled status dots on folders and files
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
- Labels:
  - create labels
  - manage labels (rename/reorder/activate/deactivate)
  - project-scoped multi-label toggle (editable only in Manage Labels mode)
- Annotation flow:
  - edit mode staging
  - batch submit staged annotations
  - direct single-submit path when not staging
  - status values: `unlabeled`, `labeled`, `skipped`, `needs_review`, `approved`
  - keyboard label selection by class index (`1..9`, top row and numpad)
- COCO-style export:
  - export record creation/listing
  - deterministic zip artifact with `manifest.json`, `annotations.json`, and `images/`
  - one-click download from web UI

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
- `GET/POST /api/v1/models` (placeholder MAL surface)
- `GET /api/v1/assets/{asset_id}/suggestions` (placeholder)
- `POST /api/v1/projects/{project_id}/suggestions/batch` (placeholder)

## Runtime Notes

- Web uses `NEXT_PUBLIC_API_BASE_URL` for browser calls.
- Next.js rewrite proxy (`/api/v1/* -> INTERNAL_API_BASE_URL`) remains available.
- Default local ports are conflict-resistant and configurable in `.env`.

## Known Gaps

- Review/QA workflow
- Bounding boxes and segmentation tools
- MAL implementation beyond placeholder endpoints
- Shared-asset reference mode (upload-once/link-many)
- Refactor/maintainability workstream in progress (tracked in `IMPLEMENTATION_TASKS.md`):
  - API integrity hardening
  - workspace page decomposition
  - expanded regression testing

## Troubleshooting

- Import `NetworkError`:
  - `curl http://localhost:8010/api/v1/health`
  - `docker compose logs --tail=200 api web`
  - `docker compose up --build web api`

- Local read failures (`AbortError`/`NotReadableError`):
  - Files are often cloud placeholders (for example OneDrive "online-only")
  - Move/sync images to a true local directory and re-import
