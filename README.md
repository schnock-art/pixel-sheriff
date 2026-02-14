# pixel-sheriff

Local-first CV annotation platform.

## Stack

- `apps/web`: Next.js labeling UI
- `apps/api`: FastAPI backend
- `apps/worker`: worker scaffold/placeholders
- Postgres + Redis via Docker Compose

## Current Features

- Project/dataset browsing with search
- Folder import for local images
- Single import dialog:
  - existing project or new project
  - editable target folder name
- Persistent asset upload/storage
- Hierarchical file tree (folders/subfolders/files)
- Viewer with aspect-ratio-preserving black letterbox
- Keyboard image navigation (`ArrowLeft` / `ArrowRight`)
- Label management:
  - create labels
  - rename/reorder/activate/deactivate
- Edit mode with staged annotation changes
- Batch submit staged annotations
- Multi-label toggle (in edit mode)
- Annotation upsert/list APIs
- COCO-style export bundle generation (manifest + annotations + images)
- One-click export download from web UI

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
- API: `http://localhost:8010/api/v1`
- API docs: `http://localhost:8010/docs`

## Useful Commands

- Start/rebuild: `docker compose up --build`
- Stop: `docker compose down`
- Logs: `docker compose logs -f web api`
- Status: `docker compose ps`

`docker compose logs ...` does not start containers.

## API Surface (Implemented)

- `GET /api/v1/health`
- `GET/POST /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `GET/POST /api/v1/projects/{project_id}/categories`
- `PATCH /api/v1/categories/{category_id}`
- `GET/POST /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `GET /api/v1/assets/{asset_id}/content`
- `GET/POST /api/v1/projects/{project_id}/annotations`
- `GET/POST /api/v1/projects/{project_id}/exports`
- `GET /api/v1/projects/{project_id}/exports/{content_hash}/download`

## Notes

- Browser currently calls API directly via `NEXT_PUBLIC_API_BASE_URL` (configured in compose).
- Next.js rewrite proxy is still configured and can be used if needed.

## Known Gaps

- Review/QA workflow
- Geometry tasks (bbox/segmentation)
- MAL beyond placeholders

## Troubleshooting

- Import `NetworkError`:
  - `curl http://localhost:8010/api/v1/health`
  - `docker compose logs --tail=200 api web`
  - `docker compose up --build web api`

- Local read errors (`NotReadableError`):
  - Source files are often cloud placeholders (e.g. OneDrive)
  - Move images to a true local folder and retry
