# pixel-sheriff

Local-first CV annotation tool (monorepo) with:
- `apps/web` (Next.js UI)
- `apps/api` (FastAPI backend)
- `apps/worker` (scaffold)
- `db` (Postgres) + `redis` via Docker Compose

## Current State

Implemented now:
- Dataset/project list and search
- Image import from local folders
- Import destination dialog:
  - existing project or new project
  - editable target folder name
- Persistent backend storage for uploaded assets
- Hierarchical file tree (folders/subfolders/files) in UI
- Viewer with aspect-ratio-preserving letterbox (`contain` + black background)
- Keyboard navigation (`ArrowLeft` / `ArrowRight`)
- Label management:
  - add labels
  - rename/reorder/activate/deactivate labels
- Edit mode + staged annotations + batch submit
- Multi-label toggle (in edit mode)
- Annotation upsert/list and export manifest record endpoints

Not implemented yet:
- Export zip generation/download
- Review workflow
- Geometry tools (bbox/polygon)
- MAL inference integration

## Repo Layout

- `apps/api`: FastAPI service + SQLAlchemy models/routers/services
- `apps/web`: Next.js app router UI
- `apps/worker`: worker/job placeholders
- `infra/db`: DB init SQL
- `data`: local storage mount for assets/exports

## Run Locally

1. Copy env file:

```bash
cp .env.example .env
```

2. Start stack:

```bash
docker compose up --build
```

3. Open (default in current `.env.example`):
- Web: `http://localhost:3010`
- API: `http://localhost:8010/api/v1`
- API docs: `http://localhost:8010/docs`

## Commands

- Start/rebuild: `docker compose up --build`
- Stop: `docker compose down`
- Logs: `docker compose logs -f web api`
- Status: `docker compose ps`

Note: `docker compose logs ...` does not start services.

## API Endpoints In Use

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

## Troubleshooting

- Import fails with `NetworkError`:
  - Check API health: `curl http://localhost:8010/api/v1/health`
  - Check logs: `docker compose logs --tail=200 api web`
  - Rebuild web+api: `docker compose up --build web api`

- Local file read errors (`NotReadableError`):
  - Usually source files are cloud placeholders (e.g. OneDrive not fully local)
  - Move files to a plain local folder and retry

- Old UI after code changes:
  - `docker compose up --build web`
  - hard refresh browser
