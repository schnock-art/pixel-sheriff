# pixel-sheriff

Annotation UI for CV with a monorepo scaffold covering API, worker, and web.

## Repository layout

- `apps/api`: FastAPI backend for projects/categories/assets/annotations/exports
- `apps/worker`: background job worker scaffolding
- `apps/web`: Next.js labeling UI scaffolding
- `packages/contracts`: OpenAPI + TS client placeholders
- `infra/db`: local Postgres initialization

## Quickstart

### 1) Environment

```bash
cp .env.example .env
```

### 2) Run with Docker

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000/api/v1`
- Web: `http://localhost:3000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

### 3) API development without Docker

```bash
cd apps/api
python -m pip install -e .[dev]
uvicorn sheriff_api.main:app --reload
```

### 4) Run tests

```bash
cd apps/api && pytest
cd ../worker && pytest
```

## Example flow (API)

1. Create project: `POST /api/v1/projects`
2. Add categories: `POST /api/v1/projects/{project_id}/categories`
3. Add assets: `POST /api/v1/projects/{project_id}/assets`
4. Add/update annotation: `POST /api/v1/projects/{project_id}/annotations`
5. Export dataset manifest: `POST /api/v1/projects/{project_id}/exports`

## Notes

- Category IDs are immutable in the API (only name/order/active state are patchable).
- Worker and web are intentionally scaffolded for incremental implementation of the roadmap.
