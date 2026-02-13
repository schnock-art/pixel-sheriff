🚀 Setup (Root Level)
docker-compose.yml

 Define services:

api (FastAPI)

web (Next.js)

db (Postgres)

redis (for queue / worker)

 Shared networks

 Bind volumes for local dev

.env.example

 PORT variables

 DB_HOST/DB_PORT/DB_USER/DB_PASS

 REDIS_URL

 STORAGE_ROOT

📌 apps/api
pyproject.toml

 FastAPI + uvicorn + SQLAlchemy + Alembic

 Pydantic models

 Async deps

 Test tools (pytest, httpx)

src/sheriff_api/main.py

 Mount routers:

health

projects

categories

assets

annotations

exports

 CORS

 Config loading

src/sheriff_api/config.py

 Load variables using pydantic

 DB URL

 Storage paths

src/sheriff_api/db/session.py

 Async SQLAlchemy engine

 SessionLocal

 Dependency for requests

src/sheriff_api/db/models.py

 Project

 Category

 Asset

 Annotation

 DatasetVersion

 Model (for MAL later)

 Suggestion (for MAL later)

src/sheriff_api/routers/*.py

Create CRUD for each:

 GET / POST / PATCH / DELETE

 Request validation via Pydantic schemas

 Pagination filters (status, project_id)

src/sheriff_api/services/storage.py

 Local filesystem abstraction

 MinIO / S3 connector

 Thumbnail generator

src/sheriff_api/services/video_frames.py

 ffmpeg wrapper

 Frame extraction jobs (later called by worker)

src/sheriff_api/services/exporter_coco.py

 Read DB

 Compile COCO JSON

 Write manifest.json

 Package zip

src/sheriff_api/schemas/*.py

 Request/response schemas

 Use unified naming conventions

 Validate category IDs immutability

🧠 apps/worker
pyproject.toml

 Redis + task queue client (e.g., RQ or celery/other)

src/sheriff_worker/main.py

 Connect to queue

 Process jobs:

extract_frames

build_export_zip

inference_suggest (MAL placeholder)

src/sheriff_worker/jobs/*.py

 Each handler has:

job input validation

progress logging

DB update hooks

src/sheriff_worker/queues/broker.py

 Redis init

 Enqueue helpers

🌐 apps/web (Next.js)
/app/(project)/

 Pages:

project list

project settings

assets grid

labeling UI

review UI

exports

/components/

 AssetGrid

 LabelPanel

 Viewer (image/frame)

 SuggestionOverlay (MAL stub)

 Filters

 Pagination

/lib/api.ts

 Generated OpenAPI TS client (place in /packages/contracts/ts-client)

 Helper fetch wrapper

/lib/hooks/*.ts

 useAssets

 useLabels

 useProject

Hotkeys

 Next / Prev

 Category assign

 Accept MAL suggestion

🧪 Testing
apps/api/tests/

 health endpoint

 CRUD endpoints for all entities

 Export manifest structure

 Asset status filters

apps/worker/tests/

 Job enqueues

 Frame extraction mock

 Export packager

apps/web/tests/

 UI snaps

 Hotkey flows

📦 packages/contracts
openapi/

 FastAPI OpenAPI schema auto-export

 Version every release

ts-client/

 Generate TypeScript client from OpenAPI

 Keep in sync on API changes (hook for codegen)

📈 Infra
infra/db/init.sql

 Base Postgres setup

 Extensions, roles

Deployment manifests

 Production docker compose (later)

 K8s manifests (future)

✨ MAL Extension Placeholder Tasks
API

 POST /models

 GET /models

 GET /assets/:id/suggestions

 POST /projects/:id/suggestions/batch

Backend Data Models

 Model table

 PredictionSuggestion table

Worker

 inference_suggest job

 model loader (ONNX runtime)

UI

 Suggestion panel

 Accept/Reject controls

 Accept hotkey

📡 Notes & Best Practices

✅ Keep related FastAPI code grouped (routers + schemas + models) — improving maintainability as the app grows.
✅ In Next.js, organize pages under app/ for filesystem routing and use route groups to organize logically.