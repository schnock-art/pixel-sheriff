# Implementation Tasks

## 🚀 Setup (Root Level)
- [x] `docker-compose.yml`
  - [x] Define services: api, web, db, redis
  - [x] Shared networks/volumes for local dev
- [x] `.env.example`
  - [x] API/Web/Postgres/Redis variables
  - [x] Storage root and CORS values

## 📌 apps/api
- [x] `pyproject.toml` with FastAPI, SQLAlchemy, test dependencies
- [x] `src/sheriff_api/main.py`
  - [x] Mount routers: health, projects, categories, assets, annotations, exports
  - [x] CORS and startup DB initialization
- [x] `src/sheriff_api/config.py` settings via pydantic
- [x] `src/sheriff_api/db/session.py` async engine/session dependency
- [x] `src/sheriff_api/db/models.py`
  - [x] Project, Category, Asset, Annotation, DatasetVersion
  - [x] MAL placeholders: Model, Suggestion
- [x] `src/sheriff_api/routers/*.py`
  - [x] CRUD-like endpoints for core entities
  - [x] Asset filtering by annotation status
  - [x] MAL placeholder endpoints
- [x] `src/sheriff_api/services/*`
  - [x] Storage abstraction
  - [x] Video frame extraction stub
  - [x] Exporter manifest/hash builder
- [x] `src/sheriff_api/schemas/*.py`
  - [x] Request/response schema definitions
  - [x] Category update does not allow ID mutation (immutable by API design)

## 🧠 apps/worker
- [x] `pyproject.toml`
- [x] `src/sheriff_worker/main.py` dispatcher for job handlers
- [x] `src/sheriff_worker/jobs/*.py`
  - [x] `extract_frames`
  - [x] `build_export_zip`
  - [x] `inference_suggest`
- [x] `src/sheriff_worker/queues/broker.py` in-memory broker abstraction

## 🌐 apps/web (Next.js scaffold)
- [x] Route scaffold under `src/app`
  - [x] Home page
  - [x] Project workspace page
- [x] Components
  - [x] AssetGrid
  - [x] LabelPanel
  - [x] Viewer
  - [x] SuggestionOverlay
  - [x] Filters
  - [x] Pagination
- [x] API helper in `src/lib/api.ts`
- [x] Hook placeholders: `useAssets`, `useLabels`, `useProject`
- [x] Hotkey behavior documented in test scaffold

## 🧪 Testing
- [x] `apps/api/tests/` health endpoint + CRUD/export flow + asset status filter
- [x] `apps/worker/tests/` job enqueue/execution coverage
- [x] `apps/web/tests/` test scaffold for hotkey flow

## 📦 packages/contracts
- [x] `openapi/` placeholder directory
- [x] `ts-client/` placeholder directory
- [x] package README

## 📈 Infra
- [x] `infra/db/init.sql` base Postgres setup/extension
- [ ] Production deployment compose
- [ ] Kubernetes manifests

## ✨ MAL Extension Placeholder Tasks
- [x] API placeholders
  - [x] `POST /models`
  - [x] `GET /models`
  - [x] `GET /assets/:id/suggestions`
  - [x] `POST /projects/:id/suggestions/batch`
- [x] Backend data models placeholders
- [x] Worker inference placeholder job
- [x] UI suggestion overlay placeholder
