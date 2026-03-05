# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the full stack (Docker)
```bash
make up              # Start all services (docker compose up -d)
make down            # Stop all services
make logs            # Tail web, api, trainer logs
make ps              # Show service status
```

### Iterating on web + API (Docker, fast path)
```bash
make build-web-api   # Rebuild web + api images
make up-web-api      # Start only web + api + deps
```

### Trainer (CUDA/PyTorch base is expensive to rebuild)
```bash
make build-trainer-bootstrap  # One-time: build base then trainer
make build-trainer            # Fast rebuild (app layer only)
make up-trainer               # Start trainer service
```

### Tests

API tests run the pytest process on the host, connecting to the Dockerized Postgres. The venv at `apps/api/.venv` must exist (see local dev setup below). Infra must be running (`make infra`).

```bash
make test-api-focused  # Run the 3 focused dataset/split tests
make test-api-safe     # Same + DB URL safety guard
make test-web          # Next.js tests
```

Run a single test by name:
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test \
STORAGE_ROOT=/tmp/pixel_sheriff_test_data \
apps/api/.venv/Scripts/python -m pytest -s apps/api/tests/test_api.py -k "<test_name>"
```

Key facts about the test suite:
- Test DB is `pixel_sheriff_test` on the same Docker Postgres (port `5433` from `.env`)
- `conftest.py` drops and recreates all tables before each test — fully isolated
- API endpoints require `task_id` (UUID from `project["default_task_id"]`), not the old `"task": "classification"` string
- Annotation creation also requires `task_id`

### Local dev without Docker (for fast iteration)

Use this to iterate on API/web changes without rebuilding Docker images. Runs against `pixel_sheriff_local` — completely isolated from the user's live `pixel_sheriff` data.

**One-time setup:**
```bash
# Create the venv and install dependencies
cd apps/api && py -3.13 -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
cd apps/web && npm install

# Start infra and create the isolated local DB
make infra
make create-local-db
```

**Then in two terminals:**
```bash
make dev-api   # FastAPI on :8000, hot reload, DB=pixel_sheriff_local, STORAGE=./data_local
make dev-web   # Next.js on :3000, hot reload, points at localhost:8000
```

Never use `DB_NAME=pixel_sheriff` with `dev-api` — that is the user's live data.

## Architecture

### Service topology

Ports are configured in `.env` (current values: web=3010, api=8010, postgres=5433, redis=6380).

```
Web (Next.js :3010) ──▶ API (FastAPI :8010) ──▶ PostgreSQL + Redis
                                    │
                                    └──▶ Trainer (:8020 inference only)
                                         └── Redis worker (training jobs)
```

### Apps

**`apps/api/`** — FastAPI backend
- `src/sheriff_api/main.py` — App entry, router mounting, lifespan runs `create_all` + `run_startup_migrations` on every start (schema is self-bootstrapping, no manual migration needed)
- `src/sheriff_api/db/models.py` — SQLAlchemy async ORM (Project, Task, Category, Asset, Annotation)
- `src/sheriff_api/routers/` — One file per resource; all routes under `/api/v1`
- `src/sheriff_api/services/` — File-backed stores (`dataset_store.py`, `experiment_store.py`, `model_store.py`) that persist records as JSON under `./data/`
- `src/sheriff_api/services/migrations.py` — Custom startup migration system; tracks applied versions in `schema_migrations` table; idempotent and safe to run multiple times
- `src/sheriff_api/ml/` — Model factory, backbone registry, ONNX adapter metadata

**`apps/web/`** — Next.js 14 App Router frontend
- `src/app/projects/[projectId]/` — All project routes (labeling workspace, datasets, models, experiments, deploy)
- `src/components/workspace/` — Labeling UI orchestrator (`ProjectAssetsWorkspace.tsx`) and sub-components
- `src/components/Viewer.tsx` — Image canvas with annotation overlays
- `src/lib/` — Hooks, workspace state, and Zod/AJV schemas

**`apps/trainer/`** — Dual-loop Python service
- `src/pixel_sheriff_trainer/main.py` — Two concurrent loops: Redis job worker (training) + FastAPI inference server (port 8020)
- `src/pixel_sheriff_trainer/runner.py` — Orchestrates a full training run
- `src/pixel_sheriff_trainer/classification/` — PyTorch dataset, train loop, eval
- `src/pixel_sheriff_trainer/io/` — Writes artifacts (events.jsonl, metrics.jsonl, checkpoints, evaluation) to `./data/experiments/`

### Data model

The DB holds mutable state (projects, tasks, categories, assets, annotations). Everything immutable (dataset versions, experiment runs, model records) is file-backed JSON under `./data/`:

```
data/
├── models/{project_id}/records.json
├── experiments/{project_id}/records.json
│   └── {experiment_id}/runs/{attempt}/  ← training artifacts
└── exports/{project_id}/               ← zip bundles
```

Dataset versions are immutable snapshots containing a COCO-format manifest plus frozen class order and train/val/test split assignments.

### Key concepts

- **Task**: scoped annotation schema within a project (kind: `classification` | `bbox` | `segmentation`). Projects can have multiple tasks; each asset has one annotation record per task. The `default_task_id` on a project is set automatically on creation.
- **DatasetVersion**: immutable once created. Class order, asset membership, and splits are frozen at creation time. Created via `POST /projects/{id}/datasets/versions` with a `task_id`.
- **Experiment**: a training run referencing a model config + dataset version. Redis carries the job; the trainer worker picks it up and streams events/metrics as `.jsonl` files; the API tails these for SSE.
- **Deployment**: links a trained ONNX model to the inference server. The trainer's inference loop loads the model on warmup and serves `/predict` requests for model-assisted labeling (MAL).

### Environment

Copy `.env.example` to `.env`. The Makefile reads `.env` via `-include .env` so all port variables (`POSTGRES_PORT`, `REDIS_PORT`, etc.) are picked up automatically. Key vars:
- `NEXT_PUBLIC_API_BASE_URL` — browser-side API URL
- `INTERNAL_API_BASE_URL=http://api:8000` — server-side (SSR) API URL inside Docker
- `STORAGE_ROOT=./data` — local artifact root (mounted into all containers)
- `TRAINER_PYTORCH_INDEX_URL` — set to CUDA wheel index for GPU support

### Annotation payload schema

Stored in `Annotation.payload_json` (v2.0):
- **classification**: `{category_ids: [UUID], primary_category_id: UUID|None}`
- **objects**: `[{id, kind:"bbox"|"polygon", category_id, bbox:[x,y,w,h]|segmentation:[[x,y,...]]}]`
- **image_basis**: `{width, height}` — used for bounds validation
- **source**: `"web-ui"` | `"api"`

Validated in: `apps/api/src/sheriff_api/services/annotation_payload.py`

### Dataset export format

Export zip contents (created by `dataset_export_pipeline.py`):
- `assets/` — image files
- `manifest.json` — v1.2: tasks, label_schema, splits, training_defaults, stats
- `coco_instances.json` — COCO format with bbox (detection) + polygons (segmentation)

Trainer classification reads `manifest.json`.
Trainer detection/segmentation reads `coco_instances.json` (already has all geometry).

### ML adapter registry

- `apps/api/src/sheriff_api/ml/registry.py` — `FAMILY_REGISTRY` maps family name → `build_fn`
- `apps/api/src/sheriff_api/ml/adapters/` — `FamilyAdapter` base + concrete adapters
- Existing: `resnet_classifier` (classification), `retinanet` (detection), `deeplabv3` (segmentation)
- `packages/pixel_sheriff_ml/src/pixel_sheriff_ml/model_factory.py` — lightweight `build_resnet_classifier` used by trainer

### TaskPipeline Protocol + PIPELINE_REGISTRY

- `apps/trainer/src/pixel_sheriff_trainer/pipeline.py` — `TaskPipeline` Protocol + `PIPELINE_REGISTRY` dict
- `PIPELINE_REGISTRY` maps task kind string → pipeline instance
- Registered: `"classification"` → `ClassificationPipeline`, `"bbox"` → `DetectionPipeline`, `"segmentation"` → `SegmentationPipeline`
- `runner.py` dispatches via `PIPELINE_REGISTRY[job.task]`

### How to add a new task type

1. Create `apps/trainer/src/pixel_sheriff_trainer/{task}/`
   - `pipeline.py` → implements `TaskPipeline` Protocol
   - `dataset.py` → parses `coco_instances.json` for task-specific geometry
   - `train.py` → training loop
   - `eval.py` → task metrics
2. Register in `apps/trainer/src/pixel_sheriff_trainer/pipeline.py`:
   `PIPELINE_REGISTRY["{task}"] = MyPipeline()`
3. Add family to `apps/api/src/sheriff_api/ml/registry.py`
4. Add inference endpoint to `apps/trainer/src/pixel_sheriff_trainer/inference/app.py`
5. Update `apps/api/src/sheriff_api/services/inference_client.py` to route by `task.kind`
