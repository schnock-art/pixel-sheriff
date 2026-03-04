# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the stack
```bash
make up              # Start all services (docker compose up -d)
make down            # Stop all services
make logs            # Tail web, api, trainer logs
make ps              # Show service status
```

### Iterating on web + API (fast path)
```bash
make build-web-api   # Rebuild web + api images
make up-web-api      # Start only web + api + deps
```

### Trainer (CUDA/PyTorch base is expensive)
```bash
make build-trainer-bootstrap  # One-time: build base then trainer
make build-trainer            # Fast rebuild (app layer only)
make up-trainer               # Start trainer service
```

### Tests
```bash
make test-web          # Next.js tests (apps/web/tests/datasetPage.test.js)
make test-api-focused  # Pytest subset against isolated test DB (:5433)
make test-api-safe     # Same + DB URL safety guard

# Run a single pytest test by name
cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test \
  STORAGE_ROOT=/tmp/pixel_sheriff_test_data \
  python3 -m pytest -s tests/test_api.py -k "<test_name>"
```

API tests use an isolated database (`pixel_sheriff_test` on port `5433`) and `/tmp/pixel_sheriff_test_data` for storage — never the live DB.

## Architecture

### Service topology

```
Web (Next.js :3010) ──▶ API (FastAPI :8010) ──▶ PostgreSQL + Redis
                                    │
                                    └──▶ Trainer (:8020 inference only)
                                         └── Redis worker (training jobs)
```

### Apps

**`apps/api/`** — FastAPI backend
- `src/sheriff_api/main.py` — App entry, router mounting
- `src/sheriff_api/db/models.py` — SQLAlchemy async ORM (Project, Task, Category, Asset, Annotation)
- `src/sheriff_api/routers/` — One file per resource; all routes under `/api/v1`
- `src/sheriff_api/services/` — File-backed stores (`dataset_store.py`, `experiment_store.py`, `model_store.py`) that persist records as JSON under `./data/`
- `src/sheriff_api/ml/` — Model factory, backbone registry, ONNX adapter metadata

**`apps/web/`** — Next.js 14 App Router frontend
- `src/app/projects/[projectId]/` — All project routes (labeling workspace, datasets, models, experiments, deploy)
- `src/components/workspace/` — Labeling UI orchestrator (`ProjectAssetsWorkspace.tsx`) and sub-components
- `src/components/Viewer.tsx` — Image canvas with annotation overlays
- `src/lib/` — Hooks, workspace state, and Zod/AJV schemas

**`apps/trainer/`** — Dual-loop Python service
- `src/pixel_sheriff_trainer/main.py` — Starts two concurrent loops: a Redis job worker (training) and a FastAPI inference server (port 8020)
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

- **Task**: scoped annotation schema within a project (kind: `classification` | `bbox` | `segmentation`). Projects can have multiple tasks; each asset has one annotation record per task.
- **DatasetVersion**: immutable once created. Class order, asset membership, and splits are frozen at creation time.
- **Experiment**: a training run referencing a model config + dataset version. Redis carries the job; the trainer worker picks it up and streams events/metrics as `.jsonl` files; the API tails these for SSE.
- **Deployment**: links a trained ONNX model to the inference server. The trainer's inference loop loads the model on warmup and serves `/predict` requests for model-assisted labeling (MAL).

### Environment

Copy `.env.example` to `.env`. Key vars:
- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8010` — browser-side API URL
- `INTERNAL_API_BASE_URL=http://api:8000` — server-side (SSR) API URL
- `STORAGE_ROOT=./data` — local artifact root (mounted into all containers)
- `TRAINER_PYTORCH_INDEX_URL` — set to CUDA wheel index for GPU support
