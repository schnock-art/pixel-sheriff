# pixel-sheriff

Local-first computer vision annotation and training platform.

## Overview

Pixel Sheriff helps you go from raw images to trained models in one workspace:

- Import images into projects and folder trees
- Create/select project tasks (`classification`, `bbox`, `segmentation`) and keep labels scoped per task
- Annotate for classification, bounding boxes, or segmentation
- Build task-scoped versioned datasets with saved split membership (`train`/`val`/`test`)
- Train experiments and monitor live metrics/logs
- Export datasets and model artifacts
- Run deployment + suggestion workflows

## Tech Stack

- `apps/web`: Next.js frontend
- `apps/api`: FastAPI backend
- `apps/trainer`: Redis worker for training/inference tasks
- `db`: PostgreSQL
- `redis`: queue/event backend
- local storage under `./data`

## Prerequisites

- Docker + Docker Compose
- Make

## Install

1. Copy environment defaults:

```bash
cp .env.example .env
```

2. Start the app:

```bash
make up
```

3. Open:

- Web: `http://localhost:3010`
- API docs: `http://localhost:8010/docs`
- API base: `http://localhost:8010/api/v1`

## How to Use

1. Create or select a project.
2. Select or create a task in `Labeling` (`Task` selector at top-right).
3. Import images (existing/new project target).
4. Label assets in `Labeling`.
5. Create a dataset version in `Dataset`.
6. Create/update a model in `Models`.
7. Start training in `Experiments` and monitor metrics/logs.
8. Export dataset zip or model artifacts as needed.

Experiment consistency note:
- Experiments default to the model's recorded `source_dataset.manifest_id`, not simply the latest active dataset version in the project.
- To train on a newer dataset version, create or refresh the model from that dataset version first.

## Useful Make Commands

Core workflow:

```bash
make help
make up
make down
make logs
make ps
```

Web/API iteration:

```bash
make build-web-api
make up-web-api
```

Trainer iteration (optimized):

```bash
make build-trainer-base      # only when torch/cuda base changes
make build-trainer           # rebuild trainer app image
make build-trainer-bootstrap # one-time base + trainer
make up-trainer
```

Build everything:

```bash
make build-all
make up-all
```

Focused tests:

```bash
make test-web
make test-api-focused
make test-api-safe
```

Direct helpers:

```bash
./scripts/run_web_tests.sh
./scripts/run_api_tests.sh
```

Contract sync:

```bash
make contracts-sync
make contracts-check
```

## Notes on Efficient Rebuilds

- Use `make up-web-api` for normal UI/API code changes.
- Use `make build-web-api` only when API/web image rebuild is needed.
- Keep trainer base separate (`make build-trainer-base`) so large CUDA/PyTorch layers are reused.

## Documentation

- Technical deep dive: `docu/TECHNICAL_REFERENCE.md`
- Architecture: `docu/Architecture.md`
- Changelog: `docu/CHANGELOG.md`
- Implementation status: `docu/IMPLEMENTATION_TASKS.md`

## Frontend Structure Notes

- Dataset route orchestration lives in `apps/web/src/lib/hooks/useDatasetPageState.ts`, with rendering split across `apps/web/src/components/workspace/dataset/*`.
- Labeling workspace orchestration stays in `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`, with task/bootstrap, suggestion state, and tree navigation split into dedicated hooks under `apps/web/src/lib/hooks/`.
- Web API access is now split under `apps/web/src/lib/api/`:
  - `client.js` for fetch/error/URI primitives
  - `types.ts` for shared request/response types
  - domain modules such as `datasets.ts`, `experiments.ts`, `deployments.ts`, `models.ts`
  - `apps/web/src/lib/api.ts` remains a compatibility barrel for existing imports
- Experiment logs are now viewed per run attempt:
  - the API/UI preserve historical `runs/{attempt}/training.log` files
  - the experiment detail page resets the visible log buffer when a new run starts and labels the current log view with the served run number
- Cancel requests are cooperative:
  - queued runs cancel immediately
  - running jobs are marked `cancel_requested` and the trainer stops at the next batch boundary
- When changing frontend structure, run both `./scripts/run_web_tests.sh` and `cd apps/web && npx tsc --noEmit`.

## Contract Artifacts

- Canonical shared schemas and generated metadata live under `packages/contracts`.
- App-local copies in `apps/api` and `apps/web` are synchronized from that directory for runtime compatibility.
- Update generated metadata and sync targets with `make contracts-sync`.
- Verify there is no artifact drift with `make contracts-check`.

## Test Environment Notes

- In WSL, prefer `./scripts/run_web_tests.sh` or `make test-web`; the wrapper avoids the Windows `npm` shim and uses `nvm` when needed.
- In this repo, API tests are most reliable through `./scripts/run_api_tests.sh`, which runs them against the Docker-backed Postgres test environment and rebuilds the `api-test` image so code/tests are current.
- For cross-boundary refactor checks, prefer `make verify-cross-boundary`.
- If `make` is unavailable in the current shell, run the underlying checks directly:
  - `python3 scripts/sync_contract_artifacts.py --check`
  - `./scripts/typecheck_web.sh`
  - `./scripts/run_web_tests.sh tests/apiClient.test.js`
  - `./scripts/run_api_tests.sh -q tests/test_cross_boundary_contracts.py`
