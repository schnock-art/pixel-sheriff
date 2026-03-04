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

## Notes on Efficient Rebuilds

- Use `make up-web-api` for normal UI/API code changes.
- Use `make build-web-api` only when API/web image rebuild is needed.
- Keep trainer base separate (`make build-trainer-base`) so large CUDA/PyTorch layers are reused.

## Documentation

- Technical deep dive: `docu/TECHNICAL_REFERENCE.md`
- Architecture: `docu/Architecture.md`
- Changelog: `docu/CHANGELOG.md`
- Implementation status: `docu/IMPLEMENTATION_TASKS.md`
