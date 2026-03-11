# pixel-sheriff

Local-first labeling, dataset, training, deployment, and AI-prelabel platform for images plus frame-based video and webcam workflows.

## What It Does

Pixel Sheriff is image-first even when the source is a video file or live camera:

- images are normal assets
- video imports extract frames into a sequence folder
- webcam capture uploads frames into a live sequence folder
- all labeling happens in the same asset workspace
- datasets and exports remain frame-based

Current task modes:
- `classification`
- `bbox`
- `segmentation`

Current AI-assisted workflow:
- deployment-backed labeling suggestions for the active asset
- sequence-first AI prelabels for bbox tasks
- pending prelabels stored separately from normal annotations until accepted or edited

## Current Product Model

```text
image files
-> assets
-> labels / boxes / polygons
-> dataset versions
-> models
-> experiments
-> deployments

video file / webcam stream
-> extracted or captured frames
-> asset sequence + folder
-> same labeling workspace
-> same dataset/export flow
```

## Main Features

- project-scoped shell with task selector and task-aware routes
- folder tree plus searchable asset browser
- image import, video import, and webcam capture
- sequence navigation for video/webcam frames
- classification, bbox, and polygon annotation tools
- staged edit workflow plus direct submit
- immutable dataset versions with saved split membership
- export zip generation with `manifest.json` and `coco_instances.json`
- project-scoped models, experiments, and deployments
- active-deployment suggestions in labeling
- bbox prelabel sessions for video and webcam review flows

## Runtime Topology

Default services:

- `apps/web`: Next.js frontend
- `apps/api`: FastAPI backend
- `apps/worker`: Redis-backed media and prelabel worker
- `apps/trainer`: training plus inference service
- `db`: PostgreSQL
- `redis`: queues

High-level flow:

```text
browser
-> web
-> api
   -> postgres
   -> redis -> worker
   -> trainer
```

Important behavior:

- `make up` starts the full stack
- `make up-web-api` is a lighter loop and does not start the worker or trainer
- video extraction requires the worker
- training, deployment inference, and Florence prelabels require the trainer

## Quickstart

### Prerequisites

- Docker + Docker Compose
- Make

Optional for local non-Docker iteration:

- Node.js
- Python 3.11+

### Start

1. Copy environment defaults:

```bash
cp .env.example .env
```

2. Start the full stack:

```bash
make up
```

3. Open:

- Web: `http://localhost:3010`
- API docs: `http://localhost:8010/docs`
- API base: `http://localhost:8010/api/v1`

## Typical Workflow

1. Create or select a project.
2. Select or create a task from the ribbon.
3. Use `Import` for images, `Video File` for extraction, or `Webcam Stream` for live capture.
4. Label assets in the main workspace.
5. For sequence-backed assets, use the timeline, thumbnails, and frame controls.
6. For bbox tasks, optionally enable AI prelabels during video import or webcam capture.
7. Review pending AI prelabels in the dedicated workspace panel and promote accepted or edited proposals into annotations.
8. Create a dataset version.
9. Create or edit a model.
10. Launch and monitor experiments.
11. Deploy a completed experiment and use deployment suggestions in labeling.

## AI Prelabels

Implemented today:

- bbox-only
- sources:
  - `active_deployment`
  - `florence2`
- video:
  - session created from `prelabel_config`
  - jobs auto-start after frame extraction
- webcam:
  - live session created at capture start
  - sampled frames enqueue while capture is running
  - modal finish closes input for the live session

Review behavior:

- pending proposals stay out of normal annotations
- `Accept` merges them into the asset annotation payload
- `Edit selected` loads a proposal into the normal bbox draft with provenance
- saved provenance-backed objects mark proposals as `accepted` or `edited`

## Storage Model

Database state:

- projects
- tasks
- categories
- folders
- asset sequences
- assets
- annotations
- prelabel sessions
- prelabel proposals
- suggestions

File-backed storage under `./data`:

- uploaded assets
- imported videos
- dataset/version records
- export zips
- model records
- experiment artifacts

## Useful Commands

Full stack:

```bash
make help
make up
make down
make logs
make ps
```

Fast web/API loop:

```bash
make build-web-api
make up-web-api
docker compose up -d worker
make up-trainer
```

Trainer iteration:

```bash
make build-trainer-base
make build-trainer
make build-trainer-bootstrap
make up-trainer
```

Local app iteration:

```bash
make infra
make create-local-db
make dev-api
make dev-web
```

Checks:

```bash
make test-web
./scripts/run_api_tests.sh
make verify-cross-boundary
make contracts-sync
make contracts-check
```

## Documentation

Current docs:

- `docs/architecture.md`
- `docu/Architecture.md`
- `docu/TECHNICAL_REFERENCE.md`
- `docu/Roadplan.md`
- `docu/IMPLEMENTATION_TASKS.md`
- `docu/VLM_COLD_START_PRELABELING_TASKS.md`
- `docu/CHANGELOG.md`

Other folders:

- `docs/demo/` contains generated README/demo media
- `docs/plans/` contains dated design notes and plan snapshots

## Codebase Map

Frontend:

- `apps/web/src/app/projects/[projectId]/`
- `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`
- `apps/web/src/components/workspace/project-assets/`
- `apps/web/src/lib/hooks/`
- `apps/web/src/lib/api/`

Backend:

- `apps/api/src/sheriff_api/main.py`
- `apps/api/src/sheriff_api/db/models.py`
- `apps/api/src/sheriff_api/routers/`
- `apps/api/src/sheriff_api/services/`

Worker and trainer:

- `apps/worker/src/sheriff_worker/main.py`
- `apps/worker/src/sheriff_worker/jobs/`
- `apps/trainer/src/pixel_sheriff_trainer/`

Shared contracts:

- `packages/contracts`

## Notes

- Sequence frames export as normal images with lineage metadata.
- Historical notes in `docu/` used to lag the codebase; they have been refreshed, but dated planning files under `docs/plans/` remain historical snapshots.
