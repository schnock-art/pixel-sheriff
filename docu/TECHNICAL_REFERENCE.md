# Technical Reference

Current implementation-oriented reference for developers working inside this repo.

## Most Used Commands

Full stack:

```bash
make up
make down
make logs
make ps
```

Web and API loop:

```bash
make build-web-api
make up-web-api
docker compose up -d worker
make up-trainer
```

Checks:

```bash
make test-web
./scripts/run_api_tests.sh
make verify-cross-boundary
make contracts-sync
make contracts-check
```

## Current Stack

- frontend: Next.js 14
- API: FastAPI + SQLAlchemy async
- worker: Redis-backed async job consumer
- trainer: training + inference app
- database: PostgreSQL
- queue/event backend: Redis
- storage: local filesystem under `./data`

## Current Route and Module Shape

### Frontend

Main project-scoped routes:

- `/projects/[projectId]/page.tsx`
- `/projects/[projectId]/dataset/page.tsx`
- `/projects/[projectId]/models/page.tsx`
- `/projects/[projectId]/models/new/page.tsx`
- `/projects/[projectId]/models/[modelId]/page.tsx`
- `/projects/[projectId]/experiments/page.tsx`
- `/projects/[projectId]/experiments/new/page.tsx`
- `/projects/[projectId]/experiments/[experimentId]/page.tsx`
- `/projects/[projectId]/deploy/page.tsx`

Main labeling workspace:

- `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`

Current project-assets subcomponents include:

- `AssetBrowser.tsx`
- `VideoImportModal.tsx`
- `WebcamCaptureModal.tsx`
- `PrelabelSettingsSection.tsx`
- `AiPrelabelsPanel.tsx`
- `SequenceToolbar.tsx`
- `SequenceTimeline.tsx`
- `SequenceThumbnailStrip.tsx`

Important hooks:

- `useAssets`
- `useFolders`
- `useSequence`
- `useSequenceNavigation`
- `useAnnotationWorkflow`
- `useWorkspaceSuggestions`
- `useWebcamCapture`
- `usePrelabels`

API client layout:

- low-level primitives in `apps/web/src/lib/api/client.js`
- shared types in `apps/web/src/lib/api/types.ts`
- domain clients in `apps/web/src/lib/api/*.ts`
- compatibility barrel in `apps/web/src/lib/api.ts`

### API

Entrypoint:

- `apps/api/src/sheriff_api/main.py`

Important routers:

- `routers/projects.py`
- `routers/tasks.py`
- `routers/categories.py`
- `routers/assets.py`
- `routers/folders.py`
- `routers/video_imports.py`
- `routers/sequences.py`
- `routers/prelabels.py`
- `routers/annotations.py`
- `routers/datasets.py`
- `routers/models.py`
- `routers/experiments/`
- `routers/deployments.py`

Important services:

- `services/sequences.py`
- `services/video_frames.py`
- `services/prelabels.py`
- `services/prelabel_adapters.py`
- `services/prelabel_queue.py`
- `services/annotation_payload.py`
- `services/dataset_selection.py`
- `services/dataset_export_builder.py`
- `services/exporter_coco.py`
- `services/inference_client.py`
- `services/augmentation.py`

### Worker

Entrypoint:

- `apps/worker/src/sheriff_worker/main.py`

Current relevant jobs:

- `jobs/extract_frames.py`
- `jobs/prelabel_asset.py`
- `jobs/inference_suggest.py`
- `jobs/build_export_zip.py`

### Trainer

Entrypoints and main modules:

- `apps/trainer/src/pixel_sheriff_trainer/main.py`
- `apps/trainer/src/pixel_sheriff_trainer/inference/app.py`
- `apps/trainer/src/pixel_sheriff_trainer/inference/schemas.py`
- `apps/trainer/src/pixel_sheriff_trainer/augmentation.py`

Current inference endpoints:

- classification infer/warmup
- detection infer/warmup
- segmentation infer
- Florence warmup/detect

Current experiment helpers in the web app:

- `apps/web/src/lib/api/types.ts`
- `apps/web/src/lib/workspace/augmentationConfig.js`
- `apps/web/src/lib/workspace/experimentAnalytics.js`
- `apps/web/src/app/projects/[projectId]/experiments/page.tsx`
- `apps/web/src/app/projects/[projectId]/experiments/[experimentId]/page.tsx`

## Current Backend Data Model Highlights

### Tasks

- projects auto-create a default task
- task kind controls allowed annotation payload structure
- classification tasks also carry `label_mode`

### Assets and Sequences

- assets are the canonical exported and annotated unit
- sequences provide grouping, ordering, and status for frame-backed workflows
- `source_kind` differentiates image, video frame, and webcam frame rows

### Annotations

Current payloads preserve:

- top-level classification fields
- normalized `classification` block
- `objects`
- `image_basis`
- optional object provenance

### Prelabels

Current tables:

- `PrelabelSession`
- `PrelabelProposal`

Current design:

- proposals remain pending until reviewed
- proposals never become normal annotations automatically
- accepted or edited proposals are promoted into the annotation payload

### Experiments and Augmentation

Current experiment training config fields:

- `augmentation_profile`: `none` | `light` | `medium` | `heavy` | `custom`
- `augmentation_spec_version`: `1` for the current contract
- `augmentation_steps`: ordered custom step list with `type`, `p`, and `params`

Supported custom augmentation steps:

- `horizontal_flip`
- `vertical_flip`
- `color_jitter`
- `rotate`

Current preset expansion:

- `none`: no augmentation
- `light`: `horizontal_flip(p=0.5)`
- `medium`: `horizontal_flip(p=0.5)` plus `color_jitter(p=1.0)` with shared defaults
- `heavy`: `medium` plus `rotate(p=1.0, degrees=8)`

Task-aware defaults:

- classification defaults to `light`
- detection defaults to `none`
- segmentation defaults to `none`

Compatibility behavior:

- legacy classification configs still honor stored preset profiles
- legacy detection and segmentation configs without `augmentation_spec_version=1` remain effectively unaugmented
- new or re-saved experiments in the current editor stamp `augmentation_spec_version=1`

Runtime application:

- classification applies image-only augmentation
- detection applies paired image and bbox augmentation
- segmentation applies paired image and mask augmentation
- detection rotation reprojects bbox corners, clips back to image bounds, and drops degenerate boxes

## Current API Surfaces Worth Knowing

### Media

- `POST /api/v1/projects/{project_id}/assets/upload`
- `POST /api/v1/projects/{project_id}/video-imports`
- `POST /api/v1/projects/{project_id}/webcam-sessions`
- `POST /api/v1/projects/{project_id}/sequences/{sequence_id}/frames`
- `GET /api/v1/projects/{project_id}/sequences`
- `GET /api/v1/projects/{project_id}/sequences/{sequence_id}`
- `GET /api/v1/projects/{project_id}/sequences/{sequence_id}/status`

### Deployments and Prediction Review

- `POST /api/v1/projects/{project_id}/deployments`
- `GET /api/v1/projects/{project_id}/deployments`
- `PATCH /api/v1/projects/{project_id}/deployments/{deployment_id}`
- `POST /api/v1/projects/{project_id}/predict`
- `POST /api/v1/projects/{project_id}/predict/batch`
- `POST /api/v1/projects/{project_id}/deployments/{deployment_id}/warmup`

### Prelabels

- `POST /api/v1/projects/{project_id}/tasks/{task_id}/prelabels`
- `GET /api/v1/projects/{project_id}/tasks/{task_id}/prelabels`
- `GET /api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}`
- `GET /api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/proposals`
- `POST /api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/accept`
- `POST /api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/reject`
- `POST /api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/close-input`
- `POST /api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/cancel`

### Suggestions and Deployments

- deployment CRUD/list/warmup
- project prediction endpoint
- suggestion list/decision endpoints

### Datasets, Models, Experiments

- task-scoped dataset version preview/create/export
- project-scoped model CRUD
- experiment CRUD/start/cancel/delete/runtime/logs/evaluation/analytics

Current experiment analytics payload includes effective augmentation metadata:

- `augmentation`
- `augmentation_mode`
- `augmentation_summary`

## Current UI Behavior Notes

### Asset Browser

- folder tree is independent from the canvas and right panel
- bottom danger-zone controls remain inside the sidebar
- tree supports filtering, collapse/expand, and folder-scope selection

### Sequence UI

- image-only assets use pagination/filmstrip
- sequence assets use the sequence toolbar, timeline, and thumbnails
- pending prelabel counts are surfaced on sequence frames
- there is a next-pending-AI navigation action

### Webcam Capture

- preview can remain visible while recording
- one sequence is created per selected camera destination
- frames are uploaded from a canvas snapshot path
- live status is shown in the modal while capturing

### AI Prelabels

- settings are shared between video import and webcam capture modals
- controls are shown only for bbox tasks
- pending proposals render as dashed overlay boxes with an AI badge
- review actions live in a dedicated AI Prelabels panel

### Experiment Editor

- the augmentation dropdown includes `none`, `light`, `medium`, `heavy`, and `custom`
- choosing `custom` reveals an ordered step editor with add, remove, and reorder controls
- each step stores its own probability and step-specific params
- switching away from `custom` hides the editor but keeps the custom draft state intact

### Experiment Analytics UI

- scatter plots continue to use categorical augmentation buckets
- custom configs plot under the `custom` bucket
- tooltips display the full custom augmentation summary

## Current Known Gaps

- full browser-side regression coverage for the AI prelabels review UI is still limited
- webcam/browser-specific intermittent frame-write issues still need deeper diagnostics
- geometry tooling is functional but not yet feature-complete for advanced editing

## Recommended Reading Order

1. `README.md`
2. `docs/architecture.md`
3. `docu/Architecture.md`
4. `docu/IMPLEMENTATION_TASKS.md`
5. `docu/VLM_COLD_START_PRELABELING_TASKS.md`
