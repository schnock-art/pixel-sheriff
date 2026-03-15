# Architecture Overview

This is the current architecture reference for the implemented system.

## Product Shape

Pixel Sheriff is an image-first system:

- images are the canonical annotation unit
- videos are imported as extracted frame assets
- webcam capture uploads frame assets live
- the labeling workspace is shared across images and sequence frames
- dataset export remains frame-based

Task kinds:

- `classification`
- `bbox`
- `segmentation`

AI-assisted paths:

- review-first active deployment predictions for current-asset and folder-queue `classification` and `bbox` review
- sequence-first AI prelabels for bbox video and webcam workflows

## Runtime Topology

Default Docker services:

- `web` on `WEB_PORT`, default `3010`
- `api` on `API_PORT`, default `8010`
- `worker`
- `trainer`
- `db` on `POSTGRES_PORT`, default `5433`
- `redis` on `REDIS_PORT`, default `6380`

Flow:

```text
browser
-> web
-> api
   -> postgres
   -> redis -> worker
   -> trainer
```

Current queue separation:

- media extraction queue
- prelabel queue
- training queue
- suggestion queue

## Storage Model

Mutable state in PostgreSQL:

- `projects`
- `tasks`
- `categories`
- `folders`
- `asset_sequences`
- `assets`
- `annotations`
- `prelabel_sessions`
- `prelabel_proposals`
- `suggestions`

File-backed data under `./data`:

- `assets/`
- `imports/`
- `datasets/`
- `exports/`
- `models/`
- `experiments/`

Startup bootstrapping:

- SQLAlchemy `Base.metadata.create_all`
- startup migration runner in `apps/api/src/sheriff_api/services/migrations.py`

## Core Domain Concepts

### Folder

First-class row used for tree navigation and sequence ownership.

Key fields:

- `id`
- `project_id`
- `parent_id`
- `name`
- `path`

### AssetSequence

Time-ordered asset group for one video import or webcam session.

Key fields:

- `id`
- `project_id`
- `task_id`
- `folder_id`
- `name`
- `source_type`
- `status`
- `frame_count`
- `processed_frames`
- `fps`
- `duration_seconds`
- `width`
- `height`
- `error_message`

### Asset

Canonical annotation/export unit.

Relevant sequence fields:

- `folder_id`
- `file_name`
- `sequence_id`
- `source_kind`
- `frame_index`
- `timestamp_seconds`

### Annotation

One annotation row per `(asset_id, task_id)`.

Current payload model:

- classification block
- object geometry list for bbox or polygon tasks
- optional per-object provenance
- optional shared `prediction_review` metadata for accepted deployment reviews

### PrelabelSession

Sequence-level AI prelabel job orchestration record.

Current fields include:

- source selection
- prompts
- sampling mode/value
- confidence threshold
- live-mode flag
- enqueue/process/proposal counters
- input-closed timestamp
- terminal status and error message

### PrelabelProposal

Pending AI-generated bbox proposal stored separately from normal annotations.

Current fields include:

- source session
- asset/task/project/category references
- label text and prompt text
- confidence
- bbox in `xywh`
- proposal review status
- reviewed bbox/category
- promoted annotation/object references

## Current Flows

### Image Import

Browser uploads image bytes to the API, which persists local storage files and asset rows.

### Video Import

API:

- creates folder and `asset_sequences` row
- stores uploaded video in `imports/`
- enqueues extraction job

Worker:

- extracts JPEG frames with FFmpeg
- persists frame assets
- marks the sequence `ready` or `failed`

If `prelabel_config` is present:

- a `PrelabelSession` is created at import time
- once extraction completes, sampled frame jobs are enqueued on the prelabel queue

### Webcam Capture

Browser:

- gets camera access with `getUserMedia`
- previews the selected devices
- captures JPEG frames from a live video element

API:

- creates one sequence per camera destination
- accepts uploaded frames through `/sequences/{sequence_id}/frames`

If `prelabel_config` is present:

- one live `PrelabelSession` is created per webcam sequence
- sampled frames are enqueued for prelabel work while capture is active
- modal finish triggers `close-input`

### AI Prelabels

Supported sources:

- `active_deployment`
- `florence2`

Source behavior:

- active deployment is resolved at session creation time
- Florence runs in `apps/trainer`
- API owns adapter selection and session orchestration
- worker owns background `prelabel_asset` execution

Review behavior:

- pending proposals do not enter normal annotations
- accept merges proposals into the annotation payload without overwriting unrelated manual objects
- edit loads a provenance-backed bbox into the normal annotation draft
- saved provenance-backed objects mark proposals as `accepted` or `edited`

### Suggestions

Separate from prelabels:

- deployment predictions are current-asset, deployment-driven review helpers
- prelabels are sequence/session-driven pending proposals for frame review

### Deployment Prediction Review

Current request path:

- browser loads compatible deployments from `/projects/{project_id}/deployments`
- the labeling panel sends `POST /projects/{project_id}/predict` for the currently selected asset
- folder review sends `POST /projects/{project_id}/predict/batch` for the asset ids in the selected folder scope
- deployed prediction review is synchronous and separate from the legacy queued `/suggestions/batch` flow

Frontend review state:

- `/predict` responses are normalized into a `pendingReview` model in `useWorkspaceSuggestions`
- `/predict/batch` responses populate per-asset pending review queue state and auto-advance between pending assets after accept/reject
- bbox responses also create `preview_objects` for the viewer overlay
- while a pending classification review exists, label toggles, edit mode, and submit are disabled to avoid mixed draft state
- while a pending bbox review exists, the normal draft remains protected, but the pending predicted boxes can be selected, moved, resized, or deleted before accept

Accept and reject semantics:

- `Reject prediction` clears the pending review only
- classification accept stages the selected class in the normal draft and attaches shared `prediction_review` metadata
- bbox accept replaces the current asset draft object set with the accepted prediction boxes
- deleting the final pending bbox prediction is treated as rejecting that review
- accepted predictions still require the normal annotation `Submit` flow to persist

Persistence details:

- accepted bbox objects keep provenance with `origin_kind: deployment_prediction`
- bbox provenance may include `source_model`, `confidence`, and `review_decision`
- accepted classification reviews are written through the normal annotation payload with a `prediction_review` block

Current UI scope:

- supported in the labeling panel for `classification` and `bbox`
- not yet surfaced for segmentation
- folder-scoped batch inference is available, but review decisions are still applied per image rather than via bulk accept-all/reject-all

## Labeling Workspace

Main composition:

- `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`

Major UI regions:

- asset browser and folder tree
- canvas and geometry tools
- right-side labeling panel
- dedicated AI prelabels panel
- image pagination or sequence controls depending on current asset

Sequence-aware pieces:

- `VideoImportModal.tsx`
- `WebcamCaptureModal.tsx`
- `SequenceToolbar.tsx`
- `SequenceTimeline.tsx`
- `SequenceThumbnailStrip.tsx`

Sequence-aware hooks:

- `useSequence`
- `useSequenceNavigation`
- `useWebcamCapture`
- `usePrelabels`

Deployment review pieces:

- `apps/web/src/lib/hooks/useWorkspaceSuggestions.ts`
- `apps/web/src/lib/hooks/useAnnotationWorkflow.ts`
- `apps/web/src/components/LabelPanel.tsx`
- `apps/web/src/components/Viewer.tsx`

## Export Contract

Exports are still image/frame based.

Current bundle contents:

- `manifest.json`
- `coco_instances.json`
- packaged asset files

Current behavior:

- UUID asset identity is preserved across manifest and COCO
- bbox and polygon geometry export into COCO object annotations
- sequence-derived assets retain lineage metadata such as `sequence_id`, `frame_index`, `timestamp_seconds`, and `source_kind`
- pending or rejected prelabel proposals are not part of export generation

## Backend Map

- `apps/api/src/sheriff_api/main.py`
- `apps/api/src/sheriff_api/db/models.py`
- `apps/api/src/sheriff_api/routers/`
- `apps/api/src/sheriff_api/services/sequences.py`
- `apps/api/src/sheriff_api/services/video_frames.py`
- `apps/api/src/sheriff_api/services/prelabels.py`
- `apps/api/src/sheriff_api/services/prelabel_adapters.py`
- `apps/api/src/sheriff_api/services/prelabel_queue.py`

## Worker and Trainer Map

Worker:

- `apps/worker/src/sheriff_worker/main.py`
- `apps/worker/src/sheriff_worker/jobs/extract_frames.py`
- `apps/worker/src/sheriff_worker/jobs/prelabel_asset.py`

Trainer:

- training execution and experiment artifacts
- deployment-backed inference endpoints
- Florence warmup and detect endpoints

Key trainer inference endpoints:

- `/infer/classification`
- `/infer/classification/warmup`
- `/infer/detection`
- `/infer/detection/warmup`
- `/infer/segmentation`
- `/infer/florence/warmup`
- `/infer/florence/detect`

## Frontend Map

- `apps/web/src/app/projects/[projectId]/`
- `apps/web/src/components/workspace/project-assets/`
- `apps/web/src/lib/hooks/`
- `apps/web/src/lib/workspace/`
- `apps/web/src/lib/api/`

## Current Gaps

Implemented system is broader than the original classification-only base, but some work is still intentionally open:

- one-click bulk accept-all/reject-all for deployment prediction review
- segmentation deployment review in the labeling UI
- broader end-to-end coverage for sequence AI prelabel review
- deeper webcam frame-write diagnostics for intermittent browser/device issues
- more advanced geometry editing polish
