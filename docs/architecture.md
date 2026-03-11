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

- active deployment suggestions for single-asset review
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

- deployment suggestions are current-asset, deployment-driven review helpers
- prelabels are sequence/session-driven pending proposals for frame review

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

- richer full-flow automated UI coverage for AI prelabel review
- deeper webcam frame-write diagnostics for intermittent browser/device issues
- more advanced geometry editing polish
