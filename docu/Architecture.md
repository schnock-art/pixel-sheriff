# Pixel Sheriff Architecture

This file mirrors the current implementation at a slightly more practical level than `docs/architecture.md`.

## System Summary

Pixel Sheriff currently consists of:

- `apps/web`: Next.js project shell and labeling workspace
- `apps/api`: FastAPI application with startup migrations
- `apps/worker`: Redis-backed background worker
- `apps/trainer`: training and inference service
- `db`: PostgreSQL
- `redis`: queues

The product is no longer image-only. It now supports:

- direct image import
- video import with extraction into frame sequences
- webcam capture into live frame sequences
- bbox AI prelabels for sequence review

## Current Workflow Layers

### Projects and Tasks

- every project has a default task
- tasks are first-class and drive annotation mode
- categories, annotations, datasets, models, experiments, deployments, and prelabels are task-scoped

Task kinds:

- `classification`
- `bbox`
- `segmentation`

### Assets, Folders, and Sequences

- folders are first-class records for tree navigation
- assets remain the canonical unit for annotation and export
- sequences group time-ordered frame assets
- webcam and video frames are ordinary assets with lineage fields

### Annotation Contract

Current annotation payloads support:

- classification label state
- bbox or polygon geometry objects
- optional `image_basis`
- optional geometry provenance

Prelabel provenance fields currently supported on saved objects:

- `origin_kind`
- `session_id`
- `proposal_id`
- `source_model`
- `prompt_text`
- `confidence`
- `review_decision`

### AI Assistance

There are now two different AI assistance paths:

1. deployment suggestions
   - single-asset inference helpers in the labeling panel
   - tied to the selected or active deployment

2. prelabels
   - sequence/session-driven pending bbox proposals
   - reviewed in the workspace before promotion into annotations

## Service Responsibilities

### Web

Main workspace composition:

- `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`

Key current UI pieces:

- `AssetBrowser`
- `Viewer`
- `LabelPanel`
- `AiPrelabelsPanel`
- `VideoImportModal`
- `WebcamCaptureModal`
- `SequenceToolbar`
- `SequenceTimeline`
- `SequenceThumbnailStrip`

Current hooks of interest:

- `useAnnotationWorkflow`
- `useSequence`
- `useSequenceNavigation`
- `useWebcamCapture`
- `usePrelabels`
- `useWorkspaceSuggestions`

### API

Main responsibilities:

- persistence and query layer
- media/session orchestration
- dataset/model/experiment/deployment APIs
- prelabel session lifecycle APIs
- adapter selection for prelabel sources
- export construction and artifact serving

Important service modules:

- `services/sequences.py`
- `services/video_frames.py`
- `services/prelabels.py`
- `services/prelabel_adapters.py`
- `services/prelabel_queue.py`
- `services/inference_client.py`

### Worker

Current background jobs:

- video frame extraction
- prelabel asset processing
- other existing asynchronous jobs such as export building and suggestion work

### Trainer

Current trainer responsibilities:

- experiment training
- ONNX/inference-backed deployment support
- Florence-2 warmup and detection endpoints

## Current Prelabel Design

Implemented scope:

- sequence-first
- bbox-only
- pending proposals live outside normal annotations
- accepted or edited results move into the asset annotation payload

Current tables:

- `prelabel_sessions`
- `prelabel_proposals`

Current public API surface:

- `POST /projects/{project_id}/tasks/{task_id}/prelabels`
- `GET /projects/{project_id}/tasks/{task_id}/prelabels/{session_id}`
- `GET /projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/proposals`
- `POST /projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/accept`
- `POST /projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/reject`
- `POST /projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/close-input`
- `POST /projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/cancel`

Current sources:

- `active_deployment`
- `florence2`

Current review UI:

- dedicated AI Prelabels panel
- dashed read-only overlay boxes with AI badge
- accept, reject, accept frame, reject frame, accept session, and edit-selected actions
- next-pending-frame navigation in sequence controls

## Current Media Flows

### Video

1. upload video
2. create folder and sequence
3. enqueue extraction
4. worker extracts frames
5. API records frame assets
6. optional prelabel session enqueues sampled frame jobs

### Webcam

1. request preview
2. create one sequence per camera destination on capture start
3. keep browser preview live while capturing
4. upload frames as JPEG snapshots
5. optional live prelabel session enqueues sampled frames
6. modal finish closes prelabel input

## Export and Dataset Behavior

- dataset versions are immutable records
- exports remain image/frame based
- bbox and segmentation geometry are emitted to COCO
- pending prelabels do not participate in dataset preview or export

## Practical Orientation

If you are modifying the current product, start here:

- feature overview: `README.md`
- architecture: `docs/architecture.md`
- implementation status: `docu/IMPLEMENTATION_TASKS.md`
- prelabel-specific status: `docu/VLM_COLD_START_PRELABELING_TASKS.md`

If you are reading older dated notes under `docs/plans/`, treat them as historical design context, not source-of-truth runtime documentation.
