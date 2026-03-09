# Architecture Overview

This is the current high-signal architecture reference for the codebase. The root [README](../README.md) is the quickstart; older notes under `docu/` are historical and may lag current implementation.

## Product Shape

Pixel Sheriff is image-first:

- Images are first-class assets.
- Video files and webcam sessions are frame sources, not a separate annotation engine.
- Extracted or captured frames become ordinary image assets inside a dedicated folder and asset sequence.
- The existing labeling canvas remains the only editor for images, video frames, and webcam frames.

Canonical media flow:

```text
video file / webcam stream
-> frame extraction or capture
-> asset rows + files in storage
-> existing annotation workflow
-> dataset export stays frame-based
```

## Runtime Topology

Default Docker services:

- `web` (`apps/web`) on `WEB_PORT`, default `3010`
- `api` (`apps/api`) on `API_PORT`, default `8010`
- `worker` (`apps/worker`) for background media jobs
- `trainer` (`apps/trainer`) for training runs and inference-backed suggestions
- `db` (PostgreSQL) on `POSTGRES_PORT`, default `5433`
- `redis` on `REDIS_PORT`, default `6380`

Data flow:

```text
browser
-> web
-> api
   -> postgres
   -> redis -> worker   (video frame extraction)
   -> trainer           (training/inference flows)
```

Important operational note:

- `make up` starts the full stack.
- `make up-web-api` does not start `worker` or `trainer`.
- Video import requires `worker`.
- Training and deployment-assisted suggestions require `trainer`.

## Storage Model

Mutable application state lives in PostgreSQL:

- `projects`
- `tasks`
- `folders`
- `asset_sequences`
- `assets`
- `annotations`
- supporting tables such as categories, suggestions, and deployments

Immutable or artifact-heavy data is file-backed under `./data`:

- `data/assets/`
- `data/imports/`
- `data/datasets/`
- `data/exports/`
- `data/models/`
- `data/experiments/`

The API bootstraps schema on startup with:

- `Base.metadata.create_all`
- `run_startup_migrations`

The startup migration layer also backfills first-class folders and sequence-related asset columns for legacy assets that only had `metadata_json.relative_path`.

## Core Data Concepts

### Folder

Folders are now first-class rows. They drive the sidebar tree and make empty processing folders visible before any extracted frames exist.

Key fields:

- `id`
- `project_id`
- `parent_id`
- `name`
- `path`

### AssetSequence

An asset sequence groups time-ordered assets that came from the same source.

Examples:

- imported video
- webcam capture session

Key fields:

- `id`
- `project_id`
- `task_id`
- `folder_id`
- `name`
- `source_type` (`video_file` or `webcam`)
- `status` (`processing`, `ready`, `failed`)
- `frame_count`
- `processed_frames`
- `fps`
- `duration_seconds`
- `width`
- `height`
- `error_message`

### Asset

Assets remain the canonical unit for annotation and export.

Relevant asset fields for the media flow:

- `folder_id`
- `file_name`
- `sequence_id`
- `source_kind` (`image`, `video_frame`, `webcam_frame`)
- `frame_index`
- `timestamp_seconds`

Sequence-backed assets are still normal assets. They are not special-cased by the annotation canvas.

## Current Media Flows

### Image Import

The image import flow still uploads files into a project/folder destination and creates asset rows. Existing image-only workflows remain unchanged.

### Video Import

API surface:

- `POST /api/v1/projects/{project_id}/video-imports`
- `GET /api/v1/projects/{project_id}/sequences`
- `GET /api/v1/projects/{project_id}/sequences/{sequence_id}`
- `GET /api/v1/projects/{project_id}/sequences/{sequence_id}/status`

Flow:

1. The API validates upload parameters and file type.
2. It creates a dedicated folder and `asset_sequences` row with `status=processing`.
3. The raw uploaded video is written under `data/imports/`.
4. A Redis media job is pushed to `MEDIA_QUEUE_KEY`.
5. `apps/worker` consumes the job and runs FFmpeg-based extraction.
6. Extracted frames are stored as normal assets, linked to the folder and sequence.
7. The sequence is marked `ready` or `failed`.

V1 behavior:

- one sequence owns one dedicated folder
- no raw video asset is stored in the normal asset table
- extracted frames are named deterministically (`frame_000001.jpg`, etc.)
- exports stay frame-based

### Webcam Capture

API surface:

- `POST /api/v1/projects/{project_id}/webcam-sessions`
- `POST /api/v1/projects/{project_id}/sequences/{sequence_id}/frames`

Flow:

1. The browser requests camera access with `getUserMedia`.
2. The UI creates a webcam session only when capture starts.
3. Frames are drawn into an offscreen canvas, encoded as JPEG, and uploaded sequentially.
4. Each uploaded frame becomes a normal asset linked to the sequence.

## Labeling Workspace

Main orchestration:

- `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`

Sequence-aware workspace pieces:

- `ImportMenu.tsx`
- `VideoImportModal.tsx`
- `WebcamCaptureModal.tsx`
- `SequenceToolbar.tsx`
- `SequenceTimeline.tsx`
- `SequenceThumbnailStrip.tsx`

Sequence-aware hooks:

- `useFolders`
- `useSequence`
- `useSequenceNavigation`
- `useWebcamCapture`

Behavior:

- plain image assets keep the existing pagination-first workflow
- sequence assets switch the bottom panel into frame navigation mode
- the center canvas and annotation tools stay unchanged

## Export Contract

Dataset export remains consumable as before:

- frames export exactly like image assets
- existing training consumers do not need a separate video path

For sequence-backed assets, export metadata now carries optional lineage details such as:

- `kind`
- `sequence_id`
- `sequence_name`
- `frame_index`
- `timestamp_seconds`

## Backend Map

- `apps/api/src/sheriff_api/main.py`
  - app entrypoint, router mounting, startup schema bootstrapping
- `apps/api/src/sheriff_api/db/models.py`
  - SQLAlchemy models including `Folder`, `AssetSequence`, and the updated `Asset` fields
- `apps/api/src/sheriff_api/services/migrations.py`
  - startup migration and legacy backfill logic
- `apps/api/src/sheriff_api/services/sequences.py`
  - folder/sequence serializers and helpers
- `apps/api/src/sheriff_api/services/video_frames.py`
  - FFmpeg extraction, failure cleanup, sequence state updates
- `apps/api/src/sheriff_api/routers/folders.py`
- `apps/api/src/sheriff_api/routers/video_imports.py`
- `apps/api/src/sheriff_api/routers/sequences.py`

## Frontend Map

- `apps/web/src/app/projects/[projectId]/`
  - project-scoped routes for labeling, datasets, models, experiments, and deploy
- `apps/web/src/components/workspace/project-assets/`
  - asset browser, import modals, sequence controls, and other labeling subcomponents
- `apps/web/src/lib/hooks/`
  - workspace, folder, sequence, import, and webcam state
- `apps/web/src/lib/api/`
  - domain-specific API clients and shared request/response types

## Legacy Docs

`docu/` still contains useful historical notes, but it should not be treated as the source of truth for the current codebase. Start with this file and the root README when orienting to the current implementation.
