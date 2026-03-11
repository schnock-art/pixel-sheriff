# Pixel Sheriff Vision

This note captures the product direction that the current repository now implements.

## Product shape

Pixel Sheriff is a local-first computer-vision workflow for:
- importing images
- extracting video into frame sequences
- capturing webcam streams into live frame sequences
- labeling classification, bbox, and segmentation tasks
- building dataset exports
- defining model configs
- running trainer-backed experiments
- deploying classification models for suggestion and review flows
- generating sequence-first bbox AI prelabels for video and webcam review

The system is intentionally opinionated:
- assets and annotations stay local
- workflows are project-scoped and task-scoped
- sequence media is reviewed inside the same workspace as ordinary images
- pending AI output is kept separate from accepted annotations until a human reviews it

## Current operating principles

### Local-first
- Core workflows run through the local `web`, `api`, `worker`, `trainer`, `db`, and `redis` stack.
- Storage is file-backed for assets and exported artifacts.
- The trainer is the only service that needs heavier ML runtime dependencies.

### Human review before promotion
- Model-assisted labeling for classification suggestions and bbox prelabels is assistive, not authoritative.
- Pending prelabels live in dedicated proposal rows and are only merged into annotations through explicit review.
- Export generation ignores pending or rejected AI proposals.

### Sequence-first media support
- Video and webcam inputs are represented as ordered asset sequences.
- Frame assets remain normal assets with lineage metadata such as `sequence_id`, `frame_index`, and `timestamp_seconds`.
- The labeling workspace exposes timeline, thumbnails, pending counts, and next-pending navigation for sequence review.

### Task-scoped contracts
- Categories, annotations, datasets, models, experiments, deployments, and prelabels are scoped by task.
- Current task kinds are:
  - `classification`
  - `bbox`
  - `segmentation`
- AI prelabels are intentionally bbox-only in the current implementation.

## Current non-goals

The repository does not currently aim to provide:
- browser-side inference
- tracking or sequence-level object identity
- segmentation or classification prelabels
- arbitrary custom training backends
- cloud-hosted orchestration as a requirement

## Immediate gaps

Areas still called out by the current codebase and docs:
- richer automated UI coverage for the AI prelabel review flow
- stronger diagnostics around intermittent webcam/browser frame-write failures
- additional sequence review ergonomics beyond the current pending-frame jump
