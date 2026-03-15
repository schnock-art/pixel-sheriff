# Roadplan

## Goal

Keep Pixel Sheriff reliable across the full contract:

```text
labeling
-> dataset version
-> model config
-> experiment
-> deployment
-> assisted review
```

## Current Snapshot

Implemented:

- project and task shell
- image import
- video import and frame extraction
- webcam capture into frame sequences
- classification, bbox, and segmentation labeling
- dataset versions and export zip generation
- models, experiments, and deployments
- deployment suggestions
- bbox sequence prelabels for video and webcam flows

Still open:

- deeper QA/review tooling
- richer export history UX
- deeper diagnostics and coverage around webcam/browser capture edge cases
- more automated coverage for the full AI prelabel review UI

## Guiding Principles

- dataset contract first
- task-scoped correctness over convenience
- deterministic export and artifact behavior
- image-first storage and labeling model
- keep orchestration simple unless a separate path is justified

## Milestones

### M0 - Foundation

Status: done

- monorepo structure
- Docker Compose stack
- test scaffolding

### M1 - Project and Task Management

Status: done

- project CRUD
- default task bootstrap
- task CRUD with reference guards
- task-scoped category management

### M2 - Media Ingestion

Status: largely done

Implemented:

- image import
- video import with extraction
- webcam capture with live frame uploads
- folder and sequence tracking

Remaining:

- optional dedup strategy
- stronger diagnostics for intermittent webcam write issues

### M3 - Labeling Workspace

Status: core done, polish ongoing

Implemented:

- task-aware labeling workspace
- classification/bbox/segmentation authoring
- folder tree and scope filtering
- sequence navigation for frame-backed assets
- staged editing plus direct submit
- deployment suggestions
- AI prelabel review panel and pending overlay for bbox sequences

Remaining:

- more advanced geometry editing
- stronger review and submit summaries
- richer automated UI coverage for prelabel review behavior

### M4 - Dataset and Export

Status: core done

Implemented:

- immutable dataset versions
- saved split membership
- export bundles with `manifest.json` and `coco_instances.json`
- frame lineage metadata
- geometry-aware COCO export
- pending prelabels excluded from preview/export

Remaining:

- export history and filtering UI polish

### M5 - Models, Experiments, Deployments

Status: core done

Implemented:

- project-scoped model drafts
- experiment create/start/cancel/delete flows
- runtime, log, and evaluation views
- deployment selection and warmup
- deployment-backed workspace suggestions

Remaining:

- continued runtime UX and artifact review polish

### M6 - Sequence AI Assistance

Status: v1 done, expansion open

Implemented:

- bbox-only prelabel sessions
- `active_deployment` and `florence2` sources
- video auto-start after extraction
- webcam live enqueue during capture
- pending proposal overlay and review panel
- accept/edit promotion into annotations with provenance

Remaining:

- broader automated frontend coverage
- more trainer-response edge-case coverage
- any future non-bbox or tracking-oriented expansion

### M7 - QA and Review Tooling

Status: planned

- review grid and moderation flows
- quality metrics and reviewer tooling
- sequence review shortcuts beyond the current pending-frame jump

## Near-Term Priority Order

1. Stabilize webcam capture behavior and diagnostics.
2. Fill AI prelabel UI automation gaps.
3. Improve export history and review UX.
4. Continue targeted geometry/editor polish.

## Post-v1 Directions

- richer MAL batch workflows
- reference-mode asset ingestion
- shared asset library
- more advanced QA/review flows
