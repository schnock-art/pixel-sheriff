# Roadplan

## Goal

Build a reliable labeling platform that preserves dataset contract integrity across:
labeling -> training -> deployment.

## Current Progress Snapshot

- `M0` Scaffold/infra: done
- `M1` Projects/categories core: mostly done
- `M2` Image ingestion: done for local files
- `M3` Labeling workstation: in progress (advanced state management now implemented)
- `M4+` Review/export hardening/video/MAL: pending

## Guiding Principles

- Dataset contract first.
- Category identity stability (immutable IDs).
- Versioned exports with deterministic hashes.
- Keep architecture simple until scale requires complexity.
- Local dev should be conflict-resistant (configurable ports).

## Milestones

### M0 - Foundation (Done)

- Monorepo + compose stack
- API/web/db/redis local startup
- Basic tests/scaffolds

### M1 - Project + Category Management (Mostly Done)

Implemented:
- project CRUD core
- category create/list/patch
- label management UI:
  - add
  - rename
  - reorder
  - activate/deactivate

Remaining:
- stronger validation UX around label edits

### M2 - Asset Ingestion (Done for image upload)

Implemented:
- local folder image import
- choose import destination:
  - existing project or new project
- configurable folder naming during import
- persisted relative paths for hierarchy rendering
- backend file persistence + content streaming endpoint

Remaining:
- image metadata extraction (`width/height`)
- optional dedup/checksum policy

### M3 - Labeling Workstation (In Progress)

Implemented:
- viewer navigation (buttons + arrow keys)
- staging/edit mode
- batch submit of staged edits
- single-label and multi-label toggle
- COCO-style classification payload fields
- hierarchical file tree navigation

Remaining:
- full hotkey map (1..9, skip, etc.)
- clearer staged/dirty indicators per asset

### M4 - Review + QA (Planned)

- review grid
- category/status filters
- bulk approve/needs_review actions

### M5 - Export v1 (Partially implemented)

Implemented:
- export version metadata + manifest hash record

Remaining:
- zip build
- downloadable artifact endpoint
- reproducibility tests

### M6 - Video Ingestion (Planned)

- video upload
- frame extraction pipeline
- provenance metadata

## Post-v1 Roadmap

### P1 - Multi-label hardening

- improve UX for multi-label assignment workflows
- category hotkeys in multi-select mode

### P2 - Bounding boxes

- geometry tools
- bbox export fields

### P3 - Segmentation/polygons

- polygon editor
- segmentation export fields

### P4 - MAL

- model registry
- suggestion generation
- accept/reject assist workflow

### P5 - Reference-mode asset ingestion (after MAL)

Support dual ingestion:
- upload mode (current)
- external-reference mode (store URI/key without uploading bytes)

### P6 - Local ops hardening

- env profiles for alternate local ports
- clearer dev startup diagnostics

### P7 - Import UX evolution (Feature request)

- keep one structured import dialog
- improve validation and defaults memory
- support import presets per project

### P8 - Shared asset library + project-specific labels (Feature request)

- upload asset once, link to multiple datasets/projects
- maintain independent annotations per project
- optional taxonomy templates reusable across projects
