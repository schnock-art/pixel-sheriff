# Roadplan

## Goal

Build a reliable labeling platform that preserves dataset contract integrity across:
labeling -> training -> deployment.

## Current Snapshot

- `M0` foundation: done
- `M1` project/category management: mostly done
- `M2` image ingestion/storage: done for local workflow
- `M3` labeling workstation: in progress (major functionality implemented)
- `M4+` review/export hardening/video/MAL: pending

## Guiding Principles

- Dataset contract first.
- Category identity stability (immutable IDs).
- Deterministic export versioning and hashability.
- Prefer simple architecture until complexity is justified.
- Maintain conflict-resistant local development defaults.

## Milestones

### M0 - Foundation (Done)

- Monorepo layout
- Docker Compose stack
- Basic scaffolds/tests

### M1 - Project + Category Management (Mostly done)

Implemented:
- project create/list/get
- category create/list/patch
- label management UI:
  - add
  - rename
  - reorder
  - activate/deactivate

Remaining:
- richer validation UX around label management actions

### M2 - Asset Ingestion (Done for image upload)

Implemented:
- local folder image import
- import destination selection:
  - existing project
  - new project
- existing-folder/subfolder destination support for existing project imports
- editable target folder naming
- persisted relative paths for hierarchy rendering
- backend image storage + streaming endpoint
- automatic asset/tree refresh after import

Remaining:
- image metadata extraction (`width/height`)
- optional dedup strategy

### M3 - Labeling Workstation (In progress)

Implemented:
- viewer navigation (buttons + arrow keys)
- staged edit mode
- batch submit staged edits
- single-label and multi-label assignment
- hierarchical file tree navigation
- folder-scoped review selection from tree
- tree expand/collapse controls (single folder + collapse/expand all)
- labeled/unlabeled visual status in pagination and tree
- COCO-style classification payload fields

Remaining:
- broader hotkey set (`1..9`, skip, etc.)
- clearer dirty/staged indicators in queue/tree/pagination
- stronger submission feedback/summary UX

### M4 - Review + QA (Planned)

- review grid and filters
- bulk moderation actions
- QA metrics panels

### M5 - Export v1 (Mostly implemented)

Implemented:
- export metadata record creation/list
- manifest hash generation
- zip artifact creation (`images/`, `annotations.json`, `manifest.json`)
- download endpoint
- web export trigger (one-click from workspace)

Remaining:
- web export workflow polish (filters/history panel)
- reproducibility/integrity tests

### M6 - Video Ingestion (Planned)

- video upload
- frame extraction
- frame provenance integration

## Post-v1 Roadmap

### P1 - Multi-label hardening

- improve multi-select UX and shortcuts
- strengthen payload validation/rules

### P2 - Bounding boxes

- geometry tools
- bbox export integration

### P3 - Segmentation/polygons

- polygon/mask tools
- segmentation export support

### P4 - Model-assisted labeling (MAL)

- model registry
- suggestion generation
- accept/reject assist workflow

### P5 - Reference-mode asset ingestion (after MAL)

Support dual ingestion:
- upload mode (current)
- external-reference mode (URI/object key without file upload)

### P6 - Local ops hardening

- environment profiles for alternate ports
- startup diagnostics and readiness checks

### P7 - Import UX evolution

- improve validation/default memory in import dialog
- optional import presets per project

### P8 - Shared asset library + project-specific labels (feature request)

- upload asset once and link to multiple projects
- independent per-project annotation state
- optional taxonomy templates reusable across projects
