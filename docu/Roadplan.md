# Roadplan

## Goal

Build a reliable labeling platform that preserves dataset contract integrity across:
labeling -> training -> deployment.

## Current Snapshot

- `M0` foundation: done
- `M1` project/category/task management: done
- `M2` image ingestion/storage: done for local workflow
- `M3` labeling workstation + model/experiment/deploy flows: done for core classification path
- `M4+` review/export hardening/video/advanced MAL: in progress

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

### M1 - Project + Category + Task Management (Done)

Implemented:
- project create/list/get
- project delete (with related asset/annotation cleanup)
- task create/list/get/delete (with reference guards)
- category create/list/patch/delete
- label management UI:
  - add
  - rename
  - reorder
  - activate/deactivate

Remaining:
- richer validation UX around destructive task/category actions

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
- import progress card (files, bytes, rates, ETA)
- robust import diagnostics:
  - extension fallback when MIME is missing
  - per-file read/network/API error details

Remaining:
- optional dedup strategy

### M3 - Labeling Workstation (Core done, polish ongoing)

Implemented:
- project-scoped workspace navigation:
  - `/projects` entry flow (last/first project resolution or create-project empty state)
  - `/projects/{project_id}/datasets|dataset|models|experiments|deploy` routing
  - model/experiment detail routes
  - reusable project ribbon with project selector, task selector, workflow tabs, and project stats
- task-scoped labeling:
  - global task selector + create-task modal in project ribbon
  - task kind controls annotation mode and payload contract
  - task label mode (`single_label`/`multi_label`) is task-owned
  - task label schema lock once dataset versions exist
- viewer navigation (buttons + arrow keys)
- skip controls (`-10`, `-5`, `<`, `>`, `+5`, `+10`)
- numeric keyboard label shortcuts (`1..9`, top row + numpad)
- staged edit mode
- batch submit staged edits
- single-label and multi-label assignment
- explicit clear-label action and assigned-label visibility summary in classification mode
- hierarchical file tree navigation
- folder-scoped review selection from tree
- tree expand/collapse controls (single folder + collapse/expand all)
- labeled/unlabeled visual status in pagination and tree
- delete tools:
  - single image remove
  - bulk image remove
  - folder/subfolder subtree remove
- responsive bounded viewport with contained image rendering
- adaptive pagination window sized to available width
- bottom filmstrip for quick in-scope asset navigation
- COCO-style classification payload fields
- bounding-box annotation workflow (draw/select/move/resize/delete + class assign)
- polygon segmentation workflow (draw/close/select/delete + class assign; close by near-start, double-click, or `Enter`)
- geometry-aware staged/pending submit workflow
- task-kind enforcement (`classification`, `bbox`, `segmentation`) across API + UI
- deterministic class-based color mapping for labels and geometry overlays
- toast-style operation summaries including delete counts
- inline geometry draft warnings in the viewer
- unsaved draft guard for project/tab/model-builder navigation
- project-scoped model scaffold flow:
  - labeling `Create Dataset` routes the user into dataset-versioning flow
  - dataset `Train Model` routes into prefilled `/models/new`
  - models page supports `+ New Model`, empty state, dataset-version-aware summary table, and inferred status chips
  - model detail supports editable builder + save validation + train navigation
- experiments/deploy:
  - experiments list/create/detail are wired with runtime/log/evaluation dashboards
  - deploy page supports deploy-from-experiment, active deployment selection, device preference, and warmup
  - labeling panel integrates MAL single-asset suggestions (`Suggest`, top-k, `Apply top-1`)

Remaining:
- continue monitoring intermittent annotation submit `404` during stale project/asset submit contexts after staged-entry pruning + regression hardening
- geometry edit tooling polish (polygon vertex editing and advanced transforms)
- stronger submission feedback/summary UX
- improve shell status counters for models/experiments (currently placeholder values in top bar)

### M4 - Review + QA (Planned)

- review grid and filters
- bulk moderation actions
- QA metrics panels

### M5 - Export v1.2 (Mostly implemented)

Implemented:
- dataset-version scoped export create/download (`/datasets/versions/{id}/export*`)
- manifest hash generation
- zip artifact creation (`assets/`, `coco_instances.json`, `manifest.json`)
- manifest schema v1.2 contract with explicit:
  - `tasks`
  - `label_schema` (stable `class_order`)
  - `splits`
  - `training_defaults`
  - `stats`
- canonical UUID identity mapping across manifest + COCO (`asset_id` / `image_id`)
- configurable detection/segmentation negative-image policy (`include_negative_images`)
- export-time geometry and join integrity validation
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

### P2 - Bounding boxes (Core implemented)

- geometry tools (v1) implemented
- bbox export integration implemented
- remaining: rotation/advanced transform tooling polish

### P3 - Segmentation/polygons (Core implemented)

- polygon tools (v1) implemented
- segmentation export support implemented
- remaining: vertex editing and mask tooling

### P4 - Model-assisted labeling (MAL)

- model registry: core endpoints implemented
- suggestion generation:
  - single-asset inference path is implemented via active deployment + `/predict`
  - queue-based batch/curation inference remains pending
- accept/reject assist workflow: API contract implemented

### P5 - Reference-mode asset ingestion (after MAL)

Support dual ingestion:
- upload mode (current)
- external-reference mode (URI/object key without file upload)

### P6 - Local ops hardening

- environment profiles for alternate ports
- startup diagnostics and readiness checks

### P7 - Import UX evolution

- task-type selection for new projects: implemented
- validation/default memory in import dialog: implemented
- optional import presets per project

### P8 - Shared asset library + project-specific labels (feature request)

- upload asset once and link to multiple projects
- independent per-project annotation state
- optional taxonomy templates reusable across projects

### P9 - Maintainability + Refactor Track

Goal:
- keep current behavior stable while reducing complexity and bug surface

Planned sequence:
1. integrity hardening:
   - validate upload project before file write
   - prevent cross-project annotation mutation
   - add rollback cleanup + targeted API tests
2. workflow consistency:
   - ensure unlabeled/clear-label submit path remains reachable
   - tighten staged vs committed annotation state handling
3. frontend decomposition:
   - split large workspace page into focused workflow hooks/modules
   - move tree/pagination logic to pure tested helpers
4. regression coverage:
   - replace placeholder hotkey tests with real UI interaction tests
   - add delete + folder-scope + submit regression suites

Completed in current milestone:
- decomposed API experiments router into concern-focused modules (`crud`, `analytics`, `evaluation`, `runs`, shared helpers)
- paired decomposition with dedicated experiments regression coverage (`apps/api/tests/test_experiments_api.py`)
