# Implementation Tasks

Status reflects current repository behavior.

## Completed

### Infra / Compose
- [x] Docker compose services for `web`, `api`, `db`, `redis`
- [x] Configurable host ports via `.env`
- [x] API data volume mount (`./data -> /app/data`)
- [x] DB init script wiring
- [x] Trainer GPU runtime defaults wired in compose (`gpus`, NVIDIA envs) with CUDA wheel index override (`TRAINER_PYTORCH_INDEX_URL`)

### API (`apps/api`)
- [x] FastAPI app + router mounting under `/api/v1`
- [x] Startup DB table initialization
- [x] Project endpoints (create/list/get)
- [x] Project delete endpoint
- [x] Category endpoints (create/list/patch)
- [x] Asset endpoints:
  - [x] create/list
  - [x] multipart upload
  - [x] content stream
  - [x] delete single asset
- [x] Annotation endpoints (upsert/list)
- [x] Export metadata endpoints (create/list)
- [x] Export zip build (`manifest.json`, `coco_instances.json`, `assets/`)
- [x] Export download endpoint
- [x] MAL placeholder endpoints (models/suggestions)
- [x] Project-scoped model scaffold endpoints:
  - [x] `GET /projects/{project_id}/models`
  - [x] `POST /projects/{project_id}/models` (manifest-derived deterministic `ModelConfig` + schema validation)
  - [x] `GET /projects/{project_id}/models/{model_id}`
  - [x] `PUT /projects/{project_id}/models/{model_id}` (server-side schema validation before persistence)
  - [x] temporary file-backed model persistence (`models/{project_id}/records.json`)
- [x] Local storage safety checks (path containment)
- [x] API packaging fixes for `src` layout Docker builds
- [x] Experiment runtime/log observability endpoints:
  - [x] `GET /projects/{project_id}/experiments/{experiment_id}/runtime`
  - [x] `GET /projects/{project_id}/experiments/{experiment_id}/logs?from_byte=&max_bytes=`
  - [x] stable `runtime_not_found` / `training_log_not_found` error codes on missing artifacts
- [x] Experiment analytics payload now includes optional runtime summary (`device_selected`) for list-view badges
- [x] ONNX artifact serving for experiments:
  - [x] `GET /projects/{project_id}/experiments/{experiment_id}/onnx` (metadata + artifact URLs)
  - [x] `GET /projects/{project_id}/experiments/{experiment_id}/onnx/download?file=model|metadata`
  - [x] stable `onnx_not_found` error code for absent ONNX artifacts

### Web (`apps/web`)
- [x] Root route now redirects to `/projects`
- [x] Project-scoped shell routes implemented:
  - [x] `/projects/{project_id}/datasets`
  - [x] `/projects/{project_id}/models`
  - [x] `/projects/{project_id}/models/new`
  - [x] `/projects/{project_id}/models/{model_id}`
  - [x] `/projects/{project_id}/experiments`
  - [x] `/projects/{project_id}/experiments/{experiment_id}`
- [x] Project shell UI:
  - [x] project selector dropdown
  - [x] create-project modal
  - [x] section tabs (`Datasets`, `Models`, `Experiments`, disabled `Deploy`)
  - [x] project status summary bar
- [x] Datasets workspace extracted to `ProjectAssetsWorkspace`
- [x] Unsaved-draft guard on project/tab/build-model navigation
- [x] Build Model CTA creates a project-scoped model draft and opens model detail
- [x] Models pages support list/create/detail + editable Model Builder draft/save workflow
- [x] AJV validation layer for web JSON Schema checks (`ajv` + `ajv-formats`) wired for ModelConfig draft validation
- [x] Experiments placeholder pages/skeletons
- [x] Fixed project switch activation/navigation bug in shell
- [x] Responsive styling + viewer letterbox rendering
- [x] Bounded responsive viewport height (stable with large datasets)
- [x] Hierarchical file tree with folder/file navigation
- [x] Import dialog:
  - [x] existing vs new project
  - [x] existing folder/subfolder destination option (existing-project imports)
  - [x] target folder naming
- [x] Auto-refresh asset/tree view after import
- [x] Hierarchical file tree with preserved parent/child ordering
- [x] Folder-scoped queue selection from tree
- [x] Tree expand/collapse (per-folder + collapse all/expand all)
- [x] Labeled/unlabeled status coloring in tree and pagination
- [x] Adaptive pagination (width-aware chip window + `First`/`Last`)
- [x] Viewer skip controls (`-10`, `-5`, `<`, `>`, `+5`, `+10`)
- [x] Robust import diagnostics:
  - [x] MIME + extension filtering fallback
  - [x] detailed per-file failure messages
- [x] Import progress/throughput/ETA panel
- [x] Label panel features:
  - [x] add label
  - [x] manage labels (rename/reorder/activate/deactivate)
  - [x] project-scoped multi-label toggle (managed in label manage mode)
  - [x] add-label input visible only in manage mode
  - [x] clear selected labels action (classification mode)
  - [x] assigned-label summary visibility (`Assigned: ...`) for current image
- [x] Annotation UX:
  - [x] edit mode staging
  - [x] batch submit staged changes
  - [x] non-edit single submit path
  - [x] inline draft warnings for uncommitted bbox/polygon geometry
- [x] Keyboard navigation with arrow keys
- [x] Keyboard labeling shortcuts (`1..9`, top-row and numpad)
- [x] Delete UX:
  - [x] project delete action
  - [x] single-image delete action
  - [x] multi-image delete mode and selection
  - [x] folder/subfolder delete from tree
  - [x] toast-style delete summaries with counts
- [x] Experiment runtime/log observability UI:
  - [x] experiment detail `Runtime & Logs` panel with runtime summary and collapsible training-log viewer
  - [x] refresh + auto-refresh polling for logs while status is `queued`/`running`
  - [x] status-adjacent runtime badge in detail view (`CUDA`/`CPU`/`MPS`)
  - [x] runtime badge surfaced in experiments list table
- [x] Experiment ONNX export panel:
  - [x] ONNX status card on experiment detail (`Exported`/`Pending`/`Failed`)
  - [x] model + metadata download actions
  - [x] ONNX metadata display (input shape, class order, validation badge)
  - [x] SSE-driven ONNX refresh on `onnx_export` events

### Trainer (`apps/trainer`)
- [x] Classification-first trainer scalability/efficiency pass:
  - [x] BN safety via train-loader `drop_last=True` default (no per-batch BN mode toggling fallback)
  - [x] explicit fail-fast guard for invalid BN small-batch config combinations
  - [x] configurable evaluation cadence (`evaluation.eval_interval_epochs`) with null `val_*` metrics on skipped epochs
  - [x] checkpoint payload policy split:
    - [x] `latest` resumable checkpoint includes optimizer/scheduler state
    - [x] `best_loss` / `best_metric` checkpoints store model-only weights + metadata
  - [x] async bounded checkpoint writer queue with `latest`-drop policy under backpressure
  - [x] checkpoint write failures downgraded to warnings/metadata (no run-failure escalation)
  - [x] CUDA throughput defaults (`pin_memory`, `non_blocking` copies, runtime worker knobs)
  - [x] resume support (`resume.enabled`, `resume.checkpoint_kind`) with explicit transparency logs/events on success or invalid resume state
  - [x] numeric parsing hardening for zero-valid fields (`lr`, `weight_decay`, scheduler values)
  - [x] run artifacts: per-attempt `training.log` and `runtime.json` + latest runtime mirror
  - [x] trainer Docker CUDA baseline moved to `cu129` for RTX 50-series (`sm_120`) compatibility
- [x] Post-training ONNX export flow:
  - [x] export best checkpoint via `torch.onnx.export` (opset 17)
  - [x] dynamic batch axes (`input`/`output`)
  - [x] per-run ONNX artifact location: `experiments/{project}/{experiment}/runs/{attempt}/onnx/model.onnx`
  - [x] ONNX metadata artifact (`onnx.metadata.json`) with class/preprocess/input-shape context
  - [x] ONNX + ONNXRuntime validation pass (dummy inference, batch size 1 and 4)
  - [x] SSE event emission for ONNX export completion/failure
  - [x] dependency updates: `onnx`, `onnxruntime`, `onnxscript`

### Docs
- [x] README aligned with implemented stack/workflow
- [x] Architecture doc aligned with code
- [x] Roadmap refreshed and feature requests recorded
- [x] Changelog maintained for major feature increments

## In Progress / Next

### Strategic Priorities (Tests + Consistency, Pre-MAL/Export Expansion)
- [x] Tests-first stabilization pass (before new MAL/model-export/curation feature work)
  - [x] add API + web regression coverage for stale submit contexts (`project/asset` churn leading to `404`)
  - [x] add MAL contract tests (queueing, suggestion persistence/retrieval, accept/reject lifecycle)
  - [x] add model-export contract tests (artifact generation, validation failures, deterministic behavior)
  - [x] add worker/trainer integration-path tests for queue -> execution -> persisted artifacts/events
- [x] Consistency hardening pass (targeted, no broad rewrite)
  - [x] standardize API error envelope usage across all routers (`api_error` with stable `error.code`)
  - [x] remove duplicated export selection/rebuild logic shared across `exports` and `experiments` flows
  - [x] keep project-scoped model records behavior stable while preparing migration from file-backed store to DB-backed table
  - [x] verify targeted regressions + full API test suite on Postgres-backed local test runtime
- [x] Surgical decomposition of complexity hotspots
  - [x] split `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx` into focused subcomponents/hooks without behavior changes
  - [x] split `apps/api/src/sheriff_api/routers/experiments.py` by concern (analytics/start-cancel/events/evaluation) to reduce regression risk
  - [x] ensure each extraction is paired with regression tests in the same PR

### Open Bugfixes
- [ ] Investigate intermittent `POST /projects/{project_id}/annotations` `404` in some submit flows:
  - [x] identify stale project/asset context transitions that can leave invalid staged entries
  - [x] harden submit path with stale-entry pruning and clearer user-facing error recovery
  - [x] add regression coverage around project/asset churn + staged submit
- [x] Fix label assignment persistence across all annotation modes:
  - [x] classification labels persist after submit
  - [x] bounding-box object class/category assignment persists
  - [x] segmentation object class/category assignment persists
  - [x] add reproducible regression tests covering import -> assign label/class -> submit -> reload
  - [x] document exact repro cases and resolution notes in `CHANGELOG.md`
  - [x] Bugfix execution steps:
    - [x] Step 1: add failing end-to-end regression tests first (classification, bbox, segmentation)
      - [x] Added API regressions in `apps/api/tests/test_api.py`:
        - [x] `test_regression_classification_preserves_label_when_classification_block_empty`
        - [x] `test_regression_bbox_preserves_class_from_object_when_classification_block_empty`
        - [x] `test_regression_segmentation_preserves_class_from_object_when_classification_block_empty`
      - [x] Baseline captured: all three regressions currently fail (label/class dropped to empty on persist)
    - [x] Step 2: capture/compare submit payload vs persisted payload to isolate drop point
      - [x] Confirmed drop happens in API normalization path (`apps/api/src/sheriff_api/services/annotation_payload.py:229`):
        - [x] when `classification` object exists, server reads only `classification.category_ids` / `classification.primary_category_id`
        - [x] top-level `category_ids`/`category_id` and geometry object categories are not used as fallback in that branch
      - [x] Regression evidence:
        - [x] submit payload includes top-level class (`category_ids=[id]`) while `classification` is empty
        - [x] persisted payload returns `category_ids=[]` (and null primary), causing apparent label loss after save/reload
      - [x] UI symptom amplification confirmed:
        - [x] committed selection reader prioritizes `classification` block first (`apps/web/src/lib/workspace/annotationState.js:16`)
        - [x] once API persists empty `classification`, labels appear to flicker/disappear on reload
    - [x] Step 3: make web payload building task-type-aware and deterministic for class/category fields
    - [x] Step 4: add server-side normalization fallback/guardrails for class/category persistence
    - [x] Step 5: run full API/web test suites + manual docker smoke checks before closing

### Refactor Workstream (Lean + Readable, No Behavior Loss)
- [x] Baseline code review completed with prioritized findings

#### P0 - Integrity Hardening (do first)
- [x] Guard annotation upsert by both `project_id` and `asset_id` (prevent cross-project updates)
- [x] Validate project existence before writing uploaded bytes
- [x] Ensure upload rollback cleanup when DB write fails after file write
- [x] Replace mutable schema defaults (for example `metadata_json = {}`) with safe factories
- [x] Add API tests for delete flows and integrity guards
- [x] Stabilize API async test harness in container/local so full suite passes reliably (loop/lifespan fixtures centralized in `conftest.py`)

#### P1 - Web Behavior Consistency
- [x] Fix submit gating so "clear label" (unlabeled submit) is possible in non-edit mode
- [x] Keep staged/selected label state transitions explicit and test-covered

#### P2 - Frontend Structure Refactor
- [x] Split `apps/web/src/app/page.tsx` into focused hooks/modules:
  - [x] `useImportWorkflow`
  - [x] `useDeleteWorkflow`
  - [x] `useAnnotationWorkflow`
  - [x] tree/pagination pure helpers in `apps/web/src/lib/workspace/*`
- [x] Move primary datasets workspace composition/render wiring to `ProjectAssetsWorkspace`
- [x] Add unit tests for extracted pure tree/pagination helpers

#### P3 - Hook Cleanup
- [x] Remove duplicated fetch logic in `useAssets` by centralizing loader function
- [x] Standardize error handling shape across hooks (`useProject`, `useLabels`, `useAssets`)

#### P4 - Regression Coverage
- [x] Replace placeholder hotkey test with real interaction tests
- [x] Add web integration tests for:
  - [x] number-key labeling (`1..9`, numpad)
  - [x] multi-delete selection/delete
  - [x] folder/subfolder delete behavior
  - [x] edit-mode stage/submit/clear flows

### API
- [x] Validate upload target project exists before persisting file
- [x] Populate image `width/height` on upload
- [x] Add richer structured API error responses for UI

### Web
- [x] Add explicit staged/dirty indicators in file tree and pagination
- [x] Improve import dialog UX (validation hints + remembered defaults)
- [x] Wire export button to backend flow
- [ ] Add export history/filter UI
- [ ] Add better loading/empty states per panel

### Testing
- [x] Add API tests for upload destination + relative path behavior
- [x] Add web integration tests for import -> label -> submit workflow
- [x] Add regression tests for edit mode + staged persistence
- [x] Add trainer/API/web coverage for experiment runtime/log features:
  - [x] trainer tests for eval cadence, BN small-batch guardrails, checkpoint payload policy, resume transparency, runtime/log artifact writes
  - [x] API tests for runtime endpoint, log-tail cursor behavior, and missing-artifact error codes
  - [x] web tests for runtime badge formatting/rendering and log-tail cursor helpers
- [x] Add ONNX coverage:
  - [x] trainer ONNX export test (artifact + dynamic batch validation path)
  - [x] API ONNX endpoint/download/not-found contract tests
  - [x] web ONNX summary helper tests

### Bounding Boxes + Segmentation (Design-First, End-to-End)
- [x] Lock annotation contract v2 (backward-compatible with current classification payloads):
  - [x] Keep current classification fields readable (`category_id`, `category_ids`, status flow)
  - [x] Add object-level entries for geometry with stable IDs per object
  - [x] Store geometry in image-pixel coordinates (`bbox` as `[x, y, width, height]`, polygons as flat point arrays)
  - [x] Persist per-asset canvas basis (`image_width`, `image_height`) for deterministic export math
- [x] API schema + validation hardening for geometry payloads:
  - [x] Add typed Pydantic models/unions for bbox and segmentation objects
  - [x] Validate project/asset/category consistency for every geometry object
  - [x] Validate geometry integrity (`>= 0`, within image bounds, valid polygon point count, non-empty instances)
  - [x] Return stable structured error codes for geometry validation failures
- [x] Web annotation-mode architecture:
  - [x] Replace static tab buttons with real mode state (`labels`, `bbox`, `segmentation`)
  - [x] Add viewer overlay layer with image-space <-> client-space transforms
  - [x] Add reusable geometry draft/selection state helpers under `apps/web/src/lib/workspace/*`
  - [x] Preserve existing classification edit-mode and staged-submit behavior unchanged
- [x] Bounding box workflow:
  - [x] Draw new box (drag gesture), select existing box, delete selected box
  - [x] Move existing box by drag
  - [x] Resize existing box using corner + edge-midpoint handles
  - [x] Assign/edit class per box using existing project labels
  - [x] Support multiple boxes per asset with clear active-object affordance
  - [x] Include keyboard affordances (`Esc` cancel draft, `Delete` remove selected object)
- [x] Segmentation workflow (polygon v1):
  - [x] Create polygon by click-to-add points, close by click-near-start, double-click, or `Enter`
  - [x] Select polygon and delete selected polygon
  - [x] Assign/edit class per polygon using existing project labels
  - [x] Include draft-cancel flow (`Esc`) and minimum-point guardrails
- [x] Annotation submit integration:
  - [x] Extend annotation submission helpers to emit classification/bbox/segmentation payloads
  - [x] Keep pending/staged behavior explicit across asset switches for geometry edits
  - [x] Extend tree/pagination dirty indicators to include geometry pending edits
  - [x] Ensure non-edit single submit and edit-mode batch submit both support geometry
- [x] COCO export upgrade:
  - [x] Export bbox instances into COCO `annotations[*].bbox`
  - [x] Export polygon instances into COCO `annotations[*].segmentation`
  - [x] Compute/populate COCO `area`, `bbox`, and `iscrowd` consistently for geometry records
  - [x] Keep deterministic hash/ordering guarantees for mixed classification + geometry exports
- [x] Testing block expansion:
  - [x] API tests:
    - [x] geometry upsert acceptance/rejection cases (invalid bounds, invalid polygons, wrong category/project)
    - [x] export payload correctness for bbox-only, seg-only, and mixed assets
    - [x] deterministic export hash stability for equivalent geometry payloads
  - [x] Web tests:
    - [x] unit tests for geometry math helpers (normalization, bbox/polygon area, hit-testing)
    - [x] integration tests for bbox draw -> label -> submit -> persisted reload
    - [x] integration tests for polygon draw -> label -> submit -> persisted reload
    - [x] regression tests confirming classification-only flows remain unchanged
- [x] Docs + release notes:
  - [x] Update `Architecture.md` (data contract, UI flows, export semantics)
  - [x] Update `README.md` (user workflow + known limitations)
  - [x] Update `Roadplan.md` milestone state for bbox/segmentation progress
  - [x] Add `CHANGELOG.md` entries for API, web, exporter, and test coverage increments

### Project Task Mode + Class Color System
- [x] Enforce per-project annotation task mode at API level:
  - [x] support explicit project task types for `classification`, `bbox`, and `segmentation`
  - [x] reject incompatible annotation payloads for project task mode with stable error codes
  - [x] add API tests for task-mode enforcement
- [x] Enforce per-project task mode at UI level:
  - [x] lock available annotation tabs/actions to the selected project task mode
  - [x] set task mode during new-project import flow
  - [x] keep existing projects backward-compatible
- [x] Class color mapping:
  - [x] deterministic per-class color assignment
  - [x] colorized label buttons/list rows by class
  - [x] colorized bbox/polygon overlays and geometry object list by class
  - [x] add tests for class-color helper determinism/range

## Deferred (Roadmap-aligned)
- [ ] Review/QA mode
- [ ] Video ingestion + frame extraction
- [ ] MAL integration beyond placeholders
- [ ] Reference-mode asset ingestion (cloud/object-store links)
- [ ] Shared asset library with project-specific annotations
