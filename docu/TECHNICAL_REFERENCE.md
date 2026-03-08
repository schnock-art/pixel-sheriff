# Technical Reference

Detailed implementation notes, API surface, runtime behavior, and troubleshooting details.

For product overview and quickstart, see `README.md`.

## Most-Used Commands

If you use Make shortcuts:

```bash
make build-web-api
make up-web-api
```

Trainer optimization flow:

```bash
make build-trainer-base   # only when torch/cuda base changes
make build-trainer        # trainer only (fast iteration)
make build-trainer-bootstrap # one-time: base then trainer
make up-trainer
```

Other handy shortcuts:

```bash
make up
make down
make logs
make ps
make help
```

## What Changed Since Last Milestone

- Import UX was unified into one dialog with existing/new project targets plus optional existing folder/subfolder destination.
- Import now shows live progress (counts, bytes, speed, ETA) and clearer per-file failure diagnostics.
- Viewer/navigation improved with bounded responsive viewport height and skip controls (`-10`, `-5`, `<`, `>`, `+5`, `+10`).
- Pagination now adapts to available width and supports `First`/`Last` quick jumps.
- Export flow is fully wired from UI to API zip generation and direct download.
- Delete workflows were added:
  - project delete
  - single-image delete
  - multi-image delete
  - folder/subfolder delete (tree subtree scope)
- Delete actions now show confirmation summaries in an auto-dismiss toast with counts.
- Keyboard labeling shortcuts now support number keys (`1..9`) including numpad keys.
- Workspace internals were refactored:
  - root route now redirects to project-scoped routes
  - datasets workspace moved to `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`
  - datasets workspace composition was further decomposed into focused project-assets subcomponents/hooks:
    - `apps/web/src/components/workspace/project-assets/*`
    - `apps/web/src/lib/hooks/useProjectMultiLabelSettings.ts`
    - `apps/web/src/lib/hooks/useWorkspaceHotkeys.ts`
    - `apps/web/src/lib/workspace/projectAssetsDerived.*`
  - dataset-version route was reduced to composition-only wiring:
    - `apps/web/src/app/projects/[projectId]/dataset/page.tsx`
    - `apps/web/src/lib/hooks/useDatasetPageState.ts`
    - `apps/web/src/components/workspace/dataset/*`
    - `apps/web/src/lib/workspace/datasetPage.*`
  - labeling workspace orchestration was further decomposed by concern:
    - `apps/web/src/lib/hooks/useWorkspaceTaskState.ts`
    - `apps/web/src/lib/hooks/useWorkspaceSuggestions.ts`
    - `apps/web/src/lib/hooks/useProjectAssetsTreeState.ts`
    - `apps/web/src/components/workspace/project-assets/ProjectAssetsTaskModal.tsx`
  - project shell layout now handles project selector, top tabs, and project status bar
  - annotation, import, and delete workflows remain in dedicated hooks
  - tree/pagination/annotation-state logic moved into pure workspace helpers with unit tests
- Web API client internals were decomposed:
  - `apps/web/src/lib/api/client.js` now holds fetch/error/URI primitives
  - `apps/web/src/lib/api/types.ts` now holds shared request/response typing
  - domain modules now live under `apps/web/src/lib/api/` (`projects`, `tasks`, `categories`, `assets`, `annotations`, `datasets`, `models`, `experiments`, `deployments`)
  - `apps/web/src/lib/api.ts` remains a barrel so existing imports stay stable during the split
- Review-state visibility improved with explicit staged/dirty indicators in tree rows and pagination chips.
- Import dialog UX was upgraded with inline validation hints and remembered defaults for mode/project/folder destination.
- Testing coverage expanded with integration/regression suites for:
  - import -> label -> submit workflow
  - edit-mode staged state persistence across asset switches
- API upload now derives image `width`/`height` when detectable from uploaded bytes.
- API error responses now use a structured shape (`error.code`, `error.message`, `error.details`) for better UI diagnostics.
- Bounding box and polygon segmentation annotation tools are now wired end-to-end (draw/select/delete/submit/export).
- COCO export now includes geometry records (`bbox`, `segmentation`, `area`) for object annotations.
- Task mode is enforced (`classification`, `bbox`, `segmentation`) and selected via project tasks in workspace.
- Task-scoped workspace flow was wired end-to-end:
  - projects now include a default task and support creating additional tasks
  - categories/annotations/dataset versions/models/experiments/deployments are task-scoped
  - labeling workspace now supports task selection via `taskId` query and a `+ New Task` flow
- Labels and geometry overlays now use deterministic class-based colors.
- Bounding-box interaction now supports move + resize (corner and edge-midpoint handles), plus inline draft warnings.
- Polygon closing is now more forgiving (`near-start`, double-click, or `Enter`) with draft-status guidance.
- Classification mode now includes explicit "Clear Selected Labels" and an assigned-label summary line.
- Export contract upgraded to v1.2:
  - `manifest.json` now includes explicit `tasks`, `label_schema`, `splits`, `training_defaults`, and `stats`
  - COCO companion file is now `coco_instances.json`
  - COCO and manifest now use the same canonical UUID asset IDs (`image_id == asset_id`)
  - trainer detection/segmentation dataset loaders accept those UUID image IDs directly; only COCO `category_id` remains integer
  - class names are normalized to lowercase slug in model-facing/export fields
  - detection/segmentation exports now support explicit negative-image policy (`include_negative_images`)
  - detection COCO annotations now omit `segmentation` instead of emitting empty lists
- Project-scoped Phase 1 UI refactor is now implemented:
  - route structure under `/projects/{project_id}/...`
  - top navigation tabs: `Labeling`, `Dataset`, `Models`, `Experiments`, `Deploy`
  - project selector dropdown with `+ Create Project` modal
  - datasets `Build Model` CTA creates a project-scoped model draft from latest manifest and opens model detail
  - unsaved draft guard for project/tab/model navigation
- Model Builder v1.0 edit flow is now implemented:
  - editable steps for Input, Backbone, Outputs (embedding aux), and Export
  - AJV-based client-side schema validation on every draft edit
  - save gating by `isDirty && isValid`
  - `PUT /api/v1/projects/{project_id}/models/{model_id}` persists schema-valid config updates
- Experiments training flow (Phase 1) is now implemented end-to-end:
  - project-scoped experiments list + create + detail routes
  - model detail `Train Model` CTA supports `Continue` or `New run`
  - training config draft save/edit in `draft`/`failed` states
  - `start`/`cancel` lifecycle APIs
  - queued cancel is immediate; running cancel is cooperative and polled between training batches so long epochs do not trap the run in `running` until epoch end
  - Redis-queued trainer worker execution with persisted metrics/checkpoints
  - run-attempt isolation under `runs/{attempt}` to avoid metric/event/checkpoint mixing across restarts
  - live SSE stream consumed by web chart UI
  - chart includes axes, legend, toggles, and hover value tooltip
  - experiment dataset export now preserves saved dataset-version split membership (`train`/`val`/`test`) from stored `splits.items`
- Classification experiment analytics + deep dashboard:
  - trainer now writes attempt-scoped classification evaluation artifacts:
    - `experiments/{project_id}/{experiment_id}/runs/{attempt}/evaluation.json`
    - `experiments/{project_id}/{experiment_id}/runs/{attempt}/predictions.jsonl`
    - `experiments/{project_id}/{experiment_id}/runs/{attempt}/predictions.meta.json`
  - trainer also writes latest mirrors at experiment root:
    - `experiments/{project_id}/{experiment_id}/evaluation.json`
    - `experiments/{project_id}/{experiment_id}/predictions.jsonl`
    - `experiments/{project_id}/{experiment_id}/predictions.meta.json`
  - predictions rows now include `confidence` (top-1 softmax probability) and optional `margin` (`top1-top2`)
  - analytics API endpoint for multi-run comparison with `max_points` (default `200`, bounded server-side)
  - evaluation/samples API endpoints include served `attempt` in responses
  - evaluation and predictions metadata now include a `provenance` block with project/experiment/attempt/model/task/dataset-export identifiers
  - experiments list page now includes summary cards, multi-run metric chart, and hyperparameter scatter
  - experiment detail page now includes confusion matrix drill-down, per-class metrics, and prediction explorer
- Backend ML model-building layer (v0) is now implemented under `apps/api/src/sheriff_api/ml`:
  - extensible `ModelFactory` + family adapter registry
  - backbone metadata registry (`resnet18/34/50/101`) + verification utilities
  - generated canonical metadata file at `packages/contracts/metadata/backbones.v1.json`
  - pytest coverage for metadata verification, registry generation, and classifier build behavior
- Tests-first stabilization pass implemented:
  - stale submit context regressions added (API + web helper tests)
  - MAL contract tests added (queueing, persistence/retrieval, accept/reject lifecycle)
  - model export contract tests added (artifact generation, validation failures, deterministic hash)
  - trainer queue-path integration test added (`payload -> parse -> runner -> persisted events/artifacts`)
- Experiments API maintainability pass completed:
  - monolithic experiments router was decomposed into concern-focused modules under `apps/api/src/sheriff_api/routers/experiments/` (`crud`, `analytics`, `evaluation`, `runs`, `shared`)
  - dedicated API regression module added at `apps/api/tests/test_experiments_api.py` for decomposition-safety coverage
- Initial MAL + model-export API contracts are now implemented:
  - batch suggestion queueing and per-asset suggestion persistence
  - suggestion decision lifecycle (`accept` / `reject`)
  - deterministic project-model export artifact generation + download
- Annotation submit stale-context handling now prunes missing staged asset entries and shows explicit recovery messaging.

## Stack

- `apps/web`: Next.js labeling UI
- `apps/api`: FastAPI backend
- `apps/trainer`: Redis worker that executes classification training jobs
- Postgres + Redis via Docker Compose
- Local filesystem storage under `./data`

## Current Features

- Project-scoped workspace shell with route-based navigation:
  - `/` redirects to `/projects`
  - `/projects` resolves to the last/first project or shows create-project empty state
  - project routes:
    - `/projects/{project_id}/datasets`
    - `/projects/{project_id}/dataset`
    - `/projects/{project_id}/models`
    - `/projects/{project_id}/models/new`
    - `/projects/{project_id}/models/{model_id}`
    - `/projects/{project_id}/experiments`
    - `/projects/{project_id}/experiments/new`
    - `/projects/{project_id}/experiments/{experiment_id}`
    - `/projects/{project_id}/deploy`
  - top shell project selector + tabs (`Labeling`, `Dataset`, `Models`, `Experiments`, `Deploy`)
  - workspace status line (live labeled/class counts; model/experiment counts currently placeholder values in shell)
- Task management in labeling workspace:
  - projects auto-create a default task
  - task selector in workspace header (`taskId` query state)
  - `+ New Task` modal supports kind (`classification`/`bbox`/`segmentation`) and classification `label_mode`
  - task kind locks authoring mode tabs and submit payload shape
  - labels are task-scoped and lock once dataset versions exist for that task (`task_locked_by_dataset`)
- Dataset workspace (DatasetVersion v2):
  - immutable task-scoped file-backed dataset versions (`datasets/{project_id}/datasets.json`)
  - version preview + create flow with filter snapshot and seeded split config
  - active dataset-version pointer with per-version export endpoint
  - list endpoint supports optional `task_id` filtering
  - preview/list endpoints support pagination/filter/search without dumping full membership
  - experiment start export uses stored version split map (membership is not recomputed from live annotation state)
- Experiment create/start consistency:
  - experiment drafts default to the model's recorded `source_dataset.manifest_id`
  - experiment create/start rejects dataset versions whose recorded task/class/source contract diverges from the saved model config (`model_dataset_mismatch`)
  - to train against a newer dataset version, create or refresh the model from that dataset version first
- Category/class identity contract uses UUID strings across API/web/trainer/deploy metadata.
- Models pages support project-scoped create/list/detail plus editable model config drafting/saving
- Model export contract:
  - `POST /projects/{project_id}/models/{model_id}/exports` creates deterministic JSON export artifacts for enabled ONNX configs
  - `GET /projects/{project_id}/models/{model_id}/exports/{hash}/download` downloads export artifacts
- Model detail page includes:
  - editable controls for Input, Backbone, Outputs (embedding aux), and Export
  - live model summary updates as draft fields change
  - AJV (`ajv` + `ajv-formats`) validation against `ModelConfig v1.0`
  - Save enablement only when draft has changes and passes schema validation
  - unsaved changes indicator + guarded navigation integration
- Experiments pages support project-scoped list/create/detail plus live training telemetry
- Experiments list analytics dashboard includes:
  - summary cards (`best accuracy`, `lowest val loss`, `total runs`, `failures`)
  - multi-run comparison chart with metric selector + log scale
  - hyperparameter scatter with hover tooltip (experiment name + x/y values) and dot click-through to detail
- Experiment analytics defaults:
  - last 3 completed runs pre-selected
  - failed runs hidden by default (`Show failed` toggle)
  - best run highlighted in legend
- Experiment detail page includes:
  - editable training params (optimizer/lr/epochs/batch/augmentation/advanced)
  - advanced runtime controls:
    - `num_workers`, `pin_memory`, `persistent_workers`
    - `prefetch_factor`, `cache_resized_images`, `max_cached_images`
  - save gating based on light config validation
  - checkpoint panel (`best_metric`, `best_loss`, `latest`) with selection placeholder action
  - metrics chart with axis/ticks/legend/toggles
  - live timing row (`Last epoch time`, `ETA`, estimated finish clock time)
  - hover crosshair + tooltip values per epoch
  - refresh-safe history rehydration and SSE resume while running
  - queued/running/terminal state handling and attempt-aware event cursors (`from_line`, `attempt`)
  - trainer failure reasons surfaced in UI toast and inline `Last run error`
  - header quick navigation actions:
    - `Back to Experiments`
    - model name links to the model detail page
- Experiment detail deep dashboard (classification) includes:
  - chart tabs (`Loss`, `Accuracy`, `F1/Precision/Recall`) + log scale
  - accuracy tab overlays `train_accuracy` and `val_accuracy`
  - timing row (`Last epoch time`, `ETA`, estimated finish clock time)
  - confusion matrix with `none`/`by_true`/`by_pred` client-side normalization
  - confusion-cell drill-down modal with thumbnails
  - per-class metrics sortable table
  - prediction explorer filters (`mode`, `true class`, `pred class`, `limit`)
  - sample image preview modal
  - detection/segmentation placeholder (`not supported yet`)
- Classification evaluation artifact + API contract:
  - per-attempt artifacts are source-of-truth under `runs/{attempt}/...`
  - latest mirrors at experiment root are default API/UI read targets
  - `GET /experiments/{id}/evaluation` serves latest available evaluation and includes `attempt`
  - `GET /experiments/{id}/samples` supports `mode`, `true_class_index`, `pred_class_index`, `limit`, and includes `attempt`
  - `GET /experiments/{id}/logs` supports optional `attempt`, `from_byte`, and `max_bytes`; responses include the served `attempt`
  - experiment detail log viewer resets byte cursor/content on run-attempt changes instead of mixing prior-run output into the current view
  - evaluation and predictions metadata carry provenance for `project_id`, `experiment_id`, `attempt`, `model_id`, `task_id`, and dataset export identity
  - confusion-matrix normalization stays client-side:
    - `none`: raw counts
    - `by_true`: row-normalized
    - `by_pred`: column-normalized
- Detection trainer contract:
  - COCO `category_id` values remain integer in exports but RetinaNet training remaps them to zero-based foreground label indices (`0..num_classes-1`)
  - trainer RetinaNet construction disables implicit backbone weight downloads (`weights=None`, `weights_backbone=None`) for deterministic local/container runs
    - zero-sum rows/columns render `0` safely
- Local folder import with one modal:
  - import into existing or new project
  - select task mode when creating a new project (`classification_single`, `bbox`, `segmentation`)
  - optional existing folder/subfolder target for existing projects
  - editable destination folder name
  - inline validation hints/errors for project and folder fields
  - remembered import defaults across sessions (mode + target project + per-project folder preference)
- Import progress UI with:
  - completed/total count
  - uploaded/failed/remaining counts
  - bytes processed
  - throughput + ETA
- Robust import diagnostics:
  - extension fallback when MIME is missing
  - per-file read/network/API failure details
- Automatic asset/tree refresh after import (no page reload)
- Persistent upload storage + streamed image serving from API
- Upload metadata enrichment on API:
  - persisted `width`/`height` when image dimensions can be inferred
  - preserved `relative_path` (or filename fallback) for tree/export consistency
- Structured API errors with machine-readable codes:
  - response envelope: `error.code`, `error.message`, `error.details`
  - validation failures include per-field issues
- Hierarchical file tree (folders/subfolders/files), with:
  - preserved hierarchy ordering
  - per-folder expand/collapse
  - collapse all / expand all
  - folder-scope filtering (review a subtree only)
  - labeled/unlabeled status dots on folders and files
  - explicit staged/dirty badges for pending annotation edits
  - folder/subfolder delete actions
  - bulk delete mode with in-scope multi-select
- Viewer:
  - aspect-ratio-preserving black letterbox (`object-fit: contain`)
  - responsive bounded viewport height
  - keyboard navigation (`ArrowLeft` / `ArrowRight`)
  - skip navigation controls (`-10`, `-5`, `<`, `>`, `+5`, `+10`)
- Adaptive pagination:
  - dynamically sized page-chip window based on available width
  - `First` / `Last` chips
  - labeled/unlabeled chip coloring
  - staged/dirty page badges for pending edits
- Labels:
  - create labels
  - manage labels (rename/reorder/activate/deactivate)
  - classification label mode is task-owned (`single_label` / `multi_label`) and reflected in the panel
  - deterministic class-based colors for label rows/chips
  - clear selected labels action in classification mode
  - assigned-label summary (`Assigned: ...`) for immediate visual confirmation
- Annotation flow:
  - edit mode staging
  - staged edits persist while navigating between assets until submitted or reset
  - batch submit staged annotations
  - direct single-submit path when not staging
  - unsaved-draft leave guard on project/tab/model navigation
  - status values: `unlabeled`, `labeled`, `skipped`, `needs_review`, `approved`
  - keyboard label selection by class index (`1..9`, top row and numpad)
- Geometry tools:
  - Bounding box mode: draw by drag, select existing boxes, move by drag, resize via 8 handles (corners + edge midpoints), delete selected (`Delete`), cancel draft (`Esc`)
  - Segmentation mode (polygon v1): click to add vertices, close polygon near start-point, by double-click, or with `Enter`; delete selected and cancel draft (`Esc`)
  - selected task kind locks available annotation tabs/actions
  - Geometry class assignment uses the same project label set
  - Geometry edits participate in the same pending/edit-mode submit workflow as classification edits
  - Inline draft warnings clarify when geometry is not committed yet
- Dataset export (manifest v1.2 + COCO companion):
  - export is pinned to dataset versions (`POST /projects/{project_id}/datasets/versions/{dataset_version_id}/export`)
  - deterministic zip artifact with `manifest.json`, `coco_instances.json`, and `assets/`
  - one-click download from web UI
  - canonical join key is UUID `asset_id` across manifest and COCO
  - trainer loaders consume UUID string `image_id` values directly from COCO/manifest joins
  - classification exports keep `coco_instances.json` with empty `annotations`
  - detection/segmentation exports include geometry records with validated `bbox`/`segmentation` and computed `area`
  - configurable detection/segmentation negative-image policy through `selection_criteria_json.include_negative_images` (`true` by default)
- MAL contract (initial):
  - global model registry endpoints (`POST/GET /api/v1/models`)
  - project batch queueing endpoint (`POST /api/v1/projects/{project_id}/suggestions/batch`)
  - per-asset suggestion retrieval endpoint (`GET /api/v1/assets/{asset_id}/suggestions`)
  - suggestion decision endpoints:
    - `POST /api/v1/projects/{project_id}/suggestions/{suggestion_id}/accept`
    - `POST /api/v1/projects/{project_id}/suggestions/{suggestion_id}/reject`
- Shared ML model utilities:
  - `packages/pixel_sheriff_ml` provides shared helpers used by API + trainer (`architecture_family`, `build_resnet_classifier`)
  - trainer uses shared classifier builder for real classification runs
  - API experiment queue/start flow uses shared architecture-family resolution
  - model config defaults normalize dataset task kinds before family selection (`bbox` -> `detection`, `classification_single` -> `classification`)
  - multi-label classification defaults now use `loss.type = "classification_bce_with_logits"` and that loss is part of the shared `ModelConfig` schema
- Trainer reliability safeguards:
  - classification dataloader defaults to `runtime.num_workers=0` (falls back to `advanced.num_workers`) for container-safe execution
  - if a shared-memory dataloader failure occurs with `num_workers > 0`, trainer retries once with `num_workers=0`
  - runtime loader tuning is available via config/UI:
    - `runtime.num_workers`, `runtime.pin_memory`, `runtime.persistent_workers`
    - `runtime.prefetch_factor`, `runtime.cache_resized_images`, `runtime.max_cached_images`
  - trainer metrics now include:
    - `train_accuracy`, `epoch_seconds`, `eta_seconds`
  - runtime payload now includes:
    - `prefetch_factor`, `cache_resized_images`, `max_cached_images`
- Backend ML model runtime (API internal library scope) remains available:
  - `build_model(config, verify_metadata=False)` and adapter registry in `apps/api/src/sheriff_api/ml`
  - metadata verification + registry generation utilities remain unchanged

## Run Locally

1. Copy env:

```bash
cp .env.example .env
```

2. Start (normal day-to-day):

```bash
docker compose up -d
```

Only rebuild when needed:

```bash
docker compose --profile build-tools build trainer-base   # when torch/cuda base changes
docker compose build trainer api web                      # day-to-day image rebuilds
docker compose up -d
```

If you want a full rebuild from Dockerfiles:

```bash
docker compose up --build
```

Trainer CUDA defaults (Docker):
- trainer image installs `torch`/`torchvision` from `https://download.pytorch.org/whl/cu129` (includes `sm_120` support for RTX 50-series)
- trainer service requests GPU (`gpus: all`) and sets NVIDIA runtime envs
- override via `.env` if needed:
  - `TRAINER_GPUS=all|none`
  - `TRAINER_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129`
  - `TRAINER_BASE_IMAGE=python:3.11-slim` (or your prebuilt base image tag)
  - `NVIDIA_VISIBLE_DEVICES=all`
  - `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

Optional: prebuild a reusable trainer CUDA base image (recommended):

```bash
docker compose --profile build-tools build trainer-base
```

Then set `.env`:

```bash
TRAINER_BASE_IMAGE=${TRAINER_BASE_TAG}
```

After that, `docker compose build trainer` reuses the prebuilt CUDA/PyTorch layer and only rebuilds app code layers.
Also, trainer Dockerfiles now share a named BuildKit pip cache (`id=pixel-sheriff-pip-cache`) so wheel downloads are reused across trainer/base builds.

3. Open (`.env.example` defaults):

- Web: `http://localhost:3010`
- API base: `http://localhost:8010/api/v1`
- API docs: `http://localhost:8010/docs`

## Useful Commands

- Start/rebuild: `docker compose up --build`
- Stop: `docker compose down`
- Logs: `docker compose logs -f web api trainer`
- Status: `docker compose ps`
- Web tests: `cd apps/web && npm test`
- Web build check: `cd apps/web && npm run build`
- API tests (local, full suite):
  - `docker compose up -d db redis`
  - `cd apps/api && python3 -m pip install -e ".[dev,ml]"`
  - `cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test STORAGE_ROOT=/tmp/pixel_sheriff_test_data python3 -m pytest -s tests`
- API tests (local, non-ML only):
  - `cd apps/api && python3 -m pip install -e ".[dev]"`
  - `cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test STORAGE_ROOT=/tmp/pixel_sheriff_test_data python3 -m pytest -s tests/test_api.py tests/test_model_store.py`
- API experiments-focused regressions (local):
  - `cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test STORAGE_ROOT=/tmp/pixel_sheriff_test_data python3 -m pytest -s tests/test_experiments_api.py`
- Safety note: never point test commands at your app DB (`pixel_sheriff`), because test fixtures reset schema/tables.
- ML tests only (local):
  - `cd apps/api && python3 -m pytest tests/ml -q --confcutdir=tests/ml`
- API tests (container):
  - `docker compose cp apps/api/tests api:/app/tests`
  - `docker compose exec api python3 -m pytest /app/tests -q`
  - note: API container image installs runtime deps only; install pytest/httpx in-container if needed before running this command
- Cross-boundary verification:
  - `make verify-cross-boundary`
  - if `make` is unavailable, run:
    - `python3 scripts/sync_contract_artifacts.py --check`
    - `./scripts/typecheck_web.sh`
    - `./scripts/run_web_tests.sh tests/apiClient.test.js`
    - `./scripts/run_api_tests.sh -q tests/test_cross_boundary_contracts.py`
- Environment note:
  - in this repo/environment, host-side direct API `pytest` can be unreliable because local async DB drivers may stall
  - prefer `./scripts/run_api_tests.sh` or `make test-api-focused` / `make test-api-safe` so tests run against the Docker-backed Postgres test path
  - `./scripts/run_api_tests.sh` rebuilds the `api-test` image on invocation so copied source/tests stay current
  - for frontend structure refactors, run both `./scripts/run_web_tests.sh` and `cd apps/web && npx tsc --noEmit`
  - for web API client changes, add focused helper coverage in `apps/web/tests/apiClient.test.js`
  - for cross-boundary contract/regression coverage, keep `apps/api/tests/test_cross_boundary_contracts.py` green
- Manual QA checklist for experiment SSE flow:
  - `docu/experiments_sse_manual_qa.md`

## ML Metadata Registry

- Backend ML metadata registry lives in `apps/api/src/sheriff_api/ml/metadata/backbones.py`.
- Canonical shared contract artifacts live in `packages/contracts`.
- Web-consumable runtime copies are synchronized from `packages/contracts/metadata/*.json`.
- API optional dependency group for ML runtime:
  - `cd apps/api && pip install -e ".[ml]"`
- Regenerate + sync from repo root:
  - `make contracts-sync`
- Verify drift from repo root:
  - `make contracts-check`
- Add new backbones/families by:
  - extending `BACKBONES` metadata in backend
  - implementing a new adapter + family registry entry
  - adding metadata verification tests under `apps/api/tests/ml`

`docker compose logs ...` only reads logs; it does not start containers.

## API Surface (Implemented)

- `GET /api/v1/health`
- `GET/POST /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `DELETE /api/v1/projects/{project_id}`
- `GET/POST /api/v1/projects/{project_id}/tasks`
- `GET /api/v1/projects/{project_id}/tasks/{task_id}`
- `DELETE /api/v1/projects/{project_id}/tasks/{task_id}`
- `GET /api/v1/projects/{project_id}/categories?task_id=...`
- `POST /api/v1/projects/{project_id}/categories`
- `PATCH /api/v1/categories/{category_id}`
- `DELETE /api/v1/categories/{category_id}`
- `GET/POST /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
- `GET /api/v1/assets/{asset_id}/content`
- `GET /api/v1/projects/{project_id}/annotations?task_id=...`
- `POST /api/v1/projects/{project_id}/annotations` (body includes `task_id`)
- Dataset versions:
  - `GET /api/v1/projects/{project_id}/datasets/versions` (optional `task_id`)
  - `POST /api/v1/projects/{project_id}/datasets/versions/preview` (requires `task_id`)
  - `POST /api/v1/projects/{project_id}/datasets/versions` (requires `task_id`)
  - `PATCH /api/v1/projects/{project_id}/datasets/active`
  - `GET /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}`
  - `GET /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/assets`
  - `POST /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export`
  - `GET /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export/download`
- Legacy exports transition endpoints remain for compatibility and currently return `410 exports_legacy_gone`:
  - `POST /api/v1/projects/{project_id}/exports`
  - `GET /api/v1/projects/{project_id}/exports`
  - `GET /api/v1/projects/{project_id}/exports/{content_hash}/download`
- `GET/POST /api/v1/projects/{project_id}/models`
- `GET /api/v1/projects/{project_id}/models/{model_id}`
- `PUT /api/v1/projects/{project_id}/models/{model_id}`
- `POST /api/v1/projects/{project_id}/models/{model_id}/exports`
- `GET /api/v1/projects/{project_id}/models/{model_id}/exports/{content_hash}/download`
- `GET/POST /api/v1/projects/{project_id}/experiments`
- `GET /api/v1/projects/{project_id}/experiments/analytics` (`max_points` query, default `200`, range `1..2000`)
- `GET/PUT /api/v1/projects/{project_id}/experiments/{experiment_id}`
- `POST /api/v1/projects/{project_id}/experiments/{experiment_id}/start`
- `POST /api/v1/projects/{project_id}/experiments/{experiment_id}/cancel`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/events` (SSE)
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/evaluation` (includes top-level `attempt`; returns `evaluation_not_found` when unavailable)
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/samples` (supports mode/class filters + `limit`; includes top-level `attempt`)
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/runtime` (latest runtime info; returns `runtime_not_found` when unavailable)
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/logs` (byte-range tailing for `training.log`; supports optional `attempt`; response includes served `attempt`; returns `logs_not_found` when unavailable)
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/onnx`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download?file=model|metadata`
- `GET/POST /api/v1/models`
- `GET /api/v1/assets/{asset_id}/suggestions`
- `POST /api/v1/projects/{project_id}/suggestions/batch`
- `POST /api/v1/projects/{project_id}/suggestions/{suggestion_id}/accept`
- `POST /api/v1/projects/{project_id}/suggestions/{suggestion_id}/reject`

## Runtime Notes

- Web uses `NEXT_PUBLIC_API_BASE_URL` for browser calls.
- Next.js rewrite proxy (`/api/v1/* -> INTERNAL_API_BASE_URL`) remains available.
- Default local ports are conflict-resistant and configurable in `.env`.
- Suggestion queue key is configurable via `SUGGESTION_QUEUE_KEY` (default `pixel_sheriff:suggest_jobs:v1`).

## Known Gaps

- Review/QA workflow
- Geometry tooling polish (no polygon vertex dragging/edit yet)
- Deploy UX is functional for classification ONNX but still early-stage (limited controls and error guidance)
- Detection/segmentation trainer implementations are not yet available (classification only for now)
- MAL batch/curation pipeline is still minimal (single-asset deploy inference is implemented; batch queue scoring is still pending)
- Shared-asset reference mode (upload-once/link-many)

## Active Known Issue

- Stale submit contexts can still occur after aggressive project/asset churn, but staged state now prunes missing asset entries and surfaces explicit recovery guidance.

## Troubleshooting

- Import `NetworkError`:
  - `curl http://localhost:8010/api/v1/health`
  - `docker compose logs --tail=200 api web`
  - `docker compose up --build web api`

- Local read failures (`AbortError`/`NotReadableError`):
  - Files are often cloud placeholders (for example OneDrive "online-only")
  - Move/sync images to a true local directory and re-import

- Annotation submit `404`:
  - Usually means the selected project/asset pair no longer matches server state.
  - Current behavior prunes stale staged entries automatically and shows a recovery message.
  - If needed, refresh the page, reselect the project, and verify the asset is still present in the tree before submitting again.
