# pixel-sheriff

Local-first CV annotation platform.

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
  - project shell layout now handles project selector, top tabs, and project status bar
  - annotation, import, and delete workflows remain in dedicated hooks
  - tree/pagination/annotation-state logic moved into pure workspace helpers with unit tests
- Review-state visibility improved with explicit staged/dirty indicators in tree rows and pagination chips.
- Import dialog UX was upgraded with inline validation hints and remembered defaults for mode/project/folder destination.
- Testing coverage expanded with integration/regression suites for:
  - import -> label -> submit workflow
  - edit-mode staged state persistence across asset switches
- API upload now derives image `width`/`height` when detectable from uploaded bytes.
- API error responses now use a structured shape (`error.code`, `error.message`, `error.details`) for better UI diagnostics.
- Bounding box and polygon segmentation annotation tools are now wired end-to-end (draw/select/delete/submit/export).
- COCO export now includes geometry records (`bbox`, `segmentation`, `area`) for object annotations.
- Project-level task mode is now enforced (`classification_single`, `bbox`, `segmentation`) and selected during new-project import.
- Labels and geometry overlays now use deterministic class-based colors.
- Bounding-box interaction now supports move + resize (corner and edge-midpoint handles), plus inline draft warnings.
- Polygon closing is now more forgiving (`near-start`, double-click, or `Enter`) with draft-status guidance.
- Classification mode now includes explicit "Clear Selected Labels" and an assigned-label summary line.
- Export contract upgraded to v1.2:
  - `manifest.json` now includes explicit `tasks`, `label_schema`, `splits`, `training_defaults`, and `stats`
  - COCO companion file is now `coco_instances.json`
  - COCO and manifest now use the same canonical UUID asset IDs (`image_id == asset_id`)
  - class names are normalized to lowercase slug in model-facing/export fields
  - detection/segmentation exports now support explicit negative-image policy (`include_negative_images`)
  - detection COCO annotations now omit `segmentation` instead of emitting empty lists
- Project-scoped Phase 1 UI refactor is now implemented:
  - route structure under `/projects/{project_id}/...`
  - top navigation tabs: `Datasets`, `Models`, `Experiments`, `Deploy` (disabled placeholder)
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
  - Redis-queued trainer worker execution with persisted metrics/checkpoints
  - run-attempt isolation under `runs/{attempt}` to avoid metric/event/checkpoint mixing across restarts
  - live SSE stream consumed by web chart UI
  - chart includes axes, legend, toggles, and hover value tooltip
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
  - experiments list page now includes summary cards, multi-run metric chart, and hyperparameter scatter
  - experiment detail page now includes confusion matrix drill-down, per-class metrics, and prediction explorer
- Backend ML model-building layer (v0) is now implemented under `apps/api/src/sheriff_api/ml`:
  - extensible `ModelFactory` + family adapter registry
  - backbone metadata registry (`resnet18/34/50/101`) + verification utilities
  - generated web metadata file at `apps/web/src/lib/metadata/backbones.v1.json`
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
    - `/projects/{project_id}/models`
    - `/projects/{project_id}/models/new`
    - `/projects/{project_id}/models/{model_id}`
    - `/projects/{project_id}/experiments`
    - `/projects/{project_id}/experiments/new`
    - `/projects/{project_id}/experiments/{experiment_id}`
  - top shell project selector + tabs (`Datasets`, `Models`, `Experiments`, disabled `Deploy`)
  - workspace status line (`images labeled`, `classes`, `models`, `experiments`)
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
  - advanced runtime defaults to `num_workers=0` (editable)
  - save gating based on light config validation
  - checkpoint panel (`best_metric`, `best_loss`, `latest`) with selection placeholder action
  - metrics chart with axis/ticks/legend/toggles
  - hover crosshair + tooltip values per epoch
  - refresh-safe history rehydration and SSE resume while running
  - queued/running/terminal state handling and attempt-aware event cursors (`from_line`, `attempt`)
  - trainer failure reasons surfaced in UI toast and inline `Last run error`
  - header quick navigation action: `Back to Experiments`
- Experiment detail deep dashboard (classification) includes:
  - chart tabs (`Loss`, `Accuracy`, `F1/Precision/Recall`) + log scale
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
  - confusion-matrix normalization stays client-side:
    - `none`: raw counts
    - `by_true`: row-normalized
    - `by_pred`: column-normalized
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
  - project-scoped multi-label toggle (editable only in Manage Labels mode)
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
  - project task mode locks available annotation tabs/actions
  - Geometry class assignment uses the same project label set
  - Geometry edits participate in the same pending/edit-mode submit workflow as classification edits
  - Inline draft warnings clarify when geometry is not committed yet
- Dataset export (manifest v1.2 + COCO companion):
  - export record creation/listing
  - deterministic zip artifact with `manifest.json`, `coco_instances.json`, and `assets/`
  - one-click download from web UI
  - canonical join key is UUID `asset_id` across manifest and COCO
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
- Trainer reliability safeguards:
  - classification dataloader defaults to `advanced.num_workers=0` for container-safe execution
  - if a shared-memory dataloader failure occurs with `num_workers > 0`, trainer retries once with `num_workers=0`
- Backend ML model runtime (API internal library scope) remains available:
  - `build_model(config, verify_metadata=False)` and adapter registry in `apps/api/src/sheriff_api/ml`
  - metadata verification + registry generation utilities remain unchanged

## Run Locally

1. Copy env:

```bash
cp .env.example .env
```

2. Start:

```bash
docker compose up --build
```

Trainer CUDA defaults (Docker):
- trainer image installs `torch`/`torchvision` from `https://download.pytorch.org/whl/cu129` (includes `sm_120` support for RTX 50-series)
- trainer service requests GPU (`gpus: all`) and sets NVIDIA runtime envs
- override via `.env` if needed:
  - `TRAINER_GPUS=all|none`
  - `TRAINER_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129`
  - `NVIDIA_VISIBLE_DEVICES=all`
  - `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

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
  - `cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff STORAGE_ROOT=/tmp/pixel_sheriff_test_data python3 -m pytest -s tests`
- API tests (local, non-ML only):
  - `cd apps/api && python3 -m pip install -e ".[dev]"`
  - `cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff STORAGE_ROOT=/tmp/pixel_sheriff_test_data python3 -m pytest -s tests/test_api.py tests/test_model_store.py`
- API experiments-focused regressions (local):
  - `cd apps/api && DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff STORAGE_ROOT=/tmp/pixel_sheriff_test_data python3 -m pytest -s tests/test_experiments_api.py`
- ML tests only (local):
  - `cd apps/api && python3 -m pytest tests/ml -q --confcutdir=tests/ml`
- API tests (container):
  - `docker compose cp apps/api/tests api:/app/tests`
  - `docker compose exec api python3 -m pytest /app/tests -q`
  - note: API container image installs runtime deps only; install pytest/httpx in-container if needed before running this command
- Manual QA checklist for experiment SSE flow:
  - `docu/experiments_sse_manual_qa.md`

## ML Metadata Registry

- Backend ML metadata registry lives in `apps/api/src/sheriff_api/ml/metadata/backbones.py`.
- Web-consumable backbone registry JSON is generated at `apps/web/src/lib/metadata/backbones.v1.json`.
- API optional dependency group for ML runtime:
  - `cd apps/api && pip install -e ".[ml]"`
- Regenerate from repo root:
  - `cd apps/api`
  - `python -m sheriff_api.ml.metadata.generate_registry_json --out ../web/src/lib/metadata/backbones.v1.json`
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
- `GET/POST /api/v1/projects/{project_id}/categories`
- `PATCH /api/v1/categories/{category_id}`
- `GET/POST /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
- `GET /api/v1/assets/{asset_id}/content`
- `GET/POST /api/v1/projects/{project_id}/annotations`
- `GET/POST /api/v1/projects/{project_id}/exports`
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
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/logs` (byte-range tailing for `training.log`; returns `training_log_not_found` when unavailable)
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
- Deploy section is still a placeholder in the project shell
- Detection/segmentation trainer implementations are not yet available (classification only for now)
- MAL inference/generation quality pipeline is still minimal (queue + persistence contracts implemented; model inference integration pending)
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
