# Pixel Sheriff Architecture

Current codebase note: this file is a historical reference and may lag implementation details. For the current high-signal overview, start with `README.md` and `docs/architecture.md`.

## 1. System Overview

Pixel Sheriff is a local-first image annotation system with four active runtime layers:

- Web app (`apps/web`, Next.js App Router)
- API (`apps/api`, FastAPI + SQLAlchemy async)
- Trainer worker (`apps/trainer`, Redis queue consumer)
- Local infra services (Postgres + Redis via Docker Compose)

Primary workflow:

1. Select or create a project.
2. Import a local folder into a project/folder destination.
3. API persists image bytes to local storage and records asset metadata in Postgres.
4. Annotate in classification, bounding-box, or polygon-segmentation mode.
   - active mode is locked by selected task kind (`tasks.kind`) in the workspace task selector
5. Submit staged edits (or submit single-image edits directly).
6. Generate and download a deterministic dataset export zip (manifest v1.2 + COCO companion).
7. Create experiments from project models, tune training params, and run worker-executed training with live SSE metrics + checkpoints.
8. Compare project runs from analytics dashboard and inspect experiment evaluation artifacts:
   - classification: confusion matrix, per-class metrics, prediction explorer
   - detection: loss/mAP/runtime trends plus a detection summary card

Current frontend workspace decomposition note:
- `ProjectAssetsWorkspace` now acts as a composition/orchestration container.
- Sidebar/footer/import-modal/status-toast rendering are extracted into focused components under `apps/web/src/components/workspace/project-assets/`.
- Dataset-version route composition is split across:
  - `apps/web/src/app/projects/[projectId]/dataset/page.tsx`
  - `apps/web/src/lib/hooks/useDatasetPageState.ts`
  - `apps/web/src/components/workspace/dataset/*`
  - `apps/web/src/lib/workspace/datasetPage.*`
- Labeling workspace concern-specific orchestration is split across:
  - `apps/web/src/lib/hooks/useWorkspaceTaskState.ts`
  - `apps/web/src/lib/hooks/useWorkspaceSuggestions.ts`
  - `apps/web/src/lib/hooks/useProjectAssetsTreeState.ts`
  - `apps/web/src/components/workspace/project-assets/ProjectAssetsTaskModal.tsx`
- Web API client access is split across:
  - `apps/web/src/lib/api/client.js` for request primitives and `ApiError`
  - `apps/web/src/lib/api/types.ts` for shared request/response typing
  - domain modules under `apps/web/src/lib/api/*.ts`
  - `apps/web/src/lib/api.ts` as a compatibility barrel
- Workspace-specific local-storage settings, hotkey handling, and view-model derivations are extracted into dedicated hooks/helpers under `apps/web/src/lib/hooks/*` and `apps/web/src/lib/workspace/projectAssetsDerived.*`.

## 2. Runtime Topology

Compose services and default host ports:

- `web` -> `WEB_PORT` (default `3010`)
- `api` -> `API_PORT` (default `8010`)
- `trainer` -> worker + internal inference service (`TRAINER_INFERENCE_PORT`, default `8020`, container-network only)
- `db` -> `POSTGRES_PORT` (default `5433`)
- `redis` -> `REDIS_PORT` (default `6380`)

Persistence:

- DB data in `postgres_data` volume
- Binary assets + export zips in repo-local `./data` (mounted to `/app/data` in API and trainer containers)

Network behavior:

- Browser API calls use `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:${API_PORT}`)
- Next rewrite proxy remains configured in `apps/web/next.config.js`
- API CORS accepts configured origins plus localhost/127.0.0.1 regex

## 3. Backend Architecture (`apps/api`)

### App Composition

- Entry: `apps/api/src/sheriff_api/main.py`
- DB schema initialization runs on startup (`Base.metadata.create_all`)
- Routers mounted under `/api/v1`
- Global exception handlers normalize API failures into a structured envelope for UI consumption

### Data Model

Defined in `apps/api/src/sheriff_api/db/models.py`:

- Core: `Project`, `Task`, `Category`, `Asset`, `Annotation` (plus legacy SQL `DatasetVersion` model retained but not used as source-of-truth)
- Initial MAL domain: `Model`, `Suggestion`
- Project creation now auto-creates a default task (`tasks` table) and stores `projects.default_task_id`
- Categories, annotations, dataset versions, models, experiments, and deployments are task-scoped
- Dataset versions are file-backed and immutable:
  - `datasets/{project_id}/datasets.json`
  - implementation: `apps/api/src/sheriff_api/services/dataset_store.py`
- Project-scoped model drafts (Phase 1 scaffold) are stored via file-backed records under storage root:
  - `models/{project_id}/records.json`
  - implementation: `apps/api/src/sheriff_api/services/model_store.py`
- Project-scoped experiment runs are stored via file-backed records under storage root:
  - `experiments/{project_id}/records.json`
  - `experiments/{project_id}/{experiment_id}/config.json`
  - `experiments/{project_id}/{experiment_id}/status.json`
  - delete uses a temporary `.trash/` move under `experiments/{project_id}/` before record rewrite + final artifact removal
  - latest evaluation mirrors at experiment root:
    - `evaluation.json`
    - `predictions.jsonl`
    - `predictions.meta.json`
  - run-attempt scoped artifacts under `experiments/{project_id}/{experiment_id}/runs/{attempt}/...`
    - `run.json`
    - `events.jsonl` + `events.meta.json`
    - `metrics.jsonl`
    - `checkpoints.json` + checkpoint files
    - `evaluation.json`
    - `predictions.jsonl`
    - `predictions.meta.json`
  - implementation: `apps/api/src/sheriff_api/services/experiment_store.py`
- Experiment source-of-truth rule:
  - model configs own the canonical training data contract through `source_dataset`
  - experiment drafts default to that recorded dataset version
  - experiment create/start rejects runs whose selected dataset version diverges from the saved model source contract

Key invariants:

- One annotation row per `(asset_id, task_id)` (`uq_annotation_asset_task`)
- A project always has at least one task; deleting a task is blocked when references exist
- Category IDs are stable identities; label edits mutate name/order/active only
- Task kind (`tasks.kind`) governs accepted annotation payload shape:
  - `classification` / `classification_single`: no geometry objects
  - `bbox`: bbox objects only
  - `segmentation`: polygon objects only

### Storage

`LocalStorage` (`apps/api/src/sheriff_api/services/storage.py`) provides:

- per-project directory setup
- safe path resolution constrained to storage root
- byte writes and reads for uploaded assets/exports
- file and subtree deletion helpers for cleanup workflows

Upload endpoint (`/projects/{project_id}/assets/upload`) writes files as:

- `assets/{project_id}/{asset_uuid}{extension}`

Upload metadata stored in `assets.metadata_json`:

- `storage_uri`
- `original_filename`
- `relative_path`
- `size_bytes`

Additional upload-enriched fields persisted on `assets` rows:

- `width` and `height` (when dimensions can be inferred from uploaded bytes)

### Export Builder

`apps/api/src/sheriff_api/services/exporter_coco.py` builds deterministic export artifacts:

- `manifest.json`
- `coco_instances.json`
- `assets/...` (packaged asset files)

Current export contract highlights:

- canonical join key is UUID `asset_id` across manifest and COCO (`coco.image_id == asset_id`)
- manifest schema is `1.2` with explicit `tasks`, `label_schema`, `splits`, `training_defaults`, and `stats`
- label names are normalized to lowercase slug for export/model-facing fields (`label_schema.rules.names_normalized = lowercase_slug`)
- detection/segmentation exports support explicit negative-image policy via `selection_criteria_json.include_negative_images` (default `true`)
- classification exports emit empty COCO instance annotations by design
- detection COCO omits `segmentation` fields on bbox-only annotations
- join and geometry integrity checks run at export time (asset/class references, bbox/polygon validity, positive area)

Export files are persisted at:

- `exports/{project_id}/{content_hash}.zip`

### Project Model Export Artifacts

Project model export artifacts are persisted as deterministic JSON blobs at:

- `model_exports/{project_id}/{model_id}/{content_hash}.json`

Current behavior:

- export is available only when `config_json.export.onnx.enabled == true`
- export artifact hash is deterministic from canonicalized export payload content
- repeated exports for unchanged config resolve to the same hash/path

### ML Model Building Layer (`apps/api/src/sheriff_api/ml`)

Extensible model-building runtime is now implemented for `ModelConfig v1.0`:

- Public entrypoint:
  - `build_model(model_config: dict, verify_metadata: bool = False) -> BuiltModel`
  - file: `apps/api/src/sheriff_api/ml/model_factory.py`
- Family adapter registry:
  - maps `architecture.family` -> adapter builder
  - file: `apps/api/src/sheriff_api/ml/registry.py`
- Adapter families implemented (v0):
  - `resnet_classifier` (normalized primary output: `{"predictions": logits}`)
  - `retinanet` (normalized primary output wraps native detection list under `predictions`)
  - `ssdlite320_mobilenet_v3_large` (lighter MobileNetV3-based detector for lower-VRAM bbox training)
  - `deeplabv3` (normalized primary output uses segmentation `out` tensor under `predictions`)
- Family metadata contract:
  - `families.v1.json` now includes per-family `input_size` rules used by the web editor and API validation
  - `resnet_classifier`: square sizes >= `32`
  - `retinanet`: square sizes >= `224` in steps of `32`
  - `deeplabv3`: square sizes >= `224` in steps of `32`
  - `ssdlite320_mobilenet_v3_large`: fixed `320x320`
- Model/dataset task normalization:
  - dataset/task kinds are normalized before model-family matching (`bbox` -> `detection`, `classification_single` -> `classification`)
  - multi-label classification model defaults use `classification_bce_with_logits`
- Output composer:
  - composes primary + aux outputs
  - deterministic output tuple ordering from `export.onnx.output_names`
  - supports v0 aux projections: `none`, `pool_linear`, `mlp`
  - compatibility mapping: `projection.type = "linear"` -> `pool_linear` (avg pool)
  - optional aux normalization: `none` / `l2`
  - files: `apps/api/src/sheriff_api/ml/outputs/*`
- Tap handling:
  - `TapManager` supports module hooks and extractor-based taps
  - file: `apps/api/src/sheriff_api/ml/taps/manager.py`

Backbone metadata registry:

- Python dict + dataclasses in `apps/api/src/sheriff_api/ml/metadata/backbones.py`
- includes verified ResNet specs:
  - `resnet18`, `resnet34`, `resnet50`, `resnet101`
  - taps: `backbone.global_pool`, `backbone.c3`, `backbone.c4`, `backbone.c5`
- backward-compatible tap alias:
  - `backbone.avgpool` -> `backbone.global_pool`
- metadata verification helper:
  - `verify_backbone_meta(backbone_name)` performs lightweight forward-pass checks
  - file: `apps/api/src/sheriff_api/ml/metadata/verify.py`

Web-facing generated metadata:

- generated JSON file:
  - `packages/contracts/metadata/backbones.v1.json`
- generator entrypoint:
  - `make contracts-sync`
  - file: `apps/api/src/sheriff_api/ml/metadata/generate_registry_json.py`

Current integration scope:

- ML builder layer remains backend-internal for model config authoring/validation and metadata workflows.
- Existing project model CRUD endpoints still manage `config_json` draft lifecycle and schema validation only.

### Shared Model Utilities (`packages/pixel_sheriff_ml`)

Shared model helpers used across API/trainer runtime:

- `packages/pixel_sheriff_ml/src/pixel_sheriff_ml/model_factory.py`
  - `architecture_family(model_config)` for canonical family resolution
  - `build_resnet_classifier(model_config, num_classes_override=...)` for classification training model construction
- API experiment start flow uses shared family resolution when building queue jobs
- Trainer classification execution uses shared classifier build helpers

### Trainer Worker Architecture (`apps/trainer`)

Trainer responsibilities and runtime behavior:

- Worker entrypoint: `apps/trainer/src/pixel_sheriff_trainer/main.py`
- Queue contract: Redis `RPUSH/BLPOP` on `pixel_sheriff:train_jobs:v1`
- Job dispatcher: `apps/trainer/src/pixel_sheriff_trainer/runner.py`
- Internal inference service:
  - FastAPI app at `pixel_sheriff_trainer/inference/app.py`
  - endpoints: `POST /infer/classification`, `POST /infer/classification/warmup`
  - request path includes `model_key` + ONNX/metadata/asset relpaths and device preference
  - response returns `device_selected` and top-k class-index scores
- ONNXRuntime session cache (inference):
  - LRU + TTL policy with env knobs:
    - `INFERENCE_CACHE_MAX_MODELS_GPU`
    - `INFERENCE_CACHE_MAX_MODELS_CPU`
    - `INFERENCE_CACHE_TTL_SECONDS`
  - cache key: `(model_key, device_selected)` where `model_key` is ONNX SHA-256
  - per-entry lease counter (`in_use`) prevents eviction while requests are in-flight
  - saturation policy: fallback CUDA->CPU where possible; return `cache_busy` when no capacity remains
  - inference/preprocess work is offloaded with `asyncio.to_thread`
- Classification implementation:
  - export-zip dataset loader (`classification/dataset.py`)
  - epoch train/eval loops (`classification/train.py`, `classification/eval.py`)
  - evaluation artifact writer (`io/evaluation.py`) persists per-attempt + latest mirrors
  - evaluation artifacts now include provenance for project/experiment/attempt/model/task/dataset export identity
- Detection/segmentation data contract:
  - export zips use UUID `asset_id` as COCO `image_id`
  - trainer detection and segmentation loaders consume string `image_id` values directly and only remap integer COCO `category_id`
  - detection loader remaps exported COCO category ids according to family label semantics:
    - `retinanet` uses zero-based foreground labels
    - `ssdlite320_mobilenet_v3_large` uses one-based foreground labels because SSD reserves `0` for background
- Shared experiment-file contract:
  - reads/writes run-attempt files under `experiments/{project_id}/{experiment_id}/runs/{attempt}/...`
  - appends `events.jsonl` for SSE tail streaming
- Classification evaluation contract:
  - per-epoch `metrics.jsonl` now includes:
    - `train_accuracy`
    - `val_macro_f1`
    - `val_macro_precision`
    - `val_macro_recall`
    - `epoch_seconds`
    - `eta_seconds`
  - end-of-run classification artifacts include:
    - confusion matrix raw counts
    - per-class precision/recall/f1/support
    - prediction rows with `asset_id`, `relative_path`, `true_class_index`, `pred_class_index`, `confidence` (top-1 softmax), optional `margin` (top1-top2)
  - `predictions.meta.json` carries `schema_version`, `attempt`, `num_samples`, `task`, `split`, `computed_at`, and provenance identifiers
- Idempotency + attempt safety:
  - validates `status.json` active job/attempt before execution
  - stale/duplicate jobs are ignored
- Worker reliability safeguards:
  - default `TrainingConfig v0.1` uses `runtime.num_workers = 0` (fallback `advanced.num_workers`) for container-safe data loading
  - if a shared-memory dataloader failure occurs with `num_workers > 0`, trainer retries once with `num_workers=0` and emits a status message event
  - detection RetinaNet builder disables implicit pretrained-backbone downloads to keep trainer behavior deterministic offline
  - detection train loaders honor `training.drop_last` and default it to `true`; this prevents SSD Lite singleton-tail BatchNorm crashes on uneven train split sizes
  - SSD Lite preflights the effective train loader shape and raises `batchnorm_small_batch_unsupported` instead of failing inside PyTorch when a one-sample batch would be emitted
  - detection evaluation falls back to CPU automatically when CUDA `torchvision` detection kernels (for example NMS/postprocessing) are not available for the current GPU/runtime stack
  - ONNX export falls back from the default `torch.export` pipeline to legacy `torch.onnx.export(..., dynamo=False)` when torchvision detection postprocessing triggers data-dependent export guards
  - detection ONNX exports currently disable dynamic batch and are validated as fixed-batch (`batch=1`) graphs because exported torchvision detection postprocessing does not validate reliably under ONNX Runtime with dynamic batch enabled
  - detection ONNX validation is task-aware, so postprocessed outputs are not incorrectly rejected for lacking a batch-shaped leading dimension
  - runtime tuning fields supported in config:
    - `runtime.pin_memory`, `runtime.persistent_workers`
    - `runtime.prefetch_factor`, `runtime.cache_resized_images`, `runtime.max_cached_images`

## 4. Implemented API Surface

Health:

- `GET /api/v1/health`

Projects:

- `POST /api/v1/projects`
- `GET /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `DELETE /api/v1/projects/{project_id}`

Project `task_type` values supported by API:

- `classification` (legacy compatibility)
- `classification_single`
- `bbox`
- `segmentation`

Categories:

- `POST /api/v1/projects/{project_id}/categories`
- `GET /api/v1/projects/{project_id}/categories?task_id=...`
- `PATCH /api/v1/categories/{category_id}`
- `DELETE /api/v1/categories/{category_id}`

Tasks:

- `POST /api/v1/projects/{project_id}/tasks`
- `GET /api/v1/projects/{project_id}/tasks`
- `GET /api/v1/projects/{project_id}/tasks/{task_id}`
- `DELETE /api/v1/projects/{project_id}/tasks/{task_id}`
- delete is guarded by `task_not_empty` / `project_must_have_task` checks

Assets:

- `POST /api/v1/projects/{project_id}/assets`
- `GET /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/upload`
- `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
- `GET /api/v1/assets/{asset_id}/content`

Annotations:

- `POST /api/v1/projects/{project_id}/annotations` (upsert; payload includes `task_id`)
- `GET /api/v1/projects/{project_id}/annotations?task_id=...`

Dataset versions (file-backed source of truth):

- `GET /api/v1/projects/{project_id}/datasets/versions` (optional `task_id` filter)
- `POST /api/v1/projects/{project_id}/datasets/versions/preview` (requires `task_id` in body)
- `POST /api/v1/projects/{project_id}/datasets/versions` (requires `task_id` in body)
- `PATCH /api/v1/projects/{project_id}/datasets/active`
- `GET /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}`
- `GET /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/assets`
- `POST /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export`
- `GET /api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export/download`

Legacy exports transition:

- `POST /api/v1/projects/{project_id}/exports` -> `410 exports_legacy_gone`
- `GET /api/v1/projects/{project_id}/exports` -> `410 exports_legacy_gone`
- `GET /api/v1/projects/{project_id}/exports/{content_hash}/download` -> `410 exports_legacy_gone`

MAL contracts:

- `POST /api/v1/models`
- `GET /api/v1/models`
- `GET /api/v1/assets/{asset_id}/suggestions`
- `POST /api/v1/projects/{project_id}/suggestions/batch`
- `POST /api/v1/projects/{project_id}/suggestions/{suggestion_id}/accept`
- `POST /api/v1/projects/{project_id}/suggestions/{suggestion_id}/reject`
- `POST /api/v1/projects/{project_id}/deployments`
- `GET /api/v1/projects/{project_id}/deployments`
- `PATCH /api/v1/projects/{project_id}/deployments/{deployment_id}`
- `POST /api/v1/projects/{project_id}/predict`
- `POST /api/v1/projects/{project_id}/deployments/{deployment_id}/warmup`

Project model export contract:

- `POST /api/v1/projects/{project_id}/models/{model_id}/exports`
- `GET /api/v1/projects/{project_id}/models/{model_id}/exports/{content_hash}/download`

Project-scoped model scaffolding:

- `GET /api/v1/projects/{project_id}/models`
- `POST /api/v1/projects/{project_id}/models`
- `GET /api/v1/projects/{project_id}/models/{model_id}`
- `PUT /api/v1/projects/{project_id}/models/{model_id}`
- model creation derives deterministic `ModelConfig v1.0` from active/latest dataset-version snapshot in `DatasetStore` and validates against schema before persistence
- model updates validate incoming `config_json` against `ModelConfig v1.0` before persistence

Project-scoped experiments:

- `GET /api/v1/projects/{project_id}/experiments`
- `POST /api/v1/projects/{project_id}/experiments`
- `GET /api/v1/projects/{project_id}/experiments/analytics`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}`
- `PUT /api/v1/projects/{project_id}/experiments/{experiment_id}`
- `POST /api/v1/projects/{project_id}/experiments/{experiment_id}/start`
- `POST /api/v1/projects/{project_id}/experiments/{experiment_id}/cancel`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/events` (SSE)
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/evaluation`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/samples`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/runtime`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/logs`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/onnx`
- `GET /api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download?file=model|metadata`
- create flow derives default `TrainingConfig v0` from model + latest `DatasetVersion` and persists in `draft`
- start flow pins dataset export, creates run attempt metadata, enqueues Redis job, and transitions to `queued`
- experiment export split membership comes from stored dataset-version `splits.items` (saved version map), not live split recomputation
- SSE events stream by tailing run-attempt `events.jsonl` with optional resume cursor (`from_line`) and run selection (`attempt`)
- logs endpoint can tail a specific run attempt via `attempt`; responses include the served attempt so the web client can keep the viewer aligned with the current run
- router implementation is decomposed by concern under `apps/api/src/sheriff_api/routers/experiments/`:
  - `crud.py`
  - `analytics.py`
  - `evaluation.py`
  - `runs.py`
  - shared helpers/dependencies in `shared.py`
- default training config sets `advanced.num_workers = 0` (user-editable in Advanced Parameters)
- analytics endpoint returns multi-run `series` with `max_points` query support (default 200, bounded server-side)
- router registration keeps `/analytics` before `/{experiment_id}` to avoid path shadowing
- evaluation/samples endpoints default to latest completed attempt and include top-level `attempt` in responses
- logs endpoint defaults to the latest attempt with a log file, but the experiment detail UI requests `activeAttempt` explicitly and clears the visible buffer on run-attempt changes
- latest mirror files at experiment root are convenience snapshots; run-attempt artifacts remain source-of-truth

Error response contract:

- non-2xx API responses use:
  - `error.code` (stable machine-readable code)
  - `error.message` (human-readable message)
  - `error.details` (context including `request_path` and `request_method`)
- validation failures use `error.code = "validation_error"` and include `details.issues`

## 5. Web Architecture (`apps/web`)

App-router entry and project shell:

- `apps/web/src/app/page.tsx` redirects to `/projects`
- `apps/web/src/app/projects/page.tsx` resolves last/first project or shows empty-state create flow
- `apps/web/src/app/projects/[projectId]/layout.tsx` provides:
  - reusable project ribbon
  - project selector dropdown + create-project modal
  - global task selector + create-task modal entry
  - section tabs (`Labeling`, `Dataset`, `Models`, `Experiments`, `Deploy`)
  - project stats chips (`Images`, `Classes`, `Models`, `Experiments`)
  - guarded navigation for unsaved drafts
- `apps/web/src/app/projects/[projectId]/datasets/page.tsx` mounts the labeling workspace
- `apps/web/src/app/projects/[projectId]/dataset/page.tsx` mounts the dataset-version workspace (versions/preview/splits/export)
- `apps/web/src/app/projects/[projectId]/deploy/page.tsx` manages active deployment selection, device preference, deployment creation from experiments, and warmup
- `apps/web/src/app/projects/[projectId]/models/page.tsx` renders project-scoped model list/empty state + create flow
- `apps/web/src/app/projects/[projectId]/models/[modelId]/page.tsx` renders editable Model Builder controls with draft/save state, AJV validation, and live summary updates
- `apps/web/src/app/projects/[projectId]/experiments/page.tsx` renders experiment list plus analytics dashboard (summary cards, multi-run chart, hyperparameter scatter)
- `apps/web/src/app/projects/[projectId]/experiments/new/page.tsx` creates experiment drafts (auto when `modelId` query is provided)
- `apps/web/src/app/projects/[projectId]/experiments/[experimentId]/page.tsx` renders train workspace with editable params/checkpoints/live chart/SSE plus deep dashboard (confusion matrix, per-class metrics, prediction explorer) and a detail-page `Danger Zone` delete action

UI structure:

- Top: project ribbon (project/task context + workflow tabs)
- Left: asset browser with search/filter/import/delete controls
- Center: canvas toolbar + viewer canvas + filmstrip/navigation
- Right: annotation panel (task tabs, objects, classes, actions, MAL suggestions, shortcut hints)

### Frontend Data/State

Custom hooks:

- `useProject`: project listing
- `useTasks`: task list/create for active project
- `useAssets`: assets + annotations for active project + task
- `useLabels`: categories for active project + task
- `useDatasetPageState`: dataset-version route loading/mutation orchestration
- `useImportWorkflow`: import dialog, import progress, validation state, and remembered defaults/folder-option loading
- `useDeleteWorkflow`: single/bulk/folder/project delete flows
- `useAnnotationWorkflow`: staged vs direct submit, selection state, and submit gating
- `useWorkspaceTaskState`: active-task selection, task bootstrap, create-task modal state, and task-lock lookup
- `useWorkspaceSuggestions`: deployment loading and suggestion/MAL state
- `useProjectAssetsTreeState`: tree visibility, folder scope, sidebar filters, and asset index navigation

Web API modules (`apps/web/src/lib/api/*`):

- `client.js`: `ApiError`, fetch helpers, no-content handling, URI resolution
- `types.ts`: shared request/response contracts used by the web app
- `projects.ts`, `tasks.ts`, `categories.ts`, `assets.ts`, `annotations.ts`: project labeling domain clients
- `datasets.ts`, `models.ts`, `experiments.ts`, `deployments.ts`: dataset/model/training/deployment domain clients
- `legacy-exports.ts`: retained legacy export endpoints
- `paths.js`: query/path builders shared by dataset and experiment clients

Local persisted setting:

- per-project active task selection in `localStorage` (`pixel-sheriff:project-active-task:v1:{projectId}`)

Schema validation:

- AJV (`ajv` + `ajv-formats`) is the standard web validation layer for JSON Schema checks
- `apps/web/src/lib/schema/validator.ts` provides reusable schema compilation + AJV error normalization
- `packages/contracts/schemas/model-config-1.0.schema.json` is the canonical `ModelConfig v1.0` schema; synchronized copies remain under `apps/api` and `apps/web` for runtime validation

Workspace pure helpers (`apps/web/src/lib/workspace/*`):

- `tree.*`: relative-path normalization, folder tree construction, folder chain helpers
- `datasetPage.*`: dataset-version helper parsing, folder/status selection logic, and summary/class-name derivations
- `pagination.*`: width-aware chip capacity and page-token window generation
- `annotationState.*`: draft vs committed selection-state comparison and submit eligibility rules
- `hotkeys.*`: keyboard shortcut parsing/routing for navigation and label selection
- `deleteState.*`: pure selection/pruning helpers for bulk and folder-scope delete flows
- `annotationSubmission.*`: payload construction helpers for single/staged annotation submit paths
- `geometry.*`: image/viewport coordinate transforms, geometry math, and hit-testing helpers
- `importDialog.*`: import form validation and default-resolution helpers for mode/project/folder destination
- `importFiles.*`: image candidate filtering and import relative-path construction helpers
- `annotationWorkflowSelection.*`: current-asset selection resolution across pending vs committed annotation state
- `classColors.*`: deterministic class-to-color mapping for label chips and geometry overlays
- `projectRouting.*`: project route section parsing and project-target href generation
- `navigationGuard.*`: pure unsaved-draft guard decision helper
- `experimentAnalytics.*`: experiment analytics shaping helpers for comparison dashboard
- `experimentDashboard.*`: confusion normalization and prediction filtering helpers

Workspace container components (`apps/web/src/components/workspace/*`):

- `ProjectAssetsWorkspace.tsx`: labeling workspace composition and top-level wiring
- `ProjectNavigationContext.tsx`: unsaved-draft guard context and guarded navigation wrapper
- `ProjectCreateModal.tsx`: project creation modal used in shell and empty state
- `ModelBuilderSkeleton.tsx`: builder layout for model routes (stepper + editable center panel + summary + save-state actions)

Dataset workspace components (`apps/web/src/components/workspace/dataset/*`):

- `DatasetVersionsPanel.tsx`: saved-version selection and export actions
- `DatasetDraftPanel.tsx`: dataset draft form and include/exclude controls
- `DatasetAssetsPanel.tsx`: browse/preview asset list-grid rendering
- `DatasetSummaryPanel.tsx`: saved/draft summary rendering
- `DatasetFilterDropdowns.tsx`: folder/status multi-select controls

Labeling workspace leaf components (`apps/web/src/components/workspace/project-assets/*`):

- `ProjectAssetsTreeSidebar.tsx`: tree, scope, sidebar filters, and delete actions
- `ProjectAssetsFooterActions.tsx`: footer action cluster
- `ProjectAssetsImportModal.tsx`: import dialog
- `ProjectAssetsStatusOverlay.tsx`: status toast + import failure display
- `ProjectAssetsTaskModal.tsx`: create-task modal

### Implemented UX Behaviors

- Import dialog supports:
  - existing vs new project target
  - task-type selection when creating a new project
  - optional existing folder/subfolder target
  - editable folder path
  - inline field validation hints/errors
  - remembered defaults across sessions for mode/project/folder destination
- Import progress panel shows:
  - percent + completed/total files
  - bytes processed / total
  - upload speed + file rate
  - elapsed + ETA
  - uploaded/failed/remaining counts
- File tree behavior:
  - deterministic hierarchy from `relative_path`
  - per-folder expand/collapse + global collapse/expand
  - folder-scope review queue filtering
  - labeled/unlabeled indicators for files and folders
  - explicit staged/dirty badges for assets/folders with pending edits
  - folder/subfolder delete from tree
  - bulk delete mode (multi-select image removal within current scope)
- Viewer behavior:
  - bounded canvas with image `contain`
  - bounded responsive viewport height
  - keyboard `ArrowLeft`/`ArrowRight`
  - skip controls: `-10`, `-5`, `<`, `>`, `+5`, `+10`
  - geometry overlay for bbox/polygon rendering and drawing interactions
- Pagination behavior:
  - width-adaptive page-token window
  - `First`/`Last` chips
  - labeled/unlabeled color status
  - staged/dirty indicators for pages with pending edits
- Label panel behavior:
  - manage mode for create/rename/reorder/activate/deactivate
  - classification label mode is task-owned (`single_label` / `multi_label`) and surfaced as `Task Multi-label`
  - classification mode includes explicit `Clear Selected Labels`
  - classification mode surfaces assigned-label summary for current image
  - edit mode stages multi-asset changes
  - staged edits persist when switching assets and restore when returning to an asset with pending state
  - submit commits staged changes in batch
  - number-key shortcuts (`1..9`, top row and numpad) map to active label order
  - mode tabs switch between `labels`, `bbox`, and `segmentation`
  - mode tabs/actions are locked by selected task kind
  - in bbox/seg modes, class buttons assign the selected geometry object category
  - selected geometry object can be deleted from panel or keyboard (`Delete`)
- Geometry authoring behavior:
  - bbox mode: drag to draw, click to select, drag selected box to move, drag corner/edge handles to resize, `Esc` to cancel draft
  - segmentation mode (polygon v1): click to add points, close near start-point, by double-click, or with `Enter`; `Esc` to cancel draft
  - geometry edits join existing pending/edit-mode submit workflow
  - inline draft-status warnings are shown while geometry is uncommitted
- Model Builder behavior:
  - model detail keeps a local draft separate from saved `config_json`
  - draft is validated on each change using AJV + `ModelConfig v1.0` schema
  - Save is enabled only when draft is both changed and valid (`isDirty && isValid`)
  - editable v0 steps are implemented for Input, Backbone, Outputs (embedding aux), and Export
  - model summary updates live from draft edits
  - unsaved model edits integrate with project navigation guard behavior
  - successful save updates persisted config snapshot and shows toast feedback
  - `Train Model` CTA now resolves model experiments:
    - creates new run when none exists
    - otherwise offers `Continue` latest vs `New run`
- Experiment training behavior:
  - experiments list shows status and best metric summary per run
  - experiments list analytics section includes:
    - summary cards (best accuracy, lowest val loss, total runs, failures)
    - multi-run comparison chart with metric selector and log-scale toggle
    - hyperparameter scatter with hover tooltip (experiment identity + x/y values) and click-through to experiment detail
    - defaults: last 3 completed runs selected, failed runs hidden unless toggled, best run highlighted
  - experiment detail supports editable training params in `draft`/`failed` states
  - experiment detail header model name links to model detail page
  - experiment detail header includes `Back to Experiments` navigation to project experiments list
  - `Start Training` enqueues worker job and transitions `queued -> running -> terminal`
  - `Cancel` supports queued cancel and running cancel-request semantics
  - `Delete` is available only for `draft` / `failed` / `canceled` / `completed`; queued/running runs stay protected and non-archived deployment references block deletion
  - trainer pipelines poll `cancel_requested` between batches, so long detection/segmentation epochs no longer need to finish before cancellation takes effect
  - checkpoints tracked as `best_metric`, `best_loss`, `latest` with selection placeholder (`Pick`)
  - after a successful run, duplicate completed-run checkpoint files for the same epoch are compacted so metadata rows may share one artifact URI (`best_metric` > `best_loss` > `latest`)
  - metrics chart supports axis/ticks, legend, series toggles, crosshair hover, and per-epoch tooltip values
  - metrics panel shows last epoch duration + ETA + estimated finish clock time
  - experiment detail dashboard (classification) includes:
    - metric tabs (`Loss`, `Accuracy`, `F1/Precision/Recall`) with log-scale toggle
    - accuracy tab overlays `train_accuracy` and `val_accuracy`
    - timing row shows epoch duration + ETA + estimated finish clock time
    - confusion matrix heatmap with client-side normalization (`none`, `by_true`, `by_pred`)
    - confusion-cell drill-down modal with sample thumbnails
    - per-class metrics table with sortable columns
    - prediction explorer with mode/class filters and limit controls
    - sample image preview modal
    - served-attempt indicator from evaluation/samples APIs
  - experiment detail dashboard (detection) includes:
    - metric tabs (`Loss`, `mAP`, `Runtime`) with log-scale toggle
    - `Loss` view for `train_loss`
    - `mAP` view for `val_map` (`mAP@50`) and `val_map_50_95` (`mAP@50:95`)
    - `Runtime` view for `epoch_seconds` and `eta_seconds`
    - detection summary card with best/latest checkpoint, classes, validation split, and final `mAP@50` / `mAP@50:95`
  - segmentation experiments still show the dashboard placeholder
  - trainer failure details are surfaced in UI (toast + inline `Last run error` in experiment header)
  - refresh-safe behavior: persisted history loads first, then live stream resumes for running experiments
  - training log panel is attempt-scoped and displays the served run number so reruns do not look like one continuous log stream
- Feedback behavior:
  - auto-dismiss toast message for success/error summaries
  - delete summaries include removed image and annotation counts
- Navigation guard behavior:
  - when staged/pending edits exist, switching project/section/model builder prompts for discard confirmation
  - accepted navigation clears draft-guard state before transition

## 6. Annotation Payload Contract

Annotation payload is normalized to a backward-compatible v2 shape:

- `version: "2.0"`
- category/class IDs are UUID strings across top-level, classification block, and geometry objects
- legacy-compatible classification fields:
  - `type: "classification"`
  - `category_id` (primary label)
  - `category_ids` (multi-label-compatible)
- v2 classification block:
  - `classification.category_ids`
  - `classification.primary_category_id`
- geometry object list:
  - bbox object: `{ id, kind: "bbox", category_id, bbox: [x, y, width, height] }`
  - polygon object: `{ id, kind: "polygon", category_id, segmentation: [[x1, y1, ...]] }`
- optional `image_basis` (`width`/`height`) for deterministic geometry bounds
- `coco` compatibility block with `image_id` / `category_id`

Annotation payload normalization enforces selected-task compatibility before persistence and returns stable structured error codes on mismatch.

Supported statuses:

- `unlabeled`
- `labeled`
- `skipped`
- `needs_review`
- `approved`

## 7. Test Coverage

- Web test suite uses Node's built-in test runner (`apps/web/tests/*.test.js`).
- Coverage includes:
  - import -> label -> submit workflow composition checks
  - edit-mode staged state persistence regression checks
  - helper-level regressions for hotkeys, delete flows, tree/pagination, import defaults, annotation state transitions, and geometry math helpers
  - model builder helper/validator regressions for draft transforms, dirty checks, and AJV schema validation
- API test suite uses `pytest` with `httpx` ASGI client fixtures in `apps/api/tests`.
- API coverage includes geometry validation/COCO export assertions and project model update validation/persistence checks.
- API coverage includes stale submit-context regression checks (`project/asset` churn -> annotation submit `404`) and recovery-path assertions.
- API coverage includes experiment create/update/start/cancel flows and SSE smoke validation.
- API coverage includes MAL contract tests for:
  - batch suggestion queueing
  - suggestion persistence/retrieval by asset
  - accept/reject lifecycle transitions
- API coverage includes project model export contract tests for:
  - artifact generation + download
  - deterministic export hash behavior
  - validation/error behavior when export is disabled
- API coverage includes experiment analytics/evaluation/samples endpoint behavior:
  - analytics structure + `max_points`
  - evaluation `attempt` inclusion and `evaluation_not_found`
  - samples mode/filter behavior with `attempt` inclusion
- API coverage includes cross-boundary contract/regression checks for:
  - dataset-version responses validated against the web dataset schema
  - model-config responses validated against the web model-config schema
  - dataset preview -> dataset version create -> model draft flow
  - task-aware labeling -> active dataset -> model/experiment alignment
- Trainer coverage includes classification evaluation artifact persistence:
  - per-attempt + latest mirror evaluation/predictions files
  - `predictions.meta.json` contract checks
  - confusion matrix shape/per-class length/accuracy bounds
- Trainer coverage includes queue-path integration test (`raw queue payload -> parse -> runner -> persisted events/artifacts`).
- Web helper coverage includes analytics/dashboard helpers:
  - run selection, summary and scatter shaping
  - confusion normalization (`none`, `by_true`, `by_pred`) and zero-sum handling
  - prediction filtering helpers for explorer/drill-down
- Web/API seam verification should be run with `make verify-cross-boundary` or the underlying wrapper commands when `make` is unavailable.
- Web coverage includes stale-submit helper regressions for:
  - annotation submit `404` detection by route/status
  - staged pending-entry pruning for missing asset IDs
- ML-specific pytest coverage added under `apps/api/tests/ml`:
  - metadata verification vs real torchvision backbones (`resnet18`, `resnet50`)
  - registry JSON generation checks for expected structure/tap entries
  - `ModelFactory` classifier build + aux embedding output checks
  - compatibility/validation checks (`avgpool` alias, invalid taps, ONNX output-name mismatch)

## 8. Known Gaps

- Review/QA moderation workflow not implemented
- Geometry tooling polish pending (no polygon vertex dragging/editing yet)
- Auth/multi-user permissions not implemented
- MAL runtime v1 is implemented for classification ONNX (`deployments` + on-demand `/predict` via trainer inference service with session cache)
- MAL batch/curation inference workflows are still pending (`/predict/batch` and queue-driven bulk scoring)
- Detection and segmentation training pipelines are implemented, but deploy/evaluation parity is still uneven across tasks
- Classification training currently supports `resnet_classifier` family only in worker runtime
- Segmentation deep evaluation dashboards are still placeholders; classification and detection dashboards are implemented
- Stale submit contexts can still occur under aggressive project/asset churn, but web submit flow now prunes missing staged assets and surfaces explicit recovery messaging.
