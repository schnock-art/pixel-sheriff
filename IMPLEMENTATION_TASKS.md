# Implementation Tasks

Status reflects current repository behavior.

## Completed

### Infra / Compose
- [x] Docker compose services for `web`, `api`, `db`, `redis`
- [x] Configurable host ports via `.env`
- [x] API data volume mount (`./data -> /app/data`)
- [x] DB init script wiring

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
- [x] Export zip build (`manifest.json`, `annotations.json`, `images/`)
- [x] Export download endpoint
- [x] MAL placeholder endpoints (models/suggestions)
- [x] Local storage safety checks (path containment)
- [x] API packaging fixes for `src` layout Docker builds

### Web (`apps/web`)
- [x] Main workspace UI integrated at `/`
- [x] Responsive styling + viewer letterbox rendering
- [x] Bounded responsive viewport height (stable with large datasets)
- [x] Dataset sidebar with search
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
- [x] Annotation UX:
  - [x] edit mode staging
  - [x] batch submit staged changes
  - [x] non-edit single submit path
- [x] Keyboard navigation with arrow keys
- [x] Keyboard labeling shortcuts (`1..9`, top-row and numpad)
- [x] Delete UX:
  - [x] project delete action
  - [x] single-image delete action
  - [x] multi-image delete mode and selection
  - [x] folder/subfolder delete from tree
  - [x] toast-style delete summaries with counts

### Docs
- [x] README aligned with implemented stack/workflow
- [x] Architecture doc aligned with code
- [x] Roadmap refreshed and feature requests recorded
- [x] Changelog maintained for major feature increments

## In Progress / Next

### API
- [ ] Validate upload target project exists before persisting file
- [ ] Populate image `width/height` on upload
- [ ] Add richer structured API error responses for UI

### Web
- [ ] Add explicit staged/dirty indicators in file tree and pagination
- [ ] Improve import dialog UX (validation hints + remembered defaults)
- [x] Wire export button to backend flow
- [ ] Add export history/filter UI
- [ ] Add better loading/empty states per panel

### Testing
- [ ] Add API tests for upload destination + relative path behavior
- [ ] Add web integration tests for import -> label -> submit workflow
- [ ] Add regression tests for edit mode + staged persistence

## Deferred (Roadmap-aligned)
- [ ] Review/QA mode
- [ ] Video ingestion + frame extraction
- [ ] MAL integration beyond placeholders
- [ ] Reference-mode asset ingestion (cloud/object-store links)
- [ ] Shared asset library with project-specific annotations
