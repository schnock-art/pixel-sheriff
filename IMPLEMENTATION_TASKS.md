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
- [x] Category endpoints (create/list/patch)
- [x] Asset endpoints:
  - [x] create/list
  - [x] multipart upload
  - [x] content stream
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
- [x] Dataset sidebar with search
- [x] Hierarchical file tree with folder/file navigation
- [x] Import dialog:
  - [x] existing vs new project
  - [x] target folder naming
- [x] Hierarchical file tree with preserved parent/child ordering
- [x] Robust import diagnostics:
  - [x] MIME + extension filtering fallback
  - [x] detailed per-file failure messages
- [x] Label panel features:
  - [x] add label
  - [x] manage labels (rename/reorder/activate/deactivate)
  - [x] multi-label toggle in edit mode
  - [x] add-label input visible only in manage mode
- [x] Annotation UX:
  - [x] edit mode staging
  - [x] batch submit staged changes
  - [x] non-edit single submit path
- [x] Keyboard navigation with arrow keys

### Docs
- [x] README aligned with implemented stack/workflow
- [x] Architecture doc aligned with code
- [x] Roadmap refreshed and feature requests recorded

## In Progress / Next

### API
- [ ] Validate upload target project exists before persisting file
- [ ] Populate image `width/height` on upload
- [ ] Add delete asset endpoint(s)
- [ ] Add richer structured API error responses for UI

### Web
- [ ] Add explicit staged/dirty indicators in file tree and pagination
- [ ] Improve import dialog UX (validation hints + remembered defaults)
- [ ] Add full keyboard labeling shortcuts (1..9, skip, etc.)
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
