# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial end-to-end labeling workspace UI in `apps/web`.
- Persistent image upload pipeline:
  - `POST /api/v1/projects/{project_id}/assets/upload`
  - `GET /api/v1/assets/{asset_id}/content`
- Project-backed dataset browsing and hierarchical file tree.
- Import dialog supporting:
  - existing project destination
  - new project creation
  - existing folder/subfolder destination selection for existing-project imports
  - editable target folder name
- Automatic asset/tree refresh after import.
- Label management UI:
  - create labels
  - rename/reorder/activate/deactivate labels
- Edit mode with staged annotations and batch submit.
- Project-scoped multi-label toggle (managed in label manage mode).
- Arrow-key image navigation.
- Detailed import diagnostics for local file-read and network failures.
- Folder-scoped review from sidebar tree.
- Tree controls:
  - per-folder expand/collapse
  - collapse-all / expand-all actions
- Labeled/unlabeled visual status indicators in tree and pagination.
- Import progress telemetry in UI:
  - completed/total
  - uploaded/failed/remaining
  - bytes processed
  - transfer speed and ETA
- Export pipeline:
  - COCO-style export bundle generation on API
  - zip artifact persistence in storage
  - export download endpoint
  - web "Export Dataset" action wired to create + download
- Viewer queue navigation enhancements:
  - skip buttons (`-10`, `-5`, `<`, `>`, `+5`, `+10`)
  - dynamic pagination chip window sized by available width
  - `First` / `Last` quick jumps
- Delete capabilities:
  - `DELETE /api/v1/projects/{project_id}`
  - `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
  - web actions for project delete, single-image delete, multi-image delete, and folder/subfolder subtree delete
- Keyboard labeling shortcuts:
  - class selection with number keys `1..9` (top row and numpad)
- Toast-style operation feedback:
  - auto-dismiss success/error notifications
  - delete summaries with image/annotation counts
- Documentation refresh:
  - `README.md`
  - `Architecture.md`
  - `Roadplan.md`
  - `IMPLEMENTATION_TASKS.md`

### Changed
- Viewer image behavior now preserves aspect ratio with letterbox rendering (`object-fit: contain`) on black background.
- Viewer layout now keeps a bounded responsive viewport height to avoid pushing controls/panels off-screen.
- API CORS handling improved for localhost/127.0.0.1 development origins.
- API Docker packaging fixed for `src/` layout installation reliability.
- Local env defaults updated to conflict-resistant ports in `.env.example`.
- Tree panel now supports delete-mode multi-selection and per-folder delete controls.
- API async test harness now uses shared `conftest.py` fixtures with explicit FastAPI lifespan handling and deterministic asyncio loop-scope settings.
- Web annotation transitions now use explicit draft-vs-committed selection state logic, including non-edit clear-label submit eligibility.
- Web test script now runs focused state-transition tests via Node's built-in test runner.

### Fixed
- Root route prerender conflict caused by duplicate `/` page definitions.
- Import flow failures caused by weak file-type assumptions and missing diagnostics.
- Submit/edit-mode inconsistencies in annotation flow.
- Folder tree display ordering (hierarchy now preserved).
