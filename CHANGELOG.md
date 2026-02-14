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
  - editable target folder name
- Label management UI:
  - create labels
  - rename/reorder/activate/deactivate labels
- Edit mode with staged annotations and batch submit.
- Multi-label toggle (available in edit mode).
- Arrow-key image navigation.
- Detailed import diagnostics for local file-read and network failures.
- Export pipeline:
  - COCO-style export bundle generation on API
  - zip artifact persistence in storage
  - export download endpoint
  - web "Export Dataset" action wired to create + download
- Documentation refresh:
  - `README.md`
  - `Architecture.md`
  - `Roadplan.md`
  - `IMPLEMENTATION_TASKS.md`

### Changed
- Viewer image behavior now preserves aspect ratio with letterbox rendering (`object-fit: contain`) on black background.
- API CORS handling improved for localhost/127.0.0.1 development origins.
- API Docker packaging fixed for `src/` layout installation reliability.
- Local env defaults updated to conflict-resistant ports in `.env.example`.

### Fixed
- Root route prerender conflict caused by duplicate `/` page definitions.
- Import flow failures caused by weak file-type assumptions and missing diagnostics.
- Submit/edit-mode inconsistencies in annotation flow.
- Folder tree display ordering (hierarchy now preserved).
