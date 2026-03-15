# VLM Cold-Start Prelabeling v1

Status tracker for the sequence-first bbox-only prelabel workflow described on 2026-03-11.

## Scope

v1 scope:
- bbox tasks only
- pending AI boxes stored as `PrelabelProposal`
- accepted or edited proposals promoted into normal annotation payloads
- sources: `active_deployment`, `florence2`
- video auto-start after extraction
- webcam live enqueue during capture, explicit close-input on finish

## Assessment Before This Pass

Already implemented in repo when this pass started:
- `PrelabelSession` and `PrelabelProposal` DB models plus startup migration
- API routes for create/get/list proposals/accept/reject/close-input/cancel
- queue separation for prelabel jobs
- video import and webcam session payloads accept `prelabel_config`
- webcam frame upload enqueues sampled live prelabel jobs
- video extraction enqueues post-extraction prelabel jobs
- annotation payload provenance schema + backend normalization preservation
- trainer Florence warmup/detect endpoints and API client methods
- worker handler for `prelabel_asset`
- per-frame and per-sequence `pending_prelabel_count` fields in sequence responses
- shared `PrelabelSettingsSection` component
- `usePrelabels` hook and API client module

Gaps found before implementation:
- labeling workspace did not surface AI prelabels at all
- pending proposal overlay was not rendered on the viewer
- sequence UI did not expose pending-frame navigation/counts
- video/webcam modals were not wired to bbox-only prelabel controls from the workspace
- no focused regression tests for accept/idempotent merge or edited-proposal sync
- no task tracker doc for this feature

## Tasks

### Backend
- [x] Keep separate `prelabel_sessions` and `prelabel_proposals` storage
- [x] Preserve object-level provenance in annotation payload normalization
- [x] Add public prelabel session/proposal review API
- [x] Extend video import and webcam session creation with `prelabel_config`
- [x] Enqueue live webcam sampled frames while input is open
- [x] Auto-start video prelabels after frame extraction
- [x] Mark proposal status from saved provenance-backed annotation objects
- [x] Add dedicated prelabel queue key/worker path
- [x] Add Florence trainer warmup/detect endpoints
- [x] Resolve active deployment at session creation
- [x] Keep case-insensitive strict task-class label matching
- [x] Add adapter registry lookup for prelabel source types
- [x] Log unmatched detections while counting/skipping them

### Frontend
- [x] Share prelabel settings between video and webcam flows
- [x] Show prelabel controls only for bbox tasks
- [x] Default Florence prompts to task class names while allowing edits
- [x] Hide prompts for project-model mode and use active deployment implicitly
- [x] Add dedicated AI Prelabels review panel in labeling workspace
- [x] Render pending proposals as read-only dashed overlay boxes with AI badge
- [x] Support Accept selected / Reject selected / Accept frame / Reject frame / Accept session / Edit selected
- [x] Prefer reviewed bbox/category state over original proposal state anywhere prelabels are rendered, edited, or accepted
- [x] Show prelabel progress and pending counts in the sequence UI
- [x] Add "next frame with pending AI" navigation

### Tests
- [x] API regression for video import + `prelabel_config` session creation
- [x] API regression for accept merge idempotency
- [x] API regression for edited proposal sync on annotation save
- [x] Web regression for provenance preservation in annotation state normalization
- [x] Coverage for Florence normalization edge cases with fake trainer responses
- [x] Coverage for multi-camera webcam close-input completion path
- [x] Coverage for full workspace review interactions

## Files Changed In This Pass

Backend:
- `apps/api/src/sheriff_api/services/prelabel_adapters.py`
- `apps/api/src/sheriff_api/services/prelabels.py`
- `apps/api/src/sheriff_api/demo_prelabel_seed.py`
- `apps/api/tests/test_prelabels_api.py`

Frontend:
- `apps/web/src/components/workspace/ProjectAssetsWorkspace.tsx`
- `apps/web/src/components/workspace/project-assets/AiPrelabelsPanel.tsx`
- `apps/web/src/components/workspace/project-assets/SequenceTimeline.tsx`
- `apps/web/src/components/workspace/project-assets/SequenceThumbnailStrip.tsx`
- `apps/web/src/components/workspace/project-assets/SequenceToolbar.tsx`
- `apps/web/src/components/Viewer.tsx`
- `apps/web/src/lib/hooks/usePrelabels.ts`
- `apps/web/src/lib/hooks/useWebcamCapture.ts`
- `apps/web/src/lib/hooks/useSequenceNavigation.ts`
- `apps/web/src/lib/workspace/webcamCaptureFinish.js`
- `apps/web/src/app/globals.css`
- `apps/web/tests/annotationState.test.js`
- `apps/web/tests/webcamCaptureFinish.test.js`
- `apps/web/tests/demo/prelabels-review.spec.mjs`

Demo / harness:
- `scripts/demo/seed-demo-project.mjs`
- `scripts/demo/run-prelabels.mjs`
- `scripts/run_demo_assets.sh`

Docs:
- `docu/VLM_COLD_START_PRELABELING_TASKS.md`

## Coverage Added In This Completion Pass

- `apps/api/tests/test_prelabels_api.py`
  - Florence malformed-row skipping and clamp/reorder normalization through the live prelabel job path
  - webcam `close-input` completion semantics for multiple live sessions with queued work still draining
- `apps/trainer/tests/test_inference_app.py`
  - `_parse_florence_generation` ignores malformed Florence payload rows without raising
- `apps/web/tests/webcamCaptureFinish.test.js`
  - webcam finish helper drains uploads, dedupes session ids, closes prelabel input, then tears down preview
- `apps/web/tests/demo/prelabels-review.spec.mjs`
  - seeded AI prelabel review workflow across overlay selection, edit/save, accept-frame, next-pending navigation, and reject-selected flows
- `apps/api/src/sheriff_api/demo_prelabel_seed.py`
  - deterministic bbox sequence + completed prelabel session + pending proposals for the Playwright demo harness

## Current Outcome

Feature status after this pass:
- Florence malformed responses are skipped consistently in both the trainer parser and API adapter path
- webcam finish now drains already-started uploads before closing live prelabel input and tearing down preview streams
- labeling workspace review interactions now have deterministic demo seed data plus dedicated browser automation coverage
- the originally tracked v1 cold-start prelabel gaps in this document are complete
