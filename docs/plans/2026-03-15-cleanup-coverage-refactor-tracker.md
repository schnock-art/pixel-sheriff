# Cleanup, Coverage, and Refactor Tracker

This file tracks the docs cleanup, test and coverage baseline work, and follow-up refactors for the current repository review.
It is the temporary source of truth for this effort instead of adding more checklist debt to `docs/archive/IMPLEMENTATION_TASKS.md`.
Use it to record progress as items move from assessment into implementation and validation.

## Progress Legend

- `[x]` done
- `[~]` in progress
- `[ ]` not started
- `[!]` blocked

## Snapshot

- Date started: `2026-03-15`
- Current phase: `Failing test follow-up validated; broader cleanup items remain`
- Source assessment: this repo review

## Findings

- docs were split between `docs/` and `docu/` at assessment start
- no coverage tooling was configured at assessment start
- the original smoke targets did not represent full-suite health
- local test environment has runner friction
- largest refactor candidates are in the web workspace, API prelabels and migrations, and monolithic test files

## Docs Consolidation

- [x] Assess current documentation layout and source-of-truth conflicts
- [x] Create `docs/README.md` as the docs index
- [x] Consolidate current docs under `docs/`
- [x] Move historical `docu/` content into an archived location under `docs/`
- [x] Update README links and wording to match the new docs structure

## Test and Coverage Baseline

- [x] Assess current test layout and runner setup
- [x] Add explicit full-suite commands for `web`, `api`, `trainer`, and `worker`
- [x] Add repo-level `make test-all`
- [x] Add baseline coverage reporting for Python apps
- [x] Add baseline coverage reporting for the web app
- [x] Document environment caveats for Docker, Node, and local Python

## Refactor Candidates

- [x] Identify high-risk large modules and oversized tests
- [ ] Split `ProjectAssetsWorkspace` orchestration responsibilities
- [ ] Extract non-UI logic from experiment detail page
- [ ] Extract non-UI logic from model detail page
- [ ] Break up API prelabels service into focused modules
- [ ] Break up startup migrations into versioned modules
- [ ] Split monolithic API and trainer test files by domain

## Failing Test Follow-up

- [x] Fix the API category response or fixture regression causing repeated `category["id"]` failures in `apps/api/tests/test_api.py`
- [x] Resolve the `/api/v1/projects/{project_id}/exports` contract mismatch between legacy `410` behavior and the current test expectation
- [x] Decide whether the default API full-suite command should install ML extras or exclude ML-only tests, then align the `api-test` flow with that decision
- [x] Rerun the full API suite in the `api-test` container after the API fixes land
- [x] Fix trainer session cache TTL eviction behavior so `test_cache_ttl_eviction_reloads_session` passes
- [x] Fix trainer checkpoint artifact generation or test expectations around `latest.pt` in `test_runner_process_writes_events_metrics_and_checkpoints`
- [x] Rerun the full trainer suite in the trainer container after the trainer fixes land

## Validation and Closeout

- [~] Run full test matrix with the new commands
  - Web and worker remained green from the earlier baseline pass; this follow-up reran `api-test` (`149 passed`), `api-test-ml` (`13 passed`), targeted trainer regressions (`7 passed`), and the full trainer suite in the trainer base container (`50 passed`).
- [ ] Generate and review baseline coverage reports
- [x] Refresh changelog and docs notes for the completed cleanup
- [ ] Mark tracker complete with final summary

## Tracking Rules

- Update this file after every meaningful change to a checklist item.
- For any `[~]` or `[!]` item, add one short status line directly below it with the reason or next step.
- Keep findings brief and stable; only the checklist and phase summary should change frequently.
- Treat this tracker as the temporary source of truth for this initiative until the work is complete.

## Assumptions

- The tracker lives under `docs/plans/` as a dated effort file, not in `docu/`.
- This first pass tracks progress and baselines only; it does not introduce coverage thresholds yet.
- The file reflects the current repo state immediately, so assessment items start as done and implementation items start as pending.
