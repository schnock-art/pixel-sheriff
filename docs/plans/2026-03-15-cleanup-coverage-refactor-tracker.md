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
- Current phase: `Complete`
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
- [x] Split `ProjectAssetsWorkspace` orchestration responsibilities
- [x] Extract non-UI logic from experiment detail page
- [x] Extract non-UI logic from model detail page
- [x] Break up API prelabels service into focused modules
- [x] Break up startup migrations into versioned modules
- [x] Split monolithic API and trainer test files by domain

## Failing Test Follow-up

- [x] Fix the API category response or fixture regression causing repeated `category["id"]` failures in `apps/api/tests/test_api.py`
- [x] Resolve the `/api/v1/projects/{project_id}/exports` contract mismatch between legacy `410` behavior and the current test expectation
- [x] Decide whether the default API full-suite command should install ML extras or exclude ML-only tests, then align the `api-test` flow with that decision
- [x] Rerun the full API suite in the `api-test` container after the API fixes land
- [x] Fix trainer session cache TTL eviction behavior so `test_cache_ttl_eviction_reloads_session` passes
- [x] Fix trainer checkpoint artifact generation or test expectations around `latest.pt` in `test_runner_process_writes_events_metrics_and_checkpoints`
- [x] Rerun the full trainer suite in the trainer container after the trainer fixes land

## Validation and Closeout

- [x] Run full test matrix with the new commands
  - Web and worker remained green from the earlier baseline pass; this follow-up reran `api-test` (`149 passed`), `api-test-ml` (`13 passed`), targeted trainer regressions (`7 passed`), and the full trainer suite in the trainer base container (`50 passed`).
  - This model-detail refactor reran the web unit suite with `node --test tests/*.test.js` (`38 passed`) plus `node ./node_modules/typescript/bin/tsc --noEmit` (`passed`). The default `npm test` / `next build` commands still hit sandbox `spawn EPERM` in this environment.
  - This prelabels-service refactor reran `PYTHONPATH=apps/api/src python3 -m pytest apps/api/tests/test_prelabels_api.py apps/api/tests/test_prelabel_matching.py -q` (`13 passed`) plus `PYTHONPATH=apps/api/src python3 -m pytest apps/api/tests/test_sequences_api.py apps/api/tests/test_video_frames_service.py -q` (`9 passed`).
  - This startup-migrations refactor reran `PYTHONPATH=apps/api/src python3 -m pytest apps/api/tests/test_migrations_sequences.py apps/api/tests/test_migrations_startup.py -q` (`3 passed`).
  - This test-file split refactor reran representative API slices across every new module with `PYTHONPATH=apps/api/src python3 -m pytest ... -q` (`18 passed`) and the full split trainer suite with `PYTHONPATH=apps/trainer/src:apps/ml/src python3 -m pytest apps/trainer/tests/test_trainer_models.py apps/trainer/tests/test_trainer_data.py apps/trainer/tests/test_trainer_detection.py apps/trainer/tests/test_trainer_runner.py apps/trainer/tests/test_trainer_checkpoints.py apps/trainer/tests/test_trainer_export_onnx.py -q` (`32 passed`, `2 skipped`).
  - API test harness now patches router-level `FileResponse` imports to an eager test-only response shim so sandbox in-process runs no longer hang on `anyio.to_thread.run_sync(os.stat, ...)`; targeted download-path regressions passed (`9 passed`) and the full split API suite passed with `PYTHONPATH=apps/api/src python3 -m pytest apps/api/tests/test_api_core.py apps/api/tests/test_api_models.py apps/api/tests/test_api_suggestions.py apps/api/tests/test_api_deployments.py apps/api/tests/test_api_experiment_runtime.py apps/api/tests/test_api_dataset_versions.py -q` (`92 passed`).
- [x] Generate and review baseline coverage reports
  - Used local coverage runs with a temporary `/tmp` `pytest-cov` toolchain because Docker daemon access was blocked in the sandbox and the host Python did not have coverage extras installed.
  - API coverage: `PYTHONPATH=/tmp/pixel_sheriff_pytest_cov_compat2 python3 -m pytest --cov=sheriff_api --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml -q --ignore=tests/ml tests` (`153 passed`, `69%` total). This run also exposed and validated the stale `test_experiments_api.py` helper import fix after the test-file split.
  - Trainer coverage: `PYTHONPATH=/tmp/pixel_sheriff_pytest_cov_compat2 python3 -m pytest --cov=pixel_sheriff_trainer --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml -q tests` (`48 passed`, `2 skipped`, `71%` total).
  - Worker coverage: `PYTHONPATH=/tmp/pixel_sheriff_pytest_cov_compat2 python3 -m pytest --cov=sheriff_worker --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml -q tests` (`2 passed`, `53%` total).
  - Web coverage: `bash ./scripts/run_web_coverage.sh` after restoring `apps/web` dependencies with `npm ci` (`38 passed`, `89.77%` statements, `66.87%` branches).
  - Review summary: strongest remaining gaps are API ML adapter/scaffolding modules and heavyweight migration backfills, trainer segmentation/detection pipeline paths plus inference app flows, and the worker runtime loop / job wrappers.
- [x] Refresh changelog and docs notes for the completed cleanup
- [x] Mark tracker complete with final summary
  - Cleanup goals are complete: docs now live under `docs/`, repo-level test and coverage entrypoints are documented, high-risk refactors were split, and stale test harness edges uncovered by the split and coverage runs were fixed.
  - Final baseline health in this environment: API tests pass locally with the test-only eager `FileResponse` shim, trainer and worker suites pass, and web unit tests plus coverage pass after restoring local Node dependencies.
  - Remaining caveats are environmental rather than tracker blockers: Docker-based API coverage was replaced with a local fallback because daemon access is sandboxed, and the default web `npm test` / `next build` flows still hit sandbox `spawn EPERM`.

## Tracking Rules

- Update this file after every meaningful change to a checklist item.
- For any `[~]` or `[!]` item, add one short status line directly below it with the reason or next step.
- Keep findings brief and stable; only the checklist and phase summary should change frequently.
- Treat this tracker as the temporary source of truth for this initiative until the work is complete.

## Assumptions

- The tracker lives under `docs/plans/` as a dated effort file, not in `docu/`.
- This first pass tracks progress and baselines only; it does not introduce coverage thresholds yet.
- The file reflects the current repo state immediately, so assessment items start as done and implementation items start as pending.
