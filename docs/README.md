# Docs Index

Current repository documentation lives under `docs/`.
Historical notes from the former `docu/` directory now live under `docs/archive/`.
If you are updating behavior or workflows, start with the current docs and active plan trackers before consulting archived notes.

## Current Docs

- `architecture.md`: current runtime and domain architecture
- `demo/README.md`: deterministic README and demo asset workflow
- `CHANGELOG.md`: notable repository changes
- `plans/`: dated design notes, implementation plans, and active cleanup trackers
- `plans/2026-03-15-cleanup-coverage-refactor-tracker.md`: active cleanup, coverage, and refactor tracker
- `archive/`: historical references moved from `docu/`

## Test and Coverage Notes

- API test and coverage commands prefer the Docker `test` profile because the API suite expects Postgres and Redis.
- Default API test and coverage commands exclude `apps/api/tests/ml`; run `make test-api-ml` for the ML-only suite in the ML-enabled API test container.
- Web test scripts prefer an nvm-managed Linux or WSL Node install; the Windows npm shim can fail in mixed-shell setups.
- Trainer and worker test commands expect either a working app-local virtual environment or a Python interpreter on `PATH`.
- The checked-in `apps/api/.venv` may not be portable across machines; use Docker or recreate the virtual environment locally if needed.
