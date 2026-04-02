# Docs Index

Current repository documentation lives under `docs/`.
Historical notes from the former `docu/` directory now live under `docs/archive/`.
If you are updating behavior or workflows, start with the current docs and active plan trackers before consulting archived notes.

## Current Docs

- `../AGENTS.md`: repo-level guidance for agents on which docs are canonical, when to update them, and the expectation that behavior changes ship with tests
- `architecture.md`: current runtime and domain architecture, including low-latency sequence import/capture preview overlays, deployment selection, and trainer inference device behavior
- `demo/README.md`: deterministic README and demo asset workflow
- `CHANGELOG.md`: notable repository changes, including modal preview inference, preview latency tuning, deployed-model selection, and CUDA runtime updates
- `plans/`: dated design notes, implementation plans, and active cleanup trackers
- `plans/2026-04-02-agent-autopilot-spec.md`: proposed constrained agent-autopilot roadmap covering dataset health, baseline recommendations, sweep planning, variant guidance, relabel loops, and deployment advice
- `plans/2026-03-15-cleanup-coverage-refactor-tracker.md`: active cleanup, coverage, and refactor tracker
- `archive/`: historical references moved from `docu/`

## Test and Coverage Notes

- API test and coverage commands prefer the Docker `test` profile because the API suite expects Postgres and Redis.
- Default API test and coverage commands exclude `apps/api/tests/ml`; run `make test-api-ml` for the ML-only suite in the ML-enabled API test container.
- Web test scripts prefer an nvm-managed Linux or WSL Node install; the Windows npm shim can fail in mixed-shell setups, and focused workspace runs may need `node --test --test-isolation=none`.
- Preview-latency helper coverage lives in focused web tests such as `apps/web/tests/mediaPreview.test.js` and `apps/web/tests/previewScheduler.test.js`.
- Trainer and worker test commands expect either a working app-local virtual environment or a Python interpreter on `PATH`.
- Non-Darwin trainer builds now install `onnxruntime-gpu`; deployed inference only selects CUDA when the container exposes `CUDAExecutionProvider`, otherwise the runtime falls back to CPU.
- The checked-in `apps/api/.venv` may not be portable across machines; use Docker or recreate the virtual environment locally if needed.
