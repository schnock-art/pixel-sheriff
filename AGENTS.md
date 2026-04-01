# Agent Guide

This repository expects agents to work from the current documentation, keep tests aligned with behavior changes, and leave the docs more accurate than they found them.

## Documentation Structure

Start with these sources, in this order:

1. `README.md`
   - Product-level overview, quickstart, runtime topology, and the user-facing workflow.
   - If a change affects the core product story, setup flow, or major feature list, update this file.
2. `docs/README.md`
   - Index for the active documentation set.
   - Use this as the map for deciding where additional documentation belongs.
3. `docs/architecture.md`
   - Canonical architecture and runtime/domain behavior reference for the implemented system.
   - Update this when behavior, topology, queues, storage, or cross-service flow changes.
4. `docs/CHANGELOG.md`
   - High-signal record of notable repository changes.
   - Add an entry when the change is user-visible, operationally meaningful, or useful future context.
5. `docs/plans/`
   - Dated design notes, active trackers, and implementation plans.
   - Update the relevant plan when a task advances, changes scope, or closes tracked work.
6. `docs/demo/README.md`
   - Documentation for the deterministic demo media pipeline and generated assets in `docs/demo/`.
   - Update when README/demo capture workflows, commands, or outputs change.
7. `packages/contracts/README.md`
   - Canonical description of shared schema and metadata artifacts.
   - Update when contracts, metadata generation, or sync/check workflows change.

Historical references live under `docs/archive/`. They are useful for context, but they are not the source of truth for current behavior unless a current doc explicitly points back to them.

## Documentation Rules

- Prefer updating an existing current doc before creating a new one.
- Keep current behavior in `README.md`, `docs/README.md`, and `docs/architecture.md` consistent with the code.
- If you add a new current doc, also add it to `docs/README.md`.
- If you change schema or metadata contracts, update the relevant contract docs and keep generated/runtime copies in sync.
- Do not treat `docs/archive/` as authoritative for new work.

## Test Expectations

Every code change should include tests that prove the new or changed behavior.

- Web changes: add or update tests under `apps/web/tests/`.
- API changes: add or update tests under `apps/api/tests/`.
- Worker changes: add or update tests under `apps/worker/tests/`.
- Trainer changes: add or update tests under `apps/trainer/tests/`.
- Contract changes: add or update the relevant contract verification coverage and sync checks.

If a task genuinely does not need an automated test, say so explicitly in the final handoff and explain why.

## Task Completion Checklist

Before considering a task done:

- Implement the code change.
- Add or update automated tests for the affected behavior.
- Run the most relevant targeted tests you can in the current environment.
- Update the current documentation that describes the changed behavior.
- Update any active plan tracker or changelog entry when the task warrants it.

If tests cannot be run in the current environment, note that clearly in the final handoff.
