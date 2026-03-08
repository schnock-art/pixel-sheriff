# contracts

Shared contract artifacts live here. This directory is the canonical source for JSON schemas and generated metadata consumed across apps.

## Layout

- `schemas/`
  - `dataset_version_v2.schema.json`
  - `model-config-1.0.schema.json`
- `metadata/`
  - `backbones.v1.json`
  - `families.v1.json`
- `openapi/`
  - reserved for OpenAPI exports
- `ts-client/`
  - reserved for generated TypeScript client artifacts

## Workflow

- Generated metadata is produced into `packages/contracts/metadata`.
- App-local runtime copies under `apps/api` and `apps/web` are synchronized from this directory.
- Run `make contracts-sync` after changing schema or metadata sources.
- Run `make contracts-check` to verify there is no drift.
