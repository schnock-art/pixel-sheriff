# Dataset–Model Contract Design

**Date:** 2026-03-05
**Status:** Approved

## Problem

Three related issues break the contract between datasets and models:

1. **Trainer crash on bbox tasks** — `normalize_task()` converts `"bbox"` → `"detection"`, so the experiment's `config_json.task` is `"detection"`. But `PIPELINE_REGISTRY` is keyed on `"bbox"`. Result: `PIPELINE_REGISTRY.get("detection")` → `None` → `"unsupported_task:detection"`.

2. **No task label in dataset dropdown** — When creating an experiment the dataset version picker shows only the version name, with no indication of which task type (classification / detection / segmentation) it represents.

3. **No dataset/family choice in model editor** — When creating a model it silently uses the active dataset. The model editor has no way to pick a different dataset or switch architecture family. Backbone options are not filtered by task.

## Design

### 1 — Fix the pipeline task key (trainer)

In `apps/trainer/src/pixel_sheriff_trainer/detection/pipeline.py`:

- Change `task_kind = "bbox"` → `task_kind = "detection"`
- Change `PIPELINE_REGISTRY["bbox"] = DetectionPipeline()` → `PIPELINE_REGISTRY["detection"] = DetectionPipeline()`
- Change the hardcoded `"task": "bbox"` string in `write_evaluation` → `"task": "detection"`

`normalize_task()` in `experiments/shared.py` already converts `"bbox"` → `"detection"` correctly; no changes needed there.

### 2 — Architecture family registry with task constraints

New static file `apps/web/src/lib/metadata/families.v1.json`:

```json
{
  "schema_version": "1",
  "families": [
    {
      "name": "resnet_classifier",
      "task": "classification",
      "allowed_backbones": ["resnet18","resnet34","resnet50","resnet101","mobilenet_v3_large","mobilenet_v3_small"]
    },
    {
      "name": "retinanet",
      "task": "detection",
      "allowed_backbones": ["resnet50","resnet101"]
    },
    {
      "name": "deeplabv3",
      "task": "segmentation",
      "allowed_backbones": ["resnet50","resnet101"]
    }
  ]
}
```

Python side: add `FAMILY_TASK_MAP: dict[str, str]` to `apps/api/src/sheriff_api/ml/registry.py` mirroring this data. A new script `apps/api/src/sheriff_api/ml/metadata/generate_families_json.py` (mirrors `generate_registry_json.py`) generates the JSON from the Python source of truth.

### 3 — Task label in experiment dataset dropdown

In `apps/web/src/app/projects/[projectId]/experiments/new/page.tsx`:

- When building `datasetVersionOptions`, also extract `version.task` from each item
- Show label as `"${name} (${task})"` in the `<select>` options

No API changes needed — the task field is already present in the dataset version payload.

### 4 — Editable dataset + family in model editor

#### 4a — API: accept dataset_version_id on model create

In `apps/api/src/sheriff_api/schemas/models.py`, add `dataset_version_id: str | None = None` to `ProjectModelCreate`.

In `apps/api/src/sheriff_api/routers/models.py`, `create_project_model`: if `payload.dataset_version_id` is provided, load that specific version instead of calling `_active_or_latest_dataset_version`.

In `apps/web/src/lib/api.ts`, add `dataset_version_id?: string` to `ProjectModelCreatePayload`.

#### 4b — `/models/new` page — dataset picker

Replace the current auto-create-and-redirect with a picker form:
- Load all dataset versions for the project (with task labels)
- Show a task selector (derived from available dataset versions' tasks) that filters the dataset dropdown
- Show dataset version dropdown (filtered by selected task), labelled `"name (task)"`
- On submit: call `createProjectModel` with `{ dataset_version_id }`, redirect to model editor

#### 4c — Model editor — editable Step 1

The model editor (`apps/web/src/app/projects/[projectId]/models/[modelId]/page.tsx`) gains a new **Step 1: Source** section. It loads all dataset versions on mount.

Three cascading selectors:

```
Task:    [classification ▼]    drives dataset and family filters
Dataset: [v3 — 12 classes ▼]  filtered by task; changes update source_dataset + head.num_classes
Family:  [resnet_classifier ▼] filtered by task (from families.v1.json); changes regenerate arch/head/loss/outputs
```

**Cascade rules:**

| Action | Effect on config |
|---|---|
| Change task | Reset dataset + family to first valid options; regenerate full config |
| Change dataset (same task) | Update `source_dataset.{manifest_id, num_classes, class_order, class_names}` + `architecture.head.num_classes` |
| Change family (same task) | Regenerate `architecture`, `head`, `loss`, `outputs`; keep backbone if in new family's `allowed_backbones`, else reset to first valid |
| Change backbone | Update `architecture.backbone.name` only |

The `modelConfigEditor.js` helper gains new functions:
- `setSourceDataset(config, datasetVersionSummary)` — patches source_dataset + head.num_classes
- `setArchitectureFamily(config, familyName, familiesMetadata)` — regenerates architecture/head/loss/outputs
- `setBackbone(config, backboneName)` — patches backbone.name

All cascade logic runs client-side; no extra API calls until Save.

## Files Changed

| File | Change |
|---|---|
| `apps/trainer/.../detection/pipeline.py` | Registry key + task_kind: `"bbox"` → `"detection"` |
| `apps/api/.../ml/registry.py` | Add `FAMILY_TASK_MAP` |
| `apps/api/.../ml/metadata/generate_families_json.py` | New script |
| `apps/web/src/lib/metadata/families.v1.json` | New static file |
| `apps/api/.../schemas/models.py` | Add `dataset_version_id` to `ProjectModelCreate` |
| `apps/api/.../routers/models.py` | Use `dataset_version_id` if provided |
| `apps/web/src/lib/api.ts` | Add `dataset_version_id` to `ProjectModelCreatePayload` |
| `apps/web/.../experiments/new/page.tsx` | Show task in dataset dropdown |
| `apps/web/.../models/new/page.tsx` | Replace auto-create with picker form |
| `apps/web/.../models/[modelId]/page.tsx` | Add editable Step 1 (task/dataset/family) |
| `apps/web/.../modelConfigEditor.js` | New helpers: setSourceDataset, setArchitectureFamily, setBackbone |
