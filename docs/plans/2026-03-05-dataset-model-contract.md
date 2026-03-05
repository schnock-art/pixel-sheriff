# Dataset–Model Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the bbox training crash, add task labels to dataset dropdowns, and make the model editor fully editable (task/dataset/family/backbone cascade).

**Architecture:** The fix normalises the trainer's pipeline registry key from `"bbox"` to `"detection"` to match the API's `normalize_task()` output. A new `families.v1.json` static file encodes task→family→backbone constraints and is read client-side. The model editor gains three cascading selectors (task → dataset → family) whose changes patch the draft config in-place using new pure helpers in `modelConfigEditor.js`.

**Tech Stack:** Python (FastAPI, Pydantic), PyTorch trainer, Next.js 14 App Router, Node.js test runner (`node:test`), pytest.

---

## Context

- Python venv lives at `apps/api/.venv/Scripts/python` (Windows paths)
- API tests need infra running: `make infra` then use the command in Task 3
- Web tests: `cd apps/web && npm test` (uses `node:test`)
- All task strings flowing from API → trainer must be one of: `"classification"`, `"detection"`, `"segmentation"` — `"bbox"` is a DB-level kind, not a pipeline key

---

### Task 1: Fix trainer — detection pipeline registry key

**Problem:** `PIPELINE_REGISTRY["bbox"]` never matches `job.task = "detection"`, causing the crash.

**Files:**
- Modify: `apps/trainer/src/pixel_sheriff_trainer/detection/pipeline.py`

**Step 1: Read the file**

Open `apps/trainer/src/pixel_sheriff_trainer/detection/pipeline.py` and confirm the three `"bbox"` occurrences to change.

**Step 2: Apply changes**

In `detection/pipeline.py`:

1. Line with `task_kind = "bbox"` → change to `task_kind = "detection"`
2. Inside `write_evaluation`, `"task": "bbox"` → `"task": "detection"` (appears twice — in `payload` dict and in `latest_eval_path` write)
3. Last line `PIPELINE_REGISTRY["bbox"] = DetectionPipeline()` → `PIPELINE_REGISTRY["detection"] = DetectionPipeline()`

**Step 3: Verify no other "bbox" task strings remain in trainer**

```bash
grep -rn '"bbox"' apps/trainer/src/pixel_sheriff_trainer/
```

Expected: zero matches (or only in comments / non-task contexts like annotation payload fields). The string `"bbox"` in evaluation payloads written as annotation kinds is fine if it's not the `task` key.

**Step 4: Commit**

```bash
git add apps/trainer/src/pixel_sheriff_trainer/detection/pipeline.py
git commit -m "fix: change detection pipeline registry key from bbox to detection"
```

---

### Task 2: Add FAMILY_TASK_MAP to API registry

**Purpose:** Single source of truth in Python for the task→family→backbone constraints; used by the script that generates `families.v1.json`.

**Files:**
- Modify: `apps/api/src/sheriff_api/ml/registry.py`

**Step 1: Read the file**

Open `apps/api/src/sheriff_api/ml/registry.py` and confirm the current `FAMILY_REGISTRY` dict.

**Step 2: Add FAMILY_TASK_MAP and FAMILY_BACKBONES**

After the existing `FAMILY_REGISTRY` dict, add:

```python
# Maps each family name to the task it implements.
FAMILY_TASK_MAP: dict[str, str] = {
    "resnet_classifier": "classification",
    "retinanet": "detection",
    "deeplabv3": "segmentation",
}

# Maps each family name to the backbone names it supports.
FAMILY_BACKBONES: dict[str, list[str]] = {
    "resnet_classifier": [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "mobilenet_v3_large", "mobilenet_v3_small",
    ],
    "retinanet": ["resnet50", "resnet101"],
    "deeplabv3": ["resnet50", "resnet101"],
}
```

**Step 3: Commit**

```bash
git add apps/api/src/sheriff_api/ml/registry.py
git commit -m "feat: add FAMILY_TASK_MAP and FAMILY_BACKBONES to ML registry"
```

---

### Task 3: Generate families.v1.json

**Purpose:** Static JSON consumed by the Next.js model editor to filter family and backbone dropdowns without a runtime API call.

**Files:**
- Create: `apps/api/src/sheriff_api/ml/metadata/generate_families_json.py`
- Create: `apps/web/src/lib/metadata/families.v1.json`

**Step 1: Create the generator script**

```python
# apps/api/src/sheriff_api/ml/metadata/generate_families_json.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sheriff_api.ml.registry import FAMILY_BACKBONES, FAMILY_TASK_MAP


def _default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[6]
        / "apps" / "web" / "src" / "lib" / "metadata" / "families.v1.json"
    )


def build_families_payload() -> dict[str, Any]:
    families: list[dict[str, Any]] = []
    for name in sorted(FAMILY_TASK_MAP.keys()):
        families.append(
            {
                "name": name,
                "task": FAMILY_TASK_MAP[name],
                "allowed_backbones": FAMILY_BACKBONES.get(name, []),
            }
        )
    return {"schema_version": "1", "families": families}


def write_families_json(path: str | Path, payload: dict[str, Any] | None = None) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    body = payload if payload is not None else build_families_payload()
    target_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pixel Sheriff families registry JSON")
    parser.add_argument("--out", default=str(_default_output_path()), help="Output path")
    args = parser.parse_args()
    written = write_families_json(args.out)
    print(str(written))


if __name__ == "__main__":
    main()
```

**Step 2: Run the generator**

```bash
cd apps/api && .venv/Scripts/python -m sheriff_api.ml.metadata.generate_families_json
```

Expected: prints the path to `apps/web/src/lib/metadata/families.v1.json`.

**Step 3: Verify the output**

```bash
cat apps/web/src/lib/metadata/families.v1.json
```

Expected content:
```json
{
  "families": [
    {
      "allowed_backbones": ["resnet50", "resnet101"],
      "name": "deeplabv3",
      "task": "segmentation"
    },
    {
      "allowed_backbones": ["resnet50", "resnet101"],
      "name": "retinanet",
      "task": "detection"
    },
    {
      "allowed_backbones": ["resnet18","resnet34","resnet50","resnet101","mobilenet_v3_large","mobilenet_v3_small"],
      "name": "resnet_classifier",
      "task": "classification"
    }
  ],
  "schema_version": "1"
}
```

**Step 4: Commit**

```bash
git add apps/api/src/sheriff_api/ml/metadata/generate_families_json.py
git add apps/web/src/lib/metadata/families.v1.json
git commit -m "feat: add families.v1.json with task/backbone constraints"
```

---

### Task 4: API — accept dataset_version_id on model create

**Purpose:** Allow callers to specify exactly which dataset version a new model should be built from.

**Files:**
- Modify: `apps/api/src/sheriff_api/schemas/models.py`
- Modify: `apps/api/src/sheriff_api/routers/models.py`

**Step 1: Add field to schema**

In `schemas/models.py`, change `ProjectModelCreate`:

```python
class ProjectModelCreate(BaseModel):
    name: str | None = None
    dataset_version_id: str | None = None
```

**Step 2: Update the router to use it**

In `routers/models.py`, find `create_project_model`. Currently it always calls `_active_or_latest_dataset_version`. Replace the block that loads the dataset version:

```python
# Before (single call):
active_dataset_version = _active_or_latest_dataset_version(project_id)

# After (prefer explicit version if provided):
if payload.dataset_version_id:
    loaded = dataset_store.get_version(project_id, payload.dataset_version_id)
    if loaded is None:
        raise api_error(
            status_code=404,
            code="dataset_version_not_found",
            message="Dataset version not found in project",
            details={"project_id": project_id, "dataset_version_id": payload.dataset_version_id},
        )
    active_dataset_version = loaded["version"]
else:
    active_dataset_version = _active_or_latest_dataset_version(project_id)
```

The remainder of the function (task lookup, manifest build, config build) is unchanged.

**Step 3: Write the failing test**

In `apps/api/tests/test_api.py`, add a test after existing model tests:

```python
async def test_create_model_with_explicit_dataset_version_id(client, project):
    """Creating a model with dataset_version_id uses that version, not the active one."""
    task_id = project["default_task_id"]
    # Create a category so the dataset has at least one class
    await client.post(f"/api/v1/projects/{project['id']}/categories",
                      json={"task_id": task_id, "name": "cat"})
    # Create a dataset version explicitly
    version_resp = await client.post(
        f"/api/v1/projects/{project['id']}/datasets/versions",
        json={
            "name": "explicit_v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {}},
        },
    )
    assert version_resp.status_code == 200
    version_id = version_resp.json()["version"]["dataset_version_id"]

    # Create model using that explicit version
    resp = await client.post(
        f"/api/v1/projects/{project['id']}/models",
        json={"dataset_version_id": version_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["config"]["source_dataset"]["manifest_id"] == version_id
```

**Step 4: Run the test to verify it fails**

```bash
make infra
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test \
STORAGE_ROOT=/tmp/pixel_sheriff_test_data \
apps/api/.venv/Scripts/python -m pytest -s apps/api/tests/test_api.py -k "test_create_model_with_explicit_dataset_version_id" -v
```

Expected: FAIL (field not accepted yet — or PASS if the test was written after schema change; either way run it).

**Step 5: Run again after schema + router changes to verify it passes**

Same command. Expected: PASS.

**Step 6: Commit**

```bash
git add apps/api/src/sheriff_api/schemas/models.py
git add apps/api/src/sheriff_api/routers/models.py
git add apps/api/tests/test_api.py
git commit -m "feat: accept dataset_version_id when creating a project model"
```

---

### Task 5: Frontend API type — dataset_version_id

**Files:**
- Modify: `apps/web/src/lib/api.ts`

**Step 1: Add field to payload type**

Find `ProjectModelCreatePayload` (around line 279) and add the optional field:

```typescript
export interface ProjectModelCreatePayload {
  name?: string;
  dataset_version_id?: string;
}
```

**Step 2: Commit**

```bash
git add apps/web/src/lib/api.ts
git commit -m "feat: add dataset_version_id to ProjectModelCreatePayload"
```

---

### Task 6: Show task in experiment dataset dropdown

**Files:**
- Modify: `apps/web/src/app/projects/[projectId]/experiments/new/page.tsx`

**Step 1: Read the file**

Open `apps/web/src/app/projects/[projectId]/experiments/new/page.tsx` and find the block that builds `datasetVersionOptions` (~line 65).

**Step 2: Extract task from version and include in label**

Change the `versionRows` map:

```typescript
const versionRows = (datasetVersions.items ?? []).map((item) => {
  const version = item.version as Record<string, unknown>;
  const id = typeof version.dataset_version_id === "string" ? version.dataset_version_id : "";
  const name = typeof version.name === "string" ? version.name : id;
  const task = typeof version.task === "string" ? version.task : "";
  const label = task ? `${name} (${task})` : name;
  return { id, label };
});
setDatasetVersionOptions(versionRows.filter((item) => item.id));
```

**Step 3: Update the JSX to use `label` instead of `name`**

Find the `<select>` that renders `datasetVersionOptions` (~line 172):

```tsx
{datasetVersionOptions.map((row) => (
  <option key={row.id} value={row.id}>
    {row.label}
  </option>
))}
```

Also update the `useState` type — change:
```typescript
const [datasetVersionOptions, setDatasetVersionOptions] = useState<Array<{ id: string; name: string }>>([]);
```
to:
```typescript
const [datasetVersionOptions, setDatasetVersionOptions] = useState<Array<{ id: string; label: string }>>([]);
```

**Step 4: Verify dev server renders correctly**

With `make dev-web` running, navigate to `/projects/{id}/experiments/new`. Dataset dropdown should now show e.g. `"my_dataset_v1 (classification)"`.

**Step 5: Commit**

```bash
git add "apps/web/src/app/projects/[projectId]/experiments/new/page.tsx"
git commit -m "feat: show task name in experiment dataset version dropdown"
```

---

### Task 7: modelConfigEditor.js — new cascade helpers

**Purpose:** Pure functions that patch the model config draft when the user changes dataset, family, or backbone. Tested with `node:test` like the existing helpers.

**Files:**
- Modify: `apps/web/src/lib/workspace/modelConfigEditor.js`
- Modify: `apps/web/tests/modelConfigEditor.test.js` (create if it doesn't exist)

**Step 1: Write failing tests**

Create (or add to) `apps/web/tests/modelConfigEditor.test.js`:

```js
const test = require("node:test");
const assert = require("node:assert/strict");

const {
  setSourceDataset,
  setArchitectureFamily,
  setBackbone,
} = require("../src/lib/workspace/modelConfigEditor.js");

// --- setSourceDataset ---

test("setSourceDataset patches source_dataset fields and head.num_classes", () => {
  const config = {
    source_dataset: { manifest_id: "old", task: "classification", num_classes: 2, class_order: ["a"], class_names: ["A"] },
    architecture: { head: { type: "linear", num_classes: 2 } },
  };
  const datasetSummary = {
    dataset_version_id: "new-id",
    task: "classification",
    num_classes: 5,
    class_order: ["x", "y", "z", "w", "v"],
    class_names: ["X", "Y", "Z", "W", "V"],
  };
  const result = setSourceDataset(config, datasetSummary);

  assert.equal(result.source_dataset.manifest_id, "new-id");
  assert.equal(result.source_dataset.num_classes, 5);
  assert.deepEqual(result.source_dataset.class_order, ["x", "y", "z", "w", "v"]);
  assert.deepEqual(result.source_dataset.class_names, ["X", "Y", "Z", "W", "V"]);
  assert.equal(result.architecture.head.num_classes, 5);
});

test("setSourceDataset does not mutate input", () => {
  const config = {
    source_dataset: { manifest_id: "old", num_classes: 2, class_order: [], class_names: [] },
    architecture: { head: { num_classes: 2 } },
  };
  const summary = { dataset_version_id: "new", num_classes: 3, class_order: [], class_names: [] };
  setSourceDataset(config, summary);
  assert.equal(config.source_dataset.manifest_id, "old");
});

// --- setArchitectureFamily ---

const FAMILIES = [
  { name: "resnet_classifier", task: "classification", allowed_backbones: ["resnet18", "resnet50"] },
  { name: "retinanet", task: "detection", allowed_backbones: ["resnet50", "resnet101"] },
  { name: "deeplabv3", task: "segmentation", allowed_backbones: ["resnet50", "resnet101"] },
];

test("setArchitectureFamily switches to retinanet defaults", () => {
  const config = {
    source_dataset: { num_classes: 3 },
    architecture: { family: "resnet_classifier", backbone: { name: "resnet18", pretrained: true }, head: { type: "linear", num_classes: 3 } },
    loss: { type: "classification_cross_entropy" },
    outputs: { primary: { name: "classification_logits", type: "task_output", task: "classification", format: "classification_logits" }, aux: [] },
    export: { onnx: { output_names: ["classification_logits"] } },
  };
  const result = setArchitectureFamily(config, "retinanet", FAMILIES);

  assert.equal(result.architecture.family, "retinanet");
  assert.equal(result.architecture.head.type, "retinanet");
  assert.equal(result.architecture.neck.type, "fpn");
  assert.equal(result.loss.type, "retinanet_default");
  assert.equal(result.outputs.primary.format, "coco_detections");
  // backbone kept if still valid
  assert.equal(result.architecture.backbone.name, "resnet50"); // resnet18 not in retinanet allowed; reset to first
});

test("setArchitectureFamily keeps backbone when compatible", () => {
  const config = {
    source_dataset: { num_classes: 2 },
    architecture: { family: "retinanet", backbone: { name: "resnet101", pretrained: true }, neck: { type: "fpn", fpn_channels: 256 }, head: { type: "retinanet", num_classes: 2 } },
    loss: { type: "retinanet_default" },
    outputs: { primary: { name: "coco_detections", type: "task_output", task: "detection", format: "coco_detections" }, aux: [] },
    export: { onnx: { output_names: ["coco_detections"] } },
  };
  const result = setArchitectureFamily(config, "retinanet", FAMILIES);
  assert.equal(result.architecture.backbone.name, "resnet101"); // still valid
});

// --- setBackbone ---

test("setBackbone patches only backbone.name", () => {
  const config = {
    architecture: {
      family: "resnet_classifier",
      backbone: { name: "resnet18", pretrained: true },
      head: { num_classes: 5 },
    },
  };
  const result = setBackbone(config, "resnet50");
  assert.equal(result.architecture.backbone.name, "resnet50");
  assert.equal(result.architecture.backbone.pretrained, true); // unchanged
  assert.equal(result.architecture.head.num_classes, 5); // unchanged
});

test("setBackbone does not mutate input", () => {
  const config = { architecture: { backbone: { name: "resnet18" } } };
  setBackbone(config, "resnet50");
  assert.equal(config.architecture.backbone.name, "resnet18");
});
```

**Step 2: Run tests to verify they fail**

```bash
cd apps/web && node --test tests/modelConfigEditor.test.js
```

Expected: module not found or function-not-exported errors.

**Step 3: Implement the helpers in modelConfigEditor.js**

Add to the bottom of `apps/web/src/lib/workspace/modelConfigEditor.js` (before `module.exports`):

```js
// Default architecture/loss/outputs/export per family name.
const _FAMILY_DEFAULTS = {
  resnet_classifier: (numClasses, labelMode) => ({
    architecture: {
      family: "resnet_classifier",
      framework: "torchvision",
      precision: "fp32",
      backbone: { name: "resnet18", pretrained: true },
      neck: { type: "none" },
      head: { type: "linear", num_classes: numClasses },
    },
    loss: { type: labelMode === "multi_label" ? "classification_bce_with_logits" : "classification_cross_entropy" },
    primaryOutput: { name: "classification_logits", type: "task_output", task: "classification", format: "classification_logits" },
    outputName: "classification_logits",
  }),
  retinanet: (numClasses) => ({
    architecture: {
      family: "retinanet",
      framework: "torchvision",
      precision: "fp32",
      backbone: { name: "resnet50", pretrained: true },
      neck: { type: "fpn", fpn_channels: 256 },
      head: { type: "retinanet", num_classes: numClasses },
    },
    loss: { type: "retinanet_default" },
    primaryOutput: { name: "coco_detections", type: "task_output", task: "detection", format: "coco_detections" },
    outputName: "coco_detections",
  }),
  deeplabv3: (numClasses) => ({
    architecture: {
      family: "deeplabv3",
      framework: "torchvision",
      precision: "fp32",
      backbone: { name: "resnet50", pretrained: true },
      neck: { type: "none" },
      head: { type: "deeplabv3_head", num_classes: numClasses },
    },
    loss: { type: "deeplabv3_default" },
    primaryOutput: { name: "coco_segmentation", type: "task_output", task: "segmentation", format: "coco_segmentation" },
    outputName: "coco_segmentation",
  }),
};

/**
 * Update source_dataset and head.num_classes when the user picks a different dataset version.
 * datasetSummary shape: { dataset_version_id, task, num_classes, class_order, class_names }
 */
function setSourceDataset(config, datasetSummary) {
  const next = cloneModelConfig(config);
  const source = isPlainObject(next.source_dataset) ? next.source_dataset : {};
  source.manifest_id = datasetSummary.dataset_version_id;
  if (datasetSummary.task) source.task = datasetSummary.task;
  source.num_classes = datasetSummary.num_classes;
  source.class_order = datasetSummary.class_order ?? source.class_order;
  source.class_names = datasetSummary.class_names ?? source.class_names;
  next.source_dataset = source;

  // Keep head.num_classes in sync
  const arch = isPlainObject(next.architecture) ? next.architecture : {};
  const head = isPlainObject(arch.head) ? arch.head : {};
  head.num_classes = datasetSummary.num_classes;
  arch.head = head;
  next.architecture = arch;
  return next;
}

/**
 * Switch to a different architecture family (within the same task).
 * Regenerates architecture, loss, outputs; keeps backbone if still in allowed_backbones.
 * familiesMetadata: array of { name, task, allowed_backbones } from families.v1.json
 */
function setArchitectureFamily(config, familyName, familiesMetadata) {
  const next = cloneModelConfig(config);
  const source = isPlainObject(next.source_dataset) ? next.source_dataset : {};
  const numClasses = typeof source.num_classes === "number" ? source.num_classes : 1;
  const labelMode = typeof source.label_mode === "string" ? source.label_mode : null;

  const defaults = _FAMILY_DEFAULTS[familyName];
  if (!defaults) return next; // unknown family — no-op
  const { architecture, loss, primaryOutput, outputName } = defaults(numClasses, labelMode);

  // Keep current backbone.pretrained and frozen_stages; only reset name if incompatible.
  const familyMeta = Array.isArray(familiesMetadata)
    ? familiesMetadata.find((f) => f.name === familyName)
    : null;
  const allowedBackbones = familyMeta?.allowed_backbones ?? [architecture.backbone.name];
  const currentBackbone = isPlainObject(next.architecture) && isPlainObject(next.architecture.backbone)
    ? next.architecture.backbone.name
    : null;
  if (currentBackbone && allowedBackbones.includes(currentBackbone)) {
    architecture.backbone.name = currentBackbone;
  }
  // Preserve pretrained flag if present
  if (isPlainObject(next.architecture) && isPlainObject(next.architecture.backbone)) {
    architecture.backbone.pretrained = Boolean(next.architecture.backbone.pretrained);
  }

  next.architecture = architecture;
  next.loss = loss;

  const outputs = isPlainObject(next.outputs) ? next.outputs : {};
  outputs.primary = primaryOutput;
  // Keep aux outputs (embeddings etc.) as-is
  next.outputs = outputs;

  // Update ONNX output_names to match new primary output
  const exportSpec = isPlainObject(next.export) ? next.export : {};
  const onnx = isPlainObject(exportSpec.onnx) ? exportSpec.onnx : {};
  onnx.output_names = [outputName];
  exportSpec.onnx = onnx;
  next.export = exportSpec;

  return next;
}

/**
 * Change only the backbone name; all other settings preserved.
 */
function setBackbone(config, backboneName) {
  const next = cloneModelConfig(config);
  const arch = isPlainObject(next.architecture) ? next.architecture : {};
  const backbone = isPlainObject(arch.backbone) ? arch.backbone : {};
  backbone.name = backboneName;
  arch.backbone = backbone;
  next.architecture = arch;
  return next;
}
```

Also update `module.exports` to include the three new functions:

```js
module.exports = {
  cloneModelConfig,
  isModelConfigDirty,
  createEmbeddingAuxOutput,
  setEmbeddingAuxEnabled,
  setEmbeddingProjection,
  setSquareInputSize,
  setDynamicShapeFlags,
  setSourceDataset,
  setArchitectureFamily,
  setBackbone,
};
```

**Step 4: Run tests to verify they pass**

```bash
cd apps/web && node --test tests/modelConfigEditor.test.js
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add apps/web/src/lib/workspace/modelConfigEditor.js
git add apps/web/tests/modelConfigEditor.test.js
git commit -m "feat: add setSourceDataset, setArchitectureFamily, setBackbone helpers"
```

---

### Task 8: /models/new — dataset picker form

**Purpose:** Replace the current auto-create-and-redirect with an explicit task + dataset picker.

**Files:**
- Modify: `apps/web/src/app/projects/[projectId]/models/new/page.tsx`

**Step 1: Read the current file**

Open the file to understand current structure (currently just auto-creates and redirects on mount).

**Step 2: Rewrite the page**

Replace the entire file content with a picker similar to `experiments/new/page.tsx`:

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, createProjectModel, listDatasetVersions } from "../../../../../lib/api";

interface NewModelPageProps {
  params: { projectId: string };
}

function parseApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.responseBody) {
    try {
      const parsed = JSON.parse(error.responseBody) as { error?: { message?: string } };
      if (parsed.error?.message) return parsed.error.message;
      return error.responseBody;
    } catch {
      return error.responseBody;
    }
  }
  if (error instanceof Error) return error.message;
  return fallback;
}

export default function NewModelPage({ params }: NewModelPageProps) {
  const router = useRouter();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);

  const [allVersions, setAllVersions] = useState<Array<{ id: string; name: string; task: string }>>([]);
  const [selectedTask, setSelectedTask] = useState("");
  const [selectedVersionId, setSelectedVersionId] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function load() {
      setIsLoading(true);
      try {
        const data = await listDatasetVersions(projectId);
        if (!isMounted) return;
        const rows = (data.items ?? []).flatMap((item) => {
          const version = item.version as Record<string, unknown>;
          const id = typeof version.dataset_version_id === "string" ? version.dataset_version_id : "";
          const name = typeof version.name === "string" ? version.name : id;
          const task = typeof version.task === "string" ? version.task : "";
          return id ? [{ id, name, task }] : [];
        });
        setAllVersions(rows);

        // Default to first available task and version
        const firstTask = rows[0]?.task ?? "";
        setSelectedTask(firstTask);
        const firstId = rows.find((r) => r.task === firstTask)?.id ?? "";
        setSelectedVersionId(firstId);
      } catch (error) {
        if (isMounted) setErrorMessage(parseApiErrorMessage(error, "Failed to load dataset versions"));
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    void load();
    return () => { isMounted = false; };
  }, [projectId]);

  // Unique task labels from available versions
  const availableTasks = useMemo(
    () => [...new Set(allVersions.map((r) => r.task))].filter(Boolean),
    [allVersions],
  );

  // Dataset versions for the selected task
  const filteredVersions = useMemo(
    () => allVersions.filter((r) => r.task === selectedTask),
    [allVersions, selectedTask],
  );

  function handleTaskChange(task: string) {
    setSelectedTask(task);
    const first = allVersions.find((r) => r.task === task);
    setSelectedVersionId(first?.id ?? "");
  }

  async function handleCreate() {
    if (!selectedVersionId) return;
    setErrorMessage(null);
    setIsCreating(true);
    try {
      const created = await createProjectModel(projectId, { dataset_version_id: selectedVersionId });
      router.replace(
        `/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(created.id)}`,
      );
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to create model"));
      setIsCreating(false);
    }
  }

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>New Model</h2>
        </header>
        <div className="placeholder-card experiment-new-card">
          {isLoading ? (
            <p>Loading dataset versions...</p>
          ) : allVersions.length === 0 ? (
            <p>No dataset versions found. Create and activate a dataset version first.</p>
          ) : (
            <>
              <label className="project-field">
                <span>Task</span>
                <select
                  value={selectedTask}
                  onChange={(e) => handleTaskChange(e.target.value)}
                  disabled={isCreating}
                >
                  {availableTasks.map((task) => (
                    <option key={task} value={task}>{task}</option>
                  ))}
                </select>
              </label>
              <label className="project-field">
                <span>Dataset Version</span>
                <select
                  value={selectedVersionId}
                  onChange={(e) => setSelectedVersionId(e.target.value)}
                  disabled={isCreating || filteredVersions.length === 0}
                >
                  {filteredVersions.length === 0 ? (
                    <option value="">No versions for this task</option>
                  ) : null}
                  {filteredVersions.map((row) => (
                    <option key={row.id} value={row.id}>{row.name}</option>
                  ))}
                </select>
              </label>
              {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
              <div className="project-modal-actions">
                <button
                  type="button"
                  className="primary-button"
                  disabled={!selectedVersionId || isCreating}
                  onClick={() => void handleCreate()}
                >
                  {isCreating ? "Creating..." : "Create Model"}
                </button>
              </div>
            </>
          )}
        </div>
      </section>
    </main>
  );
}
```

**Step 3: Verify in dev**

With `make dev-web` running, navigate to `/projects/{id}/models/new`. Should show a task selector and dataset dropdown. Select a bbox dataset → click "Create Model" → should redirect to the model editor.

**Step 4: Commit**

```bash
git add "apps/web/src/app/projects/[projectId]/models/new/page.tsx"
git commit -m "feat: replace auto-create model with dataset picker form"
```

---

### Task 9: Model editor — editable Step 1 (task/dataset/family/backbone)

**Purpose:** The model editor loads all dataset versions and shows cascading selectors. Changing task/dataset/family/backbone updates the draft config client-side.

**Files:**
- Modify: `apps/web/src/app/projects/[projectId]/models/[modelId]/page.tsx`

This is the largest single change. Work through it in three sub-steps.

**Step 1: Read families.v1.json and add imports**

At the top of the file, add the import for the new helpers and the families metadata:

```tsx
import familiesMetadata from "../../../../../lib/metadata/families.v1.json";
import {
  cloneModelConfig,
  isModelConfigDirty,
  setArchitectureFamily,
  setBackbone,
  setDynamicShapeFlags,
  setEmbeddingAuxEnabled,
  setEmbeddingProjection,
  setSourceDataset,
  setSquareInputSize,
} from "../../../../../lib/workspace/modelConfigEditor";
```

Also add the API import:
```tsx
import {
  // existing imports ...
  listDatasetVersions,
  type DatasetVersionListPayload,
} from "../../../../../lib/api";
```

**Step 2: Add state and load dataset versions**

Add state variables alongside the existing ones:

```tsx
const [allDatasetVersions, setAllDatasetVersions] = useState<
  Array<{ id: string; name: string; task: string; numClasses: number; classOrder: string[]; classNames: string[] }>
>([]);
```

Add a `useEffect` to load dataset versions when the component mounts (independent of the model load):

```tsx
useEffect(() => {
  let isMounted = true;
  async function loadVersions() {
    try {
      const data = await listDatasetVersions(projectId);
      if (!isMounted) return;
      const rows = (data.items ?? []).flatMap((item) => {
        const v = item.version as Record<string, unknown>;
        const id = typeof v.dataset_version_id === "string" ? v.dataset_version_id : "";
        const name = typeof v.name === "string" ? v.name : id;
        const task = typeof v.task === "string" ? v.task : "";
        const labels = (v.labels as Record<string, unknown>) ?? {};
        const schema = (labels.label_schema as Record<string, unknown>) ?? {};
        const classOrder = Array.isArray(schema.class_order)
          ? (schema.class_order as string[])
          : [];
        const classes = Array.isArray(schema.classes)
          ? (schema.classes as Array<Record<string, unknown>>)
          : [];
        const classNames = classOrder.map((cid) => {
          const row = classes.find((c) => c.category_id === cid);
          return typeof row?.name === "string" ? row.name : cid;
        });
        return id ? [{ id, name, task, numClasses: classOrder.length, classOrder, classNames }] : [];
      });
      setAllDatasetVersions(rows);
    } catch {
      // non-fatal — model editor still works without this list
    }
  }
  void loadVersions();
  return () => { isMounted = false; };
}, [projectId]);
```

**Step 3: Derive computed values from draftConfig for the Step 1 selectors**

Near the existing `const input = asRecord(draftConfig?.input);` block, add:

```tsx
const sourceDataset = asRecord(draftConfig?.source_dataset);
const modelTask = typeof sourceDataset.task === "string" ? sourceDataset.task : "";
const modelManifestId = typeof sourceDataset.manifest_id === "string" ? sourceDataset.manifest_id : "";
const modelFamily = typeof architecture.family === "string" ? architecture.family : "";

// Families available for the current task
const familiesForTask = (familiesMetadata.families as Array<{ name: string; task: string; allowed_backbones: string[] }>)
  .filter((f) => f.task === modelTask);

// Dataset versions for the current task
const datasetsForTask = allDatasetVersions.filter((v) => v.task === modelTask);

// Allowed backbones for the current family
const currentFamilyMeta = familiesForTask.find((f) => f.name === modelFamily);
const allowedBackbones = currentFamilyMeta?.allowed_backbones ?? [];
```

**Step 4: Add Step 1 section to editorContent**

At the very top of `editorContent` (before the existing `<section>` for Step 2), add:

```tsx
<section className="model-builder-step">
  <h4>Step 1: Source</h4>
  <label className="project-field">
    <span>Task</span>
    <select
      value={modelTask}
      onChange={(event) => {
        const newTask = event.target.value;
        // Find first family for this task
        const firstFamily = (familiesMetadata.families as Array<{ name: string; task: string; allowed_backbones: string[] }>)
          .find((f) => f.task === newTask);
        if (!firstFamily) return;
        // Find first dataset version for this task
        const firstDataset = allDatasetVersions.find((v) => v.task === newTask);
        setDraftConfig((current) => {
          if (!current) return current;
          let next = cloneModelConfig(current);
          // Apply family defaults (regenerates arch/loss/outputs)
          next = setArchitectureFamily(next, firstFamily.name, familiesMetadata.families);
          // Apply dataset if available
          if (firstDataset) {
            next = setSourceDataset(next, {
              dataset_version_id: firstDataset.id,
              task: firstDataset.task,
              num_classes: firstDataset.numClasses,
              class_order: firstDataset.classOrder,
              class_names: firstDataset.classNames,
            });
          } else {
            // At least update the task on source_dataset
            const src = asRecord(next.source_dataset);
            src.task = newTask;
            next.source_dataset = src;
          }
          return next;
        });
      }}
    >
      {["classification", "detection", "segmentation"].map((t) => (
        <option key={t} value={t}>{t}</option>
      ))}
    </select>
  </label>
  <label className="project-field">
    <span>Dataset Version</span>
    <select
      value={modelManifestId}
      disabled={datasetsForTask.length === 0}
      onChange={(event) => {
        const versionId = event.target.value;
        const found = allDatasetVersions.find((v) => v.id === versionId);
        if (!found) return;
        setDraftConfig((current) =>
          current
            ? setSourceDataset(current, {
                dataset_version_id: found.id,
                task: found.task,
                num_classes: found.numClasses,
                class_order: found.classOrder,
                class_names: found.classNames,
              })
            : current,
        );
      }}
    >
      {datasetsForTask.length === 0 ? (
        <option value="">No dataset versions for this task</option>
      ) : null}
      {datasetsForTask.map((v) => (
        <option key={v.id} value={v.id}>
          {v.name} ({v.numClasses} classes)
        </option>
      ))}
    </select>
  </label>
  <label className="project-field">
    <span>Architecture Family</span>
    <select
      value={modelFamily}
      disabled={familiesForTask.length === 0}
      onChange={(event) => {
        const familyName = event.target.value;
        setDraftConfig((current) =>
          current
            ? setArchitectureFamily(current, familyName, familiesMetadata.families)
            : current,
        );
      }}
    >
      {familiesForTask.length === 0 ? (
        <option value="">No families for this task</option>
      ) : null}
      {familiesForTask.map((f) => (
        <option key={f.name} value={f.name}>{f.name}</option>
      ))}
    </select>
  </label>
</section>
```

**Step 5: Update backbone selector to use allowedBackbones**

Find the existing backbone `<select>` in "Step 3: Backbone". Replace the static `BACKBONE_OPTIONS.map(...)` with the dynamic list:

```tsx
{(allowedBackbones.length > 0 ? allowedBackbones : Array.from(BACKBONE_OPTIONS)).map((value) => (
  <option key={value} value={value}>
    {value}
  </option>
))}
```

Also change the `onChange` handler to use `setBackbone`:
```tsx
onChange={(event) => {
  setDraftConfig((current) =>
    current ? setBackbone(current, event.target.value) : current,
  );
}}
```

**Step 6: Verify in dev**

With `make dev-web` running:
1. Open an existing model → should see Step 1 with task/dataset/family dropdowns pre-populated from `source_dataset`
2. Change family → backbone options should update, arch/loss/outputs regenerate
3. Change dataset → classes count updates, head.num_classes updates
4. Change task → resets family + dataset to first valid for that task
5. Save → model persists new config

**Step 7: Commit**

```bash
git add "apps/web/src/app/projects/[projectId]/models/[modelId]/page.tsx"
git commit -m "feat: add editable task/dataset/family selectors to model editor"
```

---

## Final verification checklist

- [ ] `grep -rn '"bbox"' apps/trainer/src/` returns no task-key uses
- [ ] `make test-api-focused` passes
- [ ] `cd apps/web && node --test tests/modelConfigEditor.test.js` all green
- [ ] Start a detection experiment end-to-end: create bbox dataset version → create model (pick bbox dataset) → create experiment → start training → no "unsupported task" error
- [ ] Model editor: switching family changes backbone options; switching dataset updates class count
