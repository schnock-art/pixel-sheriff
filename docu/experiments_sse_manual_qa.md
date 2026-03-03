# Experiments Manual QA (Phase 3)

Use this checklist to verify:
- create/save/start/cancel + live SSE updates + surfaced trainer failure messages
- classification analytics and deep-dive dashboard data flows
- attempt-aware evaluation artifacts + API responses
- runtime/log observability APIs + runtime badges/log tail UI

## Preconditions
- `docker compose up -d --build api trainer web db redis`
- Web is reachable at `http://localhost:3010`
- API is reachable at `http://localhost:8010/api/v1/health`

## API Smoke Script (PowerShell)
Run step-by-step and keep returned IDs in variables.

```powershell
$api = "http://localhost:8010/api/v1"

# 1) Create project
$project = Invoke-RestMethod -Method Post -Uri "$api/projects" -ContentType "application/json" -Body (@{
  name = "qa-exp-sse"
  task_type = "classification_single"
} | ConvertTo-Json)
$projectId = $project.id

# 2) Create category + upload + annotate + export (needed so model/experiment creation works)
$cat = Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/categories" -ContentType "application/json" -Body (@{
  name = "class_a"
} | ConvertTo-Json)

$tmpImage = Join-Path $env:TEMP "qa-exp-sse.jpg"
[System.IO.File]::WriteAllBytes($tmpImage, [byte[]](255,216,255,217))
$uploadForm = @{
  file = Get-Item $tmpImage
}
$asset = Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/assets/upload" -Form $uploadForm

Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/annotations" -ContentType "application/json" -Body (@{
  asset_id = $asset.id
  status = "approved"
  payload_json = @{ category_ids = @($cat.id) }
} | ConvertTo-Json -Depth 8) | Out-Null

Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/exports" -ContentType "application/json" -Body (@{
  selection_criteria_json = @{ status = "approved" }
} | ConvertTo-Json -Depth 8) | Out-Null

# 3) Create model
$model = Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/models" -ContentType "application/json" -Body "{}"
$modelId = $model.id

# 4) Create experiment
$experiment = Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/experiments" -ContentType "application/json" -Body (@{
  model_id = $modelId
} | ConvertTo-Json)
$experimentId = $experiment.id

# 5) Start experiment
Invoke-RestMethod -Method Post -Uri "$api/projects/$projectId/experiments/$experimentId/start" -ContentType "application/json" -Body "{}"

# 6) Poll detail a few times to confirm metrics/checkpoints appear
1..5 | ForEach-Object {
  Start-Sleep -Milliseconds 700
  $detail = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/$experimentId"
  "status=$($detail.status), metrics=$($detail.metrics.Count), last_epoch=$($detail.summary_json.last_epoch)"
}

# 7) Check analytics/evaluation/samples contracts
$analytics = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/analytics?max_points=50"
"analytics_items=$($analytics.items.Count), available_series=$($analytics.available_series -join ',')"
"runtime_device_on_first_item=$($analytics.items[0].runtime.device_selected)"

$evaluation = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/$experimentId/evaluation"
"evaluation_attempt=$($evaluation.attempt), task=$($evaluation.task), num_samples=$($evaluation.num_samples)"

$samples = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/$experimentId/samples?mode=misclassified&limit=25"
"samples_attempt=$($samples.attempt), samples_count=$($samples.items.Count)"

# 8) Runtime/log observability contracts
$runtime = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/$experimentId/runtime"
"runtime_device=$($runtime.device_selected), amp=$($runtime.amp_enabled), torch=$($runtime.torch_version)"

$logChunk1 = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/$experimentId/logs?from_byte=0&max_bytes=4096"
"logs_chunk_1 from=$($logChunk1.from_byte) to=$($logChunk1.to_byte) bytes=$($logChunk1.content.Length)"

$logChunk2 = Invoke-RestMethod -Method Get -Uri "$api/projects/$projectId/experiments/$experimentId/logs?from_byte=$($logChunk1.to_byte)&max_bytes=4096"
"logs_chunk_2 from=$($logChunk2.from_byte) to=$($logChunk2.to_byte) bytes=$($logChunk2.content.Length)"
```

## Browser SSE Checklist
- Open `http://localhost:3010/projects/{projectId}/models/{modelId}`.
- Click `Train Model`.
- If modal appears:
  - Verify `Continue` opens latest experiment.
  - Verify `New run` creates and opens a new experiment.
- On experiment page:
  - Edit `epochs`/`batch size`/`lr`, click `Save`, refresh, verify values persist.
  - Verify default `Advanced Parameters -> Num Workers` is `0` (recommended in Docker).
  - Click `Start Training`.
  - Verify status changes to `running`.
  - Verify runtime badge appears beside status (`CUDA`/`CPU`/`MPS`) once runtime info is available.
  - Verify chart updates every epoch without page refresh.
  - If `evaluation.eval_interval_epochs > 1`, verify skipped epochs still render a metric point and `val_*` is empty/null for those rows.
  - Verify checkpoints (`latest`, `best_loss`, `best_metric`) update.
  - Refresh mid-run and verify history remains and updates continue.
  - Open `Runtime & Logs` panel:
    - verify runtime fields are populated (`device`, `cuda_available`, `amp_enabled`, torch/torchvision versions)
    - enable auto-refresh while run is active and verify log content keeps appending
    - verify refresh button appends content and cursor moves forward (`from_byte -> to_byte`)
    - complete/cancel run and verify auto-refresh stops on terminal status
  - Click `Cancel` during run and verify terminal status `canceled`.
  - If run fails, verify toast shows failure reason and header shows `Last run error: ...`.

## Browser Analytics + Dashboard Checklist
- Open `http://localhost:3010/projects/{projectId}/experiments`.
- Verify analytics section appears above the experiments table.
- Verify each experiment row can show runtime badge (`CUDA`/`CPU`/`MPS`) when runtime data exists.
- Verify summary cards show:
  - `Best accuracy`
  - `Lowest val loss`
  - `Total runs`
  - `Failures`
- Verify run-selection defaults:
  - last 3 completed runs selected
  - failed runs hidden until `Show failed` is enabled
  - best run is visually highlighted
- In comparison chart:
  - switch metric dropdown (for example `val_accuracy`, `val_loss`)
  - toggle log scale and confirm chart re-renders without errors
- In hyperparameter scatter:
  - change x/y selectors
  - hover a dot and verify tooltip shows experiment name and x/y values
  - click a dot and verify navigation to experiment detail page
- On experiment detail page (`/experiments/{experimentId}`):
  - verify header action is `Back to Experiments` and returns to `/projects/{projectId}/experiments`
  - verify dashboard appears below existing training sections
  - verify chart tabs (`Loss`, `Accuracy`, `F1/Precision/Recall`) switch correctly
  - verify confusion normalization modes:
    - `none` = raw counts
    - `by_true` = row-normalized
    - `by_pred` = column-normalized
    - no `NaN`/`inf` on zero-sum rows/columns
  - click a confusion cell and verify sample drill-down panel/modal opens
  - verify per-class metrics table supports sorting
  - verify prediction explorer mode/class/limit filters update thumbnails
  - verify served attempt label is shown for evaluation/sample data

## SSE Endpoint Spot Check (raw stream)
In a separate shell (after start), inspect SSE output:

```powershell
curl.exe -N "http://localhost:8010/api/v1/projects/$projectId/experiments/$experimentId/events"
```

Expected events:
- `{"type":"status","status":"running"}`
- repeated `{"type":"metric", ...}`
- occasional `{"type":"checkpoint", ...}`
- terminal `{"type":"done","status":"completed|canceled|failed"}`
- failed done events include `message` + `error_code` fields for UI/runtime diagnostics.

## Artifact Filesystem Spot Check (optional)
After a completed classification run, verify files exist under storage root:

- `experiments/{project_id}/{experiment_id}/runs/{attempt}/evaluation.json`
- `experiments/{project_id}/{experiment_id}/runs/{attempt}/predictions.jsonl`
- `experiments/{project_id}/{experiment_id}/runs/{attempt}/predictions.meta.json`
- `experiments/{project_id}/{experiment_id}/runs/{attempt}/runtime.json`
- `experiments/{project_id}/{experiment_id}/runs/{attempt}/training.log`
- `experiments/{project_id}/{experiment_id}/evaluation.json` (latest mirror)
- `experiments/{project_id}/{experiment_id}/predictions.jsonl` (latest mirror)
- `experiments/{project_id}/{experiment_id}/predictions.meta.json` (latest mirror)
- `experiments/{project_id}/{experiment_id}/runtime.json` (latest mirror)
