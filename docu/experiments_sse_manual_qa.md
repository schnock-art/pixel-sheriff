# Experiments SSE Manual QA (Phase 2)

Use this checklist to verify create/save/start/cancel + live SSE updates + surfaced trainer failure messages.

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
  - Verify chart updates every epoch without page refresh.
  - Verify checkpoints (`latest`, `best_loss`, `best_metric`) update.
  - Refresh mid-run and verify history remains and updates continue.
  - Click `Cancel` during run and verify terminal status `canceled`.
  - If run fails, verify toast shows failure reason and header shows `Last run error: ...`.

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
