from __future__ import annotations

import json
from pathlib import Path

from httpx import AsyncClient
import pytest
from sheriff_api.config import get_settings

from .api_test_helpers import (
    _create_classification_project_model,
    _create_dataset_version_for_task,
    _create_detection_project_with_dataset_version,
    _create_project_model,
    _seed_experiment_run_artifacts,
    assert_api_error,
)

@pytest.mark.asyncio
async def test_experiment_create_from_model_returns_draft_record(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-create")
    response = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "run-a"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == project_id
    assert payload["model_id"] == model_id
    assert payload["status"] == "draft"
    assert payload["name"] == "run-a"
    assert payload["config_json"]["schema_version"] == "0.1"
    assert payload["config_json"]["dataset_version_id"]
    assert payload["checkpoints"]
    assert payload["metrics"] == []


@pytest.mark.asyncio
async def test_experiment_update_persists_when_draft(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-update")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id},
    )
    assert created.status_code == 200
    experiment = created.json()

    updated_config = experiment["config_json"]
    updated_config["epochs"] = 8
    updated_config["batch_size"] = 4
    updated_config["optimizer"]["lr"] = 0.0005
    update = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{experiment['id']}",
        json={"name": "run-updated", "config_json": updated_config, "selected_checkpoint_kind": "latest"},
    )
    assert update.status_code == 200
    payload = update.json()
    assert payload["name"] == "run-updated"
    assert payload["config_json"]["epochs"] == 8
    assert payload["config_json"]["batch_size"] == 4
    assert payload["config_json"]["optimizer"]["lr"] == 0.0005
    assert payload["artifacts_json"]["selected_checkpoint_kind"] == "latest"

    detail = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment['id']}")
    assert detail.status_code == 200
    assert detail.json()["name"] == "run-updated"


@pytest.mark.asyncio
async def test_experiment_start_generates_metrics_and_checkpoints(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-start")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "config_overrides": {"epochs": 4}},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    import sheriff_api.routers.experiments as experiments_router

    calls: list[dict] = []

    async def _enqueue(job_payload: dict) -> None:
        calls.append(job_payload)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()

    assert started.status_code == 200
    assert started.json()["ok"] is True
    payload = started.json()
    assert payload["status"] == "queued"
    assert isinstance(payload["attempt"], int) and payload["attempt"] >= 1
    assert payload["job_id"]
    assert len(calls) == 1
    assert calls[0]["job_id"] == payload["job_id"]
    assert calls[0]["attempt"] == payload["attempt"]
    assert calls[0]["job_type"] == "train"

    detail = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["status"] == "queued"
    assert detail_payload["current_run_attempt"] == payload["attempt"]
    assert detail_payload["active_job_id"] == payload["job_id"]
    assert detail_payload["metrics"] == []


@pytest.mark.asyncio
async def test_experiment_start_rebuilds_missing_dataset_zip(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-rebuild-zip")

    versions = await client.get(f"/api/v1/projects/{project_id}/datasets/versions")
    assert versions.status_code == 200
    version_items = versions.json()["items"]
    assert len(version_items) == 1
    dataset_version_id = version_items[0]["version"]["dataset_version_id"]

    export = await client.post(f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export")
    assert export.status_code == 200
    content_hash = export.json()["hash"]

    settings = get_settings()
    zip_path = Path(settings.storage_root) / "exports" / project_id / f"{content_hash}.zip"
    assert zip_path.exists()
    zip_path.unlink()
    assert zip_path.exists() is False

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "rebuild-missing-zip"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    import sheriff_api.routers.experiments as experiments_router

    calls: list[dict] = []

    async def _enqueue(job_payload: dict) -> None:
        calls.append(job_payload)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()

    assert started.status_code == 200
    assert started.json()["ok"] is True
    assert len(calls) == 1
    assert zip_path.exists()


@pytest.mark.asyncio
async def test_experiment_analytics_endpoint_returns_series_and_honors_max_points(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-analytics")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "analytics-run"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1)

    analytics = await client.get(f"/api/v1/projects/{project_id}/experiments/analytics?max_points=2")
    assert analytics.status_code == 200
    payload = analytics.json()
    assert "items" in payload
    assert "available_series" in payload
    item = next((row for row in payload["items"] if row["experiment_id"] == experiment_id), None)
    assert item is not None
    assert item["series"]["epochs"] == [2, 3]
    assert len(item["series"]["val_accuracy"]) == 2
    assert "val_accuracy" in payload["available_series"]
    assert item["runtime"]["device_selected"] == "cuda"


@pytest.mark.asyncio
async def test_experiment_analytics_reports_custom_and_legacy_augmentation_modes(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-analytics-augmentation")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "analytics-aug"},
    )
    assert created.status_code == 200
    experiment_payload = created.json()
    config_json = dict(experiment_payload["config_json"])
    config_json["augmentation_profile"] = "custom"
    config_json["augmentation_spec_version"] = 1
    config_json["augmentation_steps"] = [
        {"type": "horizontal_flip", "p": 0.5, "params": {}},
        {"type": "rotate", "p": 1.0, "params": {"degrees": 8}},
    ]
    updated = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{experiment_payload['id']}",
        json={"config_json": config_json},
    )
    assert updated.status_code == 200

    analytics = await client.get(f"/api/v1/projects/{project_id}/experiments/analytics")
    assert analytics.status_code == 200
    item = next(row for row in analytics.json()["items"] if row["experiment_id"] == experiment_payload["id"])
    assert item["config"]["augmentation"] == "custom"
    assert item["config"]["augmentation_mode"] == "custom"
    assert item["config"]["augmentation_summary"].startswith("custom:")

    detection_project_id, task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="exp-analytics-legacy-detection"
    )
    created_model = await client.post(
        f"/api/v1/projects/{detection_project_id}/models",
        json={"dataset_version_id": dataset_version_id},
    )
    assert created_model.status_code == 200
    detection_experiment = await client.post(
        f"/api/v1/projects/{detection_project_id}/experiments",
        json={"model_id": created_model.json()["id"], "name": "legacy-detection"},
    )
    assert detection_experiment.status_code == 200
    legacy_config = dict(detection_experiment.json()["config_json"])
    legacy_config["task_id"] = task_id
    legacy_config["augmentation_profile"] = "light"
    legacy_config.pop("augmentation_spec_version", None)
    legacy_config.pop("augmentation_steps", None)
    updated_detection = await client.put(
        f"/api/v1/projects/{detection_project_id}/experiments/{detection_experiment.json()['id']}",
        json={"config_json": legacy_config},
    )
    assert updated_detection.status_code == 200

    detection_analytics = await client.get(f"/api/v1/projects/{detection_project_id}/experiments/analytics")
    assert detection_analytics.status_code == 200
    detection_item = next(row for row in detection_analytics.json()["items"] if row["experiment_id"] == detection_experiment.json()["id"])
    assert detection_item["config"]["augmentation"] == "none"
    assert detection_item["config"]["augmentation_mode"] == "none"
    assert detection_item["config"]["augmentation_summary"] == "none"


@pytest.mark.asyncio
async def test_experiment_evaluation_endpoint_returns_attempt_payload(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-evaluation")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "evaluation-run"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=3)

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/evaluation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == 3
    assert payload["schema_version"] == "1"
    assert payload["overall"]["accuracy"] == 0.75
    assert payload["provenance"]["dataset_version_id"] == "dv-1"
    assert payload["provenance"]["attempt"] == 3
    assert payload["provenance"]["project_id"] == project_id


@pytest.mark.asyncio
async def test_experiment_evaluation_endpoint_returns_not_found_when_missing(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-evaluation-missing")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "evaluation-missing"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/evaluation")
    assert_api_error(
        response,
        status_code=404,
        code="evaluation_not_found",
        message="Evaluation not available for this experiment",
    )


@pytest.mark.asyncio
async def test_experiment_runtime_endpoint_returns_runtime_payload(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-runtime")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "runtime-run"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=2)

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/runtime")
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == 2
    assert payload["device_selected"] == "cuda"
    assert payload["amp_enabled"] is True


@pytest.mark.asyncio
async def test_experiment_runtime_endpoint_returns_not_found_when_missing(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-runtime-missing")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "runtime-missing"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1)

    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    run_dir = experiment_dir / "runs" / "1"
    for runtime_path in [run_dir / "runtime.json", experiment_dir / "runtime.json"]:
        if runtime_path.exists():
            runtime_path.unlink()

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/runtime")
    assert_api_error(
        response,
        status_code=404,
        code="runtime_not_found",
        message="Runtime not available for this experiment",
    )


@pytest.mark.asyncio
async def test_experiment_onnx_endpoint_returns_metadata_and_urls(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-onnx")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "onnx-run"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=2,
        include_onnx=True,
        onnx_status="exported",
    )

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx")
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == 2
    assert payload["status"] == "exported"
    assert payload["model_onnx_url"].endswith("/onnx/download?file=model")
    assert payload["metadata_url"].endswith("/onnx/download?file=metadata")
    assert payload["input_shape"] == [3, 224, 224]
    assert payload["class_names"] == ["one", "two"]


@pytest.mark.asyncio
async def test_experiment_onnx_download_endpoints_stream_model_and_metadata(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-onnx-download")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "onnx-download"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=1,
        include_onnx=True,
        onnx_status="exported",
    )

    model_response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download?file=model")
    assert model_response.status_code == 200
    assert model_response.headers["content-type"].startswith("application/octet-stream")
    assert model_response.content == b"fake-onnx-binary-content"

    metadata_response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx/download?file=metadata")
    assert metadata_response.status_code == 200
    assert metadata_response.headers["content-type"].startswith("application/json")
    metadata_payload = metadata_response.json()
    assert metadata_payload["status"] == "exported"
    assert metadata_payload["input_shape"] == [3, 224, 224]


@pytest.mark.asyncio
async def test_experiment_onnx_endpoint_returns_not_found_when_missing(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-onnx-missing")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "onnx-missing"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/onnx")
    assert_api_error(
        response,
        status_code=404,
        code="onnx_not_found",
        message="ONNX export not available for this experiment",
    )

@pytest.mark.asyncio
async def test_experiment_logs_endpoint_returns_chunk_and_cursor(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-logs")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "logs-run"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1)

    response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/logs?from_byte=0&max_bytes=32"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == 1
    assert payload["from_byte"] == 0
    assert payload["to_byte"] > 0
    assert "epoch=1" in payload["content"]


@pytest.mark.asyncio
async def test_experiment_logs_endpoint_resets_cursor_when_from_byte_exceeds_file(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-logs-reset")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "logs-reset"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1)

    response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/logs?from_byte=99999&max_bytes=32"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == 1
    assert payload["from_byte"] == 0
    assert payload["to_byte"] > 0


@pytest.mark.asyncio
async def test_experiment_logs_endpoint_returns_requested_attempt_chunk(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-logs-attempt")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "logs-attempt"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=1,
        log_content="attempt=1 epoch=1 train_loss=0.90\n",
    )
    _seed_experiment_run_artifacts(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=2,
        log_content="attempt=2 epoch=1 train_loss=0.40\n",
    )

    response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/logs?attempt=1&from_byte=0&max_bytes=64"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == 1
    assert "attempt=1" in payload["content"]
    assert "attempt=2" not in payload["content"]


@pytest.mark.asyncio
async def test_experiment_logs_endpoint_returns_not_found_when_missing(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-logs-missing")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "logs-missing"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    response = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/logs")
    assert_api_error(
        response,
        status_code=404,
        code="logs_not_found",
        message="Training logs not available for this experiment",
    )


@pytest.mark.asyncio
async def test_experiment_samples_endpoint_filters_rows_and_returns_attempt(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-samples")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "samples-run"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=2)

    misclassified = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/samples?mode=misclassified&true_class_index=0&pred_class_index=1&limit=10"
    )
    assert misclassified.status_code == 200
    payload = misclassified.json()
    assert payload["attempt"] == 2
    assert payload["mode"] == "misclassified"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["asset_id"] == "asset-1"

    lowest_correct = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/samples?mode=lowest_confidence_correct&limit=1"
    )
    assert lowest_correct.status_code == 200
    payload_correct = lowest_correct.json()
    assert payload_correct["attempt"] == 2
    assert payload_correct["mode"] == "lowest_confidence_correct"
    assert len(payload_correct["items"]) == 1
    assert payload_correct["items"][0]["pred_class_index"] == payload_correct["items"][0]["true_class_index"]


@pytest.mark.asyncio
async def test_experiment_events_sse_smoke(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-events")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "config_overrides": {"epochs": 3}},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    import sheriff_api.routers.experiments as experiments_router

    async def _enqueue(_job_payload: dict) -> None:
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    start = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()
    assert start.status_code == 200
    attempt = start.json()["attempt"]

    saw_status = False
    saw_line = False
    async with client.stream(
        "GET",
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/events?attempt={attempt}&from_line=0&follow=false",
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            assert "line" in event
            assert "event" in event
            saw_line = True
            event_type = event["event"].get("type")
            if event_type == "status":
                saw_status = True
                break
            if saw_status:
                break

    assert saw_status is True
    assert saw_line is True


@pytest.mark.asyncio
async def test_experiment_cancel_stops_run(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-cancel")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "config_overrides": {"epochs": 12}},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    import sheriff_api.routers.experiments as experiments_router

    async def _enqueue(_job_payload: dict) -> None:
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()
    assert started.status_code == 200

    canceled = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["ok"] is True
    assert canceled.json()["status"] == "canceled"

    detail = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "canceled"


@pytest.mark.asyncio
async def test_experiment_cancel_running_sets_cancel_requested(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-cancel-running")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "config_overrides": {"epochs": 2}},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    import sheriff_api.routers.experiments as experiments_router

    async def _enqueue(_job_payload: dict) -> None:
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()
    assert started.status_code == 200

    experiments_router.experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="running")
    canceled = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "running"

    status_row = experiments_router.experiment_store.get_status_row(project_id, experiment_id)
    assert status_row["cancel_requested"] is True


@pytest.mark.asyncio
async def test_project_delete_removes_experiment_storage(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-delete-cleanup")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id},
    )
    assert created.status_code == 200

@pytest.mark.asyncio
async def test_experiment_create_defaults_to_model_source_dataset_not_latest_active_version(client: AsyncClient) -> None:
    project_id, task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="exp-source-dataset-default"
    )

    created_model = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": dataset_version_id},
    )
    assert created_model.status_code == 200
    model_id = created_model.json()["id"]

    newer_dataset_version_id = await _create_dataset_version_for_task(
        client,
        project_id=project_id,
        task_id=task_id,
        name="v2",
        set_active=True,
    )
    assert newer_dataset_version_id != dataset_version_id

    created_experiment = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "uses-model-source-dataset"},
    )
    assert created_experiment.status_code == 200
    payload = created_experiment.json()
    assert payload["config_json"]["dataset_version_id"] == dataset_version_id


@pytest.mark.asyncio
async def test_experiment_create_rejects_dataset_version_mismatch_with_model_source_dataset(client: AsyncClient) -> None:
    project_id, task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="exp-source-dataset-mismatch"
    )

    created_model = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": dataset_version_id},
    )
    assert created_model.status_code == 200
    model_id = created_model.json()["id"]

    newer_dataset_version_id = await _create_dataset_version_for_task(
        client,
        project_id=project_id,
        task_id=task_id,
        name="v2",
        set_active=True,
    )

    created_experiment = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "mismatch", "dataset_version_id": newer_dataset_version_id},
    )
    assert_api_error(
        created_experiment,
        status_code=409,
        code="model_dataset_mismatch",
        message="Model source dataset does not match the selected dataset version",
    )
    payload = created_experiment.json()
    assert payload["error"]["details"]["dataset_version_id"] == newer_dataset_version_id
    assert payload["error"]["details"]["issues"][0]["path"] == "source_dataset.manifest_id"


@pytest.mark.asyncio
async def test_experiment_create_defaults_augmentation_by_task_and_stamps_spec_version(client: AsyncClient) -> None:
    classification_project_id, classification_model_id, _classification_task_id = await _create_classification_project_model(
        client,
        project_name="exp-augmentation-default-classification",
    )
    classification_experiment = await client.post(
        f"/api/v1/projects/{classification_project_id}/experiments",
        json={"model_id": classification_model_id, "name": "classification-defaults"},
    )
    assert classification_experiment.status_code == 200
    classification_config = classification_experiment.json()["config_json"]
    assert classification_config["augmentation_profile"] == "light"
    assert classification_config["augmentation_spec_version"] == 1
    assert classification_config["augmentation_steps"] == []

    detection_project_id, _detection_task_id, detection_dataset_version_id = await _create_detection_project_with_dataset_version(
        client,
        project_name="exp-augmentation-default-detection",
    )
    detection_model = await client.post(
        f"/api/v1/projects/{detection_project_id}/models",
        json={"dataset_version_id": detection_dataset_version_id},
    )
    assert detection_model.status_code == 200
    detection_experiment = await client.post(
        f"/api/v1/projects/{detection_project_id}/experiments",
        json={"model_id": detection_model.json()["id"], "name": "detection-defaults"},
    )
    assert detection_experiment.status_code == 200
    detection_config = detection_experiment.json()["config_json"]
    assert detection_config["augmentation_profile"] == "none"
    assert detection_config["augmentation_spec_version"] == 1
    assert detection_config["augmentation_steps"] == []


@pytest.mark.asyncio
async def test_experiment_update_rejects_invalid_custom_augmentation_configs(client: AsyncClient) -> None:
    project_id, model_id, _task_id = await _create_classification_project_model(
        client,
        project_name="exp-augmentation-validation",
    )
    created_experiment = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "invalid-custom-augmentation"},
    )
    assert created_experiment.status_code == 200
    base_config = dict(created_experiment.json()["config_json"])

    empty_custom = dict(base_config)
    empty_custom["augmentation_profile"] = "custom"
    empty_custom["augmentation_spec_version"] = 1
    empty_custom["augmentation_steps"] = []
    empty_response = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{created_experiment.json()['id']}",
        json={"config_json": empty_custom},
    )
    assert_api_error(
        empty_response,
        status_code=422,
        code="validation_error",
        message="Experiment config validation failed",
    )

    bad_rotate = dict(base_config)
    bad_rotate["augmentation_profile"] = "custom"
    bad_rotate["augmentation_spec_version"] = 1
    bad_rotate["augmentation_steps"] = [{"type": "rotate", "p": 1.0, "params": {}}]
    rotate_response = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{created_experiment.json()['id']}",
        json={"config_json": bad_rotate},
    )
    assert_api_error(
        rotate_response,
        status_code=422,
        code="validation_error",
        message="Experiment config validation failed",
    )

    bad_color_jitter = dict(base_config)
    bad_color_jitter["augmentation_profile"] = "custom"
    bad_color_jitter["augmentation_spec_version"] = 1
    bad_color_jitter["augmentation_steps"] = [
        {"type": "color_jitter", "p": 1.2, "params": {"brightness": 0.1, "bogus": 0.2}},
    ]
    color_jitter_response = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{created_experiment.json()['id']}",
        json={"config_json": bad_color_jitter},
    )
    assert_api_error(
        color_jitter_response,
        status_code=422,
        code="validation_error",
        message="Experiment config validation failed",
    )

    valid_custom = dict(base_config)
    valid_custom["augmentation_profile"] = "custom"
    valid_custom["augmentation_spec_version"] = 1
    valid_custom["augmentation_steps"] = [
        {"type": "horizontal_flip", "p": 0.5, "params": {}},
        {"type": "color_jitter", "p": 1.0, "params": {"brightness": 0.1, "contrast": 0.1}},
    ]
    valid_response = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{created_experiment.json()['id']}",
        json={"config_json": valid_custom},
    )
    assert valid_response.status_code == 200
    assert valid_response.json()["config_json"]["augmentation_profile"] == "custom"


@pytest.mark.asyncio
async def test_experiment_start_rejects_dataset_version_mismatch_with_model_source_dataset(client: AsyncClient) -> None:
    project_id, task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="exp-start-source-dataset-mismatch"
    )

    created_model = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": dataset_version_id},
    )
    assert created_model.status_code == 200
    model_id = created_model.json()["id"]

    newer_dataset_version_id = await _create_dataset_version_for_task(
        client,
        project_id=project_id,
        task_id=task_id,
        name="v2",
        set_active=True,
    )

    created_experiment = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "draft"},
    )
    assert created_experiment.status_code == 200
    experiment_payload = created_experiment.json()
    updated_config = dict(experiment_payload["config_json"])
    updated_config["dataset_version_id"] = newer_dataset_version_id

    updated_experiment = await client.put(
        f"/api/v1/projects/{project_id}/experiments/{experiment_payload['id']}",
        json={"config_json": updated_config},
    )
    assert updated_experiment.status_code == 200

    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_payload['id']}/start")
    assert_api_error(
        started,
        status_code=409,
        code="model_dataset_mismatch",
        message="Model source dataset does not match the selected dataset version",
    )
