import json
from io import BytesIO
from pathlib import Path
import zipfile

from httpx import AsyncClient
import pytest

from sheriff_api.config import get_settings
from test_api import _create_project_model, _seed_experiment_run_artifacts, assert_api_error


def _parse_sse_events(raw_text: str) -> list[dict]:
    events: list[dict] = []
    decoder = json.JSONDecoder()
    for chunk in raw_text.split("data: "):
        if not chunk.strip():
            continue
        payload = chunk.split("\n\n", 1)[0].strip()
        if not payload:
            continue
        event, _ = decoder.raw_decode(payload)
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_experiments_router_exports_shared_singletons() -> None:
    import sheriff_api.routers.experiments as experiments_router
    from sheriff_api.routers.experiments.runs import train_queue as runs_queue
    from sheriff_api.routers.experiments.shared import experiment_store as shared_store

    assert experiments_router.router is not None
    assert experiments_router.train_queue is runs_queue
    assert experiments_router.experiment_store is shared_store


@pytest.mark.asyncio
async def test_experiment_start_queue_failure_sets_failed_and_emits_done_event(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-queue-failure")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "queue-fails"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    import sheriff_api.routers.experiments as experiments_router

    async def _enqueue(_job_payload: dict) -> None:
        raise RuntimeError("queue unavailable in test")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()

    assert_api_error(
        started,
        status_code=503,
        code="train_queue_unavailable",
        message="Training queue is unavailable",
    )

    detail = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["status"] == "failed"
    assert isinstance(detail_payload.get("current_run_attempt"), int)
    attempt = detail_payload["current_run_attempt"]

    events_response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/events?attempt={attempt}&from_line=0&follow=false"
    )
    assert events_response.status_code == 200
    events = _parse_sse_events(events_response.text)
    assert any(event.get("event", {}).get("type") == "done" for event in events)
    assert any(event.get("event", {}).get("status") == "failed" for event in events)


@pytest.mark.asyncio
async def test_experiment_events_snapshot_for_no_attempt_and_terminal_state(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-events-no-attempt")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "events-snapshot"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    draft_response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/events?from_line=0&follow=false"
    )
    assert draft_response.status_code == 200
    draft_events = _parse_sse_events(draft_response.text)
    assert draft_events[0]["event"]["type"] == "status"
    assert draft_events[0]["event"]["status"] == "draft"
    assert draft_events[1]["event"]["type"] == "done"
    assert draft_events[1]["event"]["status"] == "draft"

    import sheriff_api.routers.experiments as experiments_router

    experiments_router.experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="completed")
    completed_response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/events?from_line=0&follow=false"
    )
    assert completed_response.status_code == 200
    completed_events = _parse_sse_events(completed_response.text)
    assert completed_events[0]["event"]["status"] == "completed"
    assert completed_events[1]["event"]["type"] == "done"
    assert completed_events[1]["event"]["status"] == "completed"


@pytest.mark.asyncio
async def test_experiment_samples_falls_back_to_evaluation_when_predictions_missing(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-samples-eval-fallback")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "samples-fallback"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    attempt = 2
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=attempt)

    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    run_dir = experiment_dir / "runs" / str(attempt)
    for predictions_path in [run_dir / "predictions.jsonl", experiment_dir / "predictions.jsonl"]:
        if predictions_path.exists():
            predictions_path.unlink()

    response = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/samples?mode=misclassified&limit=10"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"] == attempt
    assert payload["mode"] == "misclassified"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["asset_id"] == "asset-1"


@pytest.mark.asyncio
async def test_experiment_samples_empty_message_and_not_found_branches(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-samples-branches")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "samples-empty"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    attempt = 3
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=attempt)

    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    run_dir = experiment_dir / "runs" / str(attempt)
    for predictions_path in [run_dir / "predictions.jsonl", experiment_dir / "predictions.jsonl"]:
        if predictions_path.exists():
            predictions_path.unlink()

    empty = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/samples?mode=lowest_confidence_correct&limit=5"
    )
    assert empty.status_code == 200
    empty_payload = empty.json()
    assert empty_payload["items"] == []
    assert empty_payload["message"] == "No matching samples found for this filter."

    for evaluation_path in [run_dir / "evaluation.json", experiment_dir / "evaluation.json"]:
        if evaluation_path.exists():
            evaluation_path.unlink()

    missing = await client.get(
        f"/api/v1/projects/{project_id}/experiments/{experiment_id}/samples?mode=misclassified&limit=5"
    )
    assert_api_error(
        missing,
        status_code=404,
        code="evaluation_not_found",
        message="Evaluation not available for this experiment",
    )


@pytest.mark.asyncio
async def test_experiment_export_preserves_saved_dataset_split_membership(client: AsyncClient) -> None:
    project_id, model_id = await _create_project_model(client, project_name="exp-saved-split-export")

    categories = await client.get(f"/api/v1/projects/{project_id}/categories")
    assert categories.status_code == 200
    category_id = categories.json()["items"][0]["id"]

    for index in range(11):
        uploaded = await client.post(
            f"/api/v1/projects/{project_id}/assets/upload",
            data={"relative_path": f"set/sample_{index}.jpg"},
            files={"file": (f"sample_{index}.jpg", b"fake-image-bytes", "image/jpeg")},
        )
        assert uploaded.status_code == 200
        asset_id = uploaded.json()["id"]
        saved = await client.post(
            f"/api/v1/projects/{project_id}/annotations",
            json={
                "asset_id": asset_id,
                "status": "approved",
                "payload_json": {
                    "version": "2.0",
                    "classification": {"category_ids": [category_id], "primary_category_id": category_id},
                    "image_basis": {"width": 100, "height": 80},
                    "objects": [{"id": f"bbox-{index}", "kind": "bbox", "category_id": category_id, "bbox": [10, 10, 20, 15]}],
                },
            },
        )
        assert saved.status_code == 200

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "split-v1",
            "task": "bbox",
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": False}},
            "split": {
                "seed": 1337,
                "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                "stratify": {"enabled": False, "by": "label_primary", "strict_stratify": False},
            },
        },
    )
    assert created_dataset.status_code == 200
    created_payload = created_dataset.json()
    dataset_version = created_payload["version"]
    dataset_version_id = dataset_version["dataset_version_id"]

    expected_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in dataset_version["splits"]["items"]:
        expected_by_split[row["split"]].append(row["asset_id"])
    assert expected_by_split["val"]
    assert expected_by_split["test"]

    created_experiment = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "split-check", "dataset_version_id": dataset_version_id},
    )
    assert created_experiment.status_code == 200
    experiment_id = created_experiment.json()["id"]

    import sheriff_api.routers.experiments as experiments_router

    async def _enqueue(_job_payload: dict) -> None:
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_router.train_queue, "enqueue_train_job", _enqueue)
    started = await client.post(f"/api/v1/projects/{project_id}/experiments/{experiment_id}/start")
    monkeypatch.undo()
    assert started.status_code == 200

    detail = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}")
    assert detail.status_code == 200
    dataset_export = detail.json()["artifacts_json"]["last_dataset_export"]
    zip_relpath = dataset_export["zip_relpath"]
    zip_path = Path(get_settings().storage_root) / zip_relpath
    assert zip_path.exists()

    with zipfile.ZipFile(BytesIO(zip_path.read_bytes()), "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    for split_name in ["train", "val", "test"]:
        assert sorted(manifest["splits"][split_name]["asset_ids"]) == sorted(expected_by_split[split_name])
