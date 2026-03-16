from __future__ import annotations

import json
from pathlib import Path

from httpx import AsyncClient
import pytest

from sheriff_api.config import get_settings
from sheriff_api.demo_experiment_seed import seed_demo_experiment

from .api_test_helpers import _create_detection_project_with_dataset_version


@pytest.mark.asyncio
async def test_seed_demo_experiment_creates_completed_detection_run_and_deployable_onnx(client: AsyncClient) -> None:
    project_id, task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client,
        project_name="demo-experiment-seed",
    )
    created_model = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": dataset_version_id, "name": "Demo Detector"},
    )
    assert created_model.status_code == 200
    model_id = created_model.json()["id"]

    metadata = await seed_demo_experiment(project_id, task_id, model_id, dataset_version_id)
    experiment_id = metadata["experimentId"]

    detail = await client.get(f"/api/v1/projects/{project_id}/experiments/{experiment_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["status"] == "completed"
    assert detail_payload["current_run_attempt"] == 1
    assert detail_payload["last_completed_attempt"] == 1
    assert detail_payload["summary_json"]["best_metric_name"] == "val_map_50_95"
    assert len(detail_payload["metrics"]) == 6

    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    run_dir = experiment_dir / "runs" / "1"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "training.log").exists()
    assert (run_dir / "runtime.json").exists()
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "onnx" / "model.onnx").exists()
    assert (run_dir / "onnx" / "onnx.metadata.json").exists()

    runtime_payload = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    assert runtime_payload["device_selected"] == "cpu"
    assert runtime_payload["cache_resized_images"] is True

    onnx_metadata = json.loads((run_dir / "onnx" / "onnx.metadata.json").read_text(encoding="utf-8"))
    assert onnx_metadata["status"] == "exported"
    assert onnx_metadata["task"] == "detection"
    assert onnx_metadata["class_ids"]

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "demo-seeded-deploy",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200
    deployment_payload = deployed.json()["deployment"]
    assert deployment_payload["task"] == "bbox"
    assert deployment_payload["source"]["attempt"] == 1
