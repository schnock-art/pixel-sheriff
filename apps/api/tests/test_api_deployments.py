from __future__ import annotations

import json
from pathlib import Path

from httpx import AsyncClient
import pytest
import sheriff_api.routers.deployments as deployments_router
from sheriff_api.config import get_settings
from sheriff_api.errors import api_error

from .api_test_helpers import (
    _create_classification_project_model,
    _create_classification_project_model_with_categories,
    _create_detection_project_model_with_categories,
    _seed_experiment_run_artifacts,
    _seed_experiment_variant_artifacts,
    assert_api_error,
)

@pytest.mark.asyncio
async def test_create_deployment_resolves_onnx_and_persists_model_key(client: AsyncClient) -> None:
    project_id, model_id, _task_id = await _create_classification_project_model(client, project_name="deploy-create")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "deploy-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    response = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "deploy-v1",
            "task": "classification",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()["deployment"]
    assert payload["name"] == "deploy-v1"
    assert len(payload["model_key"]) == 64

    listing = await client.get(f"/api/v1/projects/{project_id}/deployments")
    assert listing.status_code == 200
    assert listing.json()["active_deployment_id"] == payload["deployment_id"]


@pytest.mark.asyncio
async def test_create_detection_deployment_maps_experiment_task_to_bbox(client: AsyncClient) -> None:
    project_id, model_id, task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name="deploy-detection",
        category_names=["boat"],
    )
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "detect-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    response = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "detect-deploy",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()["deployment"]
    assert payload["task"] == "bbox"
    assert payload["task_id"] == task_id


@pytest.mark.asyncio
async def test_create_deployment_pins_explicit_variant_key(client: AsyncClient) -> None:
    project_id, model_id, _task_id = await _create_classification_project_model(client, project_name="deploy-variant")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "deploy-variant-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)
    _seed_experiment_variant_artifacts(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=1,
        variant_key="ptq_int8",
        preferred_variant_key="ptq_int8",
    )

    response = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "deploy-int8",
            "task": "classification",
            "device_preference": "auto",
            "source": {
                "experiment_id": experiment_id,
                "attempt": 1,
                "checkpoint_kind": "best_metric",
                "variant_key": "ptq_int8",
            },
            "is_active": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()["deployment"]
    assert payload["source"]["variant_key"] == "ptq_int8"
    assert "/variants/ptq_int8/" in payload["source"]["onnx_relpath"]


@pytest.mark.asyncio
async def test_create_deployment_accepts_fp16_variant_key(client: AsyncClient) -> None:
    project_id, model_id, _task_id = await _create_classification_project_model(client, project_name="deploy-fp16")
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "deploy-fp16-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)
    _seed_experiment_variant_artifacts(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=1,
        variant_key="fp16",
        preferred_variant_key="fp16",
    )

    response = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "deploy-fp16",
            "task": "classification",
            "device_preference": "auto",
            "source": {
                "experiment_id": experiment_id,
                "attempt": 1,
                "checkpoint_kind": "best_metric",
                "variant_key": "fp16",
            },
            "is_active": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()["deployment"]
    assert payload["source"]["variant_key"] == "fp16"
    assert "/variants/fp16/" in payload["source"]["onnx_relpath"]


@pytest.mark.asyncio
async def test_predict_without_active_returns_no_active_deployment(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "predict-no-active"})).json()
    project_id = project["id"]
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    response = await client.post(f"/api/v1/projects/{project_id}/predict", json={"asset_id": asset_id, "top_k": 5})
    assert_api_error(response, status_code=409, code="no_active_deployment", message="No active deployment is configured")


@pytest.mark.asyncio
async def test_predict_maps_inference_predictions_with_class_ids(client: AsyncClient) -> None:
    project_id, model_id, _task_id, class_ids = await _create_classification_project_model_with_categories(
        client,
        project_name="predict-mapped",
        category_names=["rock", "paper"],
    )
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = class_ids
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "predict-deploy",
            "task": "classification",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    async def _infer(_payload: dict) -> dict:
        return {
            "device_selected": "cpu",
            "predictions": [
                {"class_index": 0, "score": 0.9},
                {"class_index": 1, "score": 0.1},
            ],
            "output_dim": 2,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_classification", _infer)
    response = await client.post(f"/api/v1/projects/{project_id}/predict", json={"asset_id": asset_id, "top_k": 5})
    monkeypatch.undo()
    assert response.status_code == 200
    payload = response.json()
    assert payload["device_selected"] == "cpu"
    assert payload["predictions"][0]["class_id"] == class_ids[0]
    assert payload["predictions"][0]["class_name"] == "rock"


@pytest.mark.asyncio
async def test_predict_detection_maps_inference_boxes_with_class_ids(client: AsyncClient) -> None:
    project_id, model_id, _task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name="predict-detection",
        category_names=["boat", "buoy"],
    )
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-detect-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "detect-predict-deploy",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    async def _infer(_payload: dict) -> dict:
        return {
            "device_selected": "cpu",
            "boxes": [
                {"class_index": 0, "score": 0.9, "bbox": [10, 20, 30, 40]},
                {"class_index": 1, "score": 0.4, "bbox": [1, 2, 3, 4]},
            ],
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_detection", _infer)
    response = await client.post(
        f"/api/v1/projects/{project_id}/predict",
        json={"asset_id": asset_id, "score_threshold": 0.55},
    )
    monkeypatch.undo()
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "bbox"
    assert payload["device_selected"] == "cpu"
    assert payload["boxes"][0]["class_id"] == category_ids[0]
    assert payload["boxes"][0]["class_name"] == "boat"
    assert payload["boxes"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


@pytest.mark.asyncio
async def test_predict_detection_maps_inference_boxes_with_class_names_fallback(client: AsyncClient) -> None:
    project_id, model_id, _task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name="predict-detection-fallback",
        category_names=["boat", "buoy"],
    )
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-detect-exp-fallback"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("class_ids", None)
    metadata["class_order"] = ["boat", "buoy"]
    metadata["class_names"] = ["boat", "buoy"]
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "detect-predict-deploy-fallback",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    async def _infer(_payload: dict) -> dict:
        return {
            "device_selected": "cpu",
            "boxes": [{"class_index": 1, "score": 0.4, "bbox": [1, 2, 3, 4]}],
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_detection", _infer)
    response = await client.post(f"/api/v1/projects/{project_id}/predict", json={"asset_id": asset_id})
    monkeypatch.undo()
    assert response.status_code == 200
    payload = response.json()
    assert payload["boxes"][0]["class_id"] == category_ids[1]
    assert payload["boxes"][0]["class_name"] == "buoy"


@pytest.mark.asyncio
async def test_predict_detection_rejects_invalid_class_index(client: AsyncClient) -> None:
    project_id, model_id, _task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name="predict-detection-mismatch",
        category_names=["boat"],
    )
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-detect-exp-mismatch"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "detect-predict-mismatch",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    async def _infer(_payload: dict) -> dict:
        return {
            "device_selected": "cpu",
            "boxes": [{"class_index": 3, "score": 0.9, "bbox": [10, 20, 30, 40]}],
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_detection", _infer)
    response = await client.post(f"/api/v1/projects/{project_id}/predict", json={"asset_id": asset_id})
    monkeypatch.undo()
    assert_api_error(
        response,
        status_code=409,
        code="deployment_output_dim_mismatch",
        message="Inference output does not match deployment class_ids",
    )


@pytest.mark.asyncio
async def test_predict_returns_output_dim_mismatch(client: AsyncClient) -> None:
    project_id, model_id, _task_id, category_ids = await _create_classification_project_model_with_categories(
        client,
        project_name="predict-dim-mismatch",
        category_names=["only"],
    )
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-exp-mismatch"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "predict-deploy-mismatch",
            "task": "classification",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    async def _infer(_payload: dict) -> dict:
        return {
            "device_selected": "cpu",
            "predictions": [{"class_index": 3, "score": 0.9}],
            "output_dim": 4,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_classification", _infer)
    response = await client.post(f"/api/v1/projects/{project_id}/predict", json={"asset_id": asset_id, "top_k": 5})
    monkeypatch.undo()
    assert_api_error(
        response,
        status_code=409,
        code="deployment_output_dim_mismatch",
        message="Inference output does not match deployment class_ids",
    )


@pytest.mark.asyncio
async def test_predict_batch_returns_folder_review_summary(client: AsyncClient) -> None:
    project_id, model_id, _task_id, class_ids = await _create_classification_project_model_with_categories(
        client,
        project_name="predict-batch-summary",
        category_names=["rock", "paper"],
    )
    upload_first = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": "batch/sample-1.jpg"},
        files={"file": ("sample-1.jpg", b"fake-image-bytes-1", "image/jpeg")},
    )
    upload_second = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": "batch/sample-2.jpg"},
        files={"file": ("sample-2.jpg", b"fake-image-bytes-2", "image/jpeg")},
    )
    assert upload_first.status_code == 200
    assert upload_second.status_code == 200
    first_asset_id = upload_first.json()["id"]
    second_asset_id = upload_second.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-batch-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = class_ids
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "predict-batch-deploy",
            "task": "classification",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    infer_calls = {"count": 0}

    async def _infer(_payload: dict) -> dict:
        infer_calls["count"] += 1
        if infer_calls["count"] == 1:
            return {
                "device_selected": "cpu",
                "predictions": [{"class_index": 0, "score": 0.9}],
                "output_dim": 2,
            }
        return {
            "device_selected": "cpu",
            "predictions": [],
            "output_dim": 2,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_classification", _infer)
    response = await client.post(
        f"/api/v1/projects/{project_id}/predict/batch",
        json={"asset_ids": [first_asset_id, second_asset_id, "missing-asset-id"], "top_k": 5},
    )
    monkeypatch.undo()
    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_count"] == 3
    assert payload["completed_count"] == 2
    assert payload["pending_review_count"] == 1
    assert payload["empty_count"] == 1
    assert payload["error_count"] == 1
    assert payload["predictions"][0]["asset_id"] == first_asset_id
    assert payload["predictions"][0]["predictions"][0]["class_id"] == class_ids[0]
    assert payload["errors"][0]["asset_id"] == "missing-asset-id"
    assert payload["errors"][0]["code"] == "asset_not_found"


@pytest.mark.asyncio
async def test_predict_batch_detection_captures_per_asset_inference_errors(client: AsyncClient) -> None:
    project_id, model_id, _task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name="predict-batch-detection-errors",
        category_names=["boat"],
    )
    upload_first = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": "batch/detect-1.jpg"},
        files={"file": ("detect-1.jpg", b"fake-image-1", "image/jpeg")},
    )
    upload_second = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": "batch/detect-2.jpg"},
        files={"file": ("detect-2.jpg", b"fake-image-2", "image/jpeg")},
    )
    assert upload_first.status_code == 200
    assert upload_second.status_code == 200
    first_asset_id = upload_first.json()["id"]
    second_asset_id = upload_second.json()["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "predict-batch-detect-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "predict-batch-detect-deploy",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200

    infer_calls = {"count": 0}

    async def _infer(_payload: dict) -> dict:
        infer_calls["count"] += 1
        if infer_calls["count"] == 1:
            return {
                "device_selected": "cpu",
                "boxes": [{"class_index": 0, "score": 0.8, "bbox": [10, 20, 30, 40]}],
            }
        raise api_error(status_code=503, code="inference_unavailable", message="Inference service unavailable")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "infer_detection", _infer)
    response = await client.post(
        f"/api/v1/projects/{project_id}/predict/batch",
        json={"asset_ids": [first_asset_id, second_asset_id], "score_threshold": 0.3},
    )
    monkeypatch.undo()
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "bbox"
    assert payload["completed_count"] == 1
    assert payload["pending_review_count"] == 1
    assert payload["error_count"] == 1
    assert payload["predictions"][0]["boxes"][0]["class_id"] == category_ids[0]
    assert payload["errors"][0]["asset_id"] == second_asset_id
    assert payload["errors"][0]["code"] == "inference_unavailable"


@pytest.mark.asyncio
async def test_warmup_deployment_supports_detection(client: AsyncClient) -> None:
    project_id, model_id, _task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name="warmup-detection",
        category_names=["boat"],
    )
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "warmup-detect-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    settings = get_settings()
    metadata_path = Path(settings.storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "warmup-detect-deploy",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200
    deployment_id = deployed.json()["deployment"]["deployment_id"]

    async def _warmup(_payload: dict) -> dict:
        return {"device_selected": "cuda", "warmed": True}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deployments_router.inference_client, "warmup_detection", _warmup)
    response = await client.post(f"/api/v1/projects/{project_id}/deployments/{deployment_id}/warmup")
    monkeypatch.undo()
    assert response.status_code == 200
    assert response.json()["device_selected"] == "cuda"
