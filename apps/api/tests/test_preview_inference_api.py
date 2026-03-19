from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

from httpx import AsyncClient
import pytest

from sheriff_api.config import get_settings

import sheriff_api.services.preview_inference as preview_service

from .api_test_helpers import (
    _create_classification_project_model_with_categories,
    _create_default_task_project,
    _create_detection_project_model_with_categories,
    _create_task_scoped_category,
    _seed_experiment_run_artifacts,
    assert_api_error,
)


def _png_bytes(width: int = 64, height: int = 48, *, rgba: tuple[int, int, int, int] = (64, 128, 192, 255)) -> bytes:
    row = bytes([0] + list(rgba) * width)
    raw = row * height

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(raw, level=9))
    iend = chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def _preview_dir(project_id: str) -> Path:
    return Path(get_settings().storage_root) / "imports" / project_id / "preview"


def _assert_preview_dir_empty(project_id: str) -> None:
    preview_dir = _preview_dir(project_id)
    if not preview_dir.exists():
        return
    assert list(preview_dir.iterdir()) == []


async def _create_active_detection_deployment(
    client: AsyncClient,
    *,
    project_name: str,
    category_names: list[str],
) -> tuple[str, str, str, list[str]]:
    project_id, model_id, task_id, category_ids = await _create_detection_project_model_with_categories(
        client,
        project_name=project_name,
        category_names=category_names,
    )
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "preview-detect-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    metadata_path = Path(get_settings().storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata["task"] = "detection"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "preview-detect-deploy",
            "task": "bbox",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200
    return project_id, task_id, deployed.json()["deployment"]["deployment_id"], category_ids


async def _create_active_classification_deployment(
    client: AsyncClient,
    *,
    project_name: str,
    category_names: list[str],
) -> tuple[str, str, str, list[str]]:
    project_id, model_id, task_id, category_ids = await _create_classification_project_model_with_categories(
        client,
        project_name=project_name,
        category_names=category_names,
    )
    created = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_id, "name": "preview-class-exp"},
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]
    _seed_experiment_run_artifacts(project_id=project_id, experiment_id=experiment_id, attempt=1, include_onnx=True)

    metadata_path = Path(get_settings().storage_root) / "experiments" / project_id / experiment_id / "runs" / "1" / "onnx" / "onnx.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_ids"] = category_ids
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    deployed = await client.post(
        f"/api/v1/projects/{project_id}/deployments",
        json={
            "name": "preview-class-deploy",
            "task": "classification",
            "device_preference": "auto",
            "source": {"experiment_id": experiment_id, "attempt": 1, "checkpoint_kind": "best_metric"},
            "is_active": True,
        },
    )
    assert deployed.status_code == 200
    return project_id, task_id, deployed.json()["deployment"]["deployment_id"], category_ids


@pytest.mark.asyncio
async def test_preview_inference_bbox_with_florence_returns_normalized_boxes_and_cleans_temp_file(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = await _create_default_task_project(client, name="preview-bbox-florence", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="person")

    async def fake_florence(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "device_selected": "cuda",
            "boxes": [
                {"label_text": "person", "score": 0.91, "bbox": [10, 12, 30, 32]},
                {"label_text": "unknown", "score": 0.41, "bbox": [1, 2, 9, 12]},
            ],
        }

    monkeypatch.setattr(preview_service.inference_client, "florence_detect", fake_florence)

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/preview-inference",
        data={
            "task_kind": "bbox",
            "prelabel_config": json.dumps(
                {
                    "source_type": "florence2",
                    "prompts": ["person"],
                    "frame_sampling": {"mode": "every_n_frames", "value": 1},
                    "confidence_threshold": 0.25,
                    "max_detections_per_frame": 5,
                }
            ),
        },
        files={"file": ("preview.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "bbox"
    assert payload["source_label"] == "Florence-2"
    assert payload["device_selected"] == "cuda"
    assert payload["preview_width"] == 64
    assert payload["preview_height"] == 48
    assert payload["boxes"][0] == {
        "class_id": payload["boxes"][0]["class_id"],
        "class_name": "person",
        "score": 0.91,
        "bbox": [10.0, 12.0, 20.0, 20.0],
        "matched": True,
    }
    assert payload["boxes"][1]["matched"] is False
    assert payload["debug"][0]["label_text"] == "unknown"
    _assert_preview_dir_empty(project_id)


@pytest.mark.asyncio
async def test_preview_inference_bbox_with_active_deployment_returns_overlay_boxes(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, task_id, _deployment_id, category_ids = await _create_active_detection_deployment(
        client,
        project_name="preview-bbox-active",
        category_names=["boat", "buoy"],
    )

    async def fake_detection(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "device_selected": "cpu",
            "boxes": [
                {"class_index": 0, "class_name": "boat", "score": 0.88, "bbox": [8, 10, 24, 18]},
            ],
        }

    monkeypatch.setattr(preview_service.inference_client, "infer_detection", fake_detection)

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/preview-inference",
        data={
            "task_kind": "bbox",
            "prelabel_config": json.dumps(
                {
                    "source_type": "active_deployment",
                    "deployment_id": _deployment_id,
                    "prompts": [],
                    "frame_sampling": {"mode": "every_n_frames", "value": 1},
                    "confidence_threshold": 0.3,
                    "max_detections_per_frame": 5,
                }
            ),
        },
        files={"file": ("preview.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_label"] == "preview-detect-deploy"
    assert payload["boxes"][0]["class_id"] == category_ids[0]
    assert payload["boxes"][0]["class_name"] == "boat"
    assert payload["boxes"][0]["matched"] is True
    assert payload["boxes"][0]["bbox"] == [8.0, 10.0, 24.0, 18.0]
    _assert_preview_dir_empty(project_id)


@pytest.mark.asyncio
async def test_preview_inference_classification_returns_top_predictions(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, task_id, _deployment_id, category_ids = await _create_active_classification_deployment(
        client,
        project_name="preview-classification",
        category_names=["rock", "paper"],
    )

    async def fake_classification(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "device_selected": "cpu",
            "predictions": [
                {"class_index": 1, "score": 0.8},
                {"class_index": 0, "score": 0.2},
            ],
            "output_dim": 2,
        }

    monkeypatch.setattr(preview_service.inference_client, "infer_classification", fake_classification)

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/preview-inference",
        data={"task_kind": "classification", "top_k": "2"},
        files={"file": ("preview.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "classification"
    assert payload["source_label"] == "preview-class-deploy"
    assert payload["predictions"][0]["class_id"] == category_ids[1]
    assert payload["predictions"][0]["class_name"] == "paper"
    assert payload["predictions"][1]["class_id"] == category_ids[0]
    _assert_preview_dir_empty(project_id)


@pytest.mark.asyncio
async def test_preview_inference_classification_rejects_incompatible_requested_deployment_and_cleans_temp_file(
    client: AsyncClient,
) -> None:
    project_id, _task_id, deployment_id, _category_ids = await _create_active_classification_deployment(
        client,
        project_name="preview-classification-incompatible",
        category_names=["cat", "dog"],
    )
    created_task = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json={"name": "Secondary", "kind": "classification"},
    )
    assert created_task.status_code == 200
    secondary_task_id = created_task.json()["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{secondary_task_id}/preview-inference",
        data={"task_kind": "classification", "deployment_id": deployment_id},
        files={"file": ("preview.png", _png_bytes(), "image/png")},
    )

    assert_api_error(response, status_code=409, code="deployment_incompatible", message="Deployment is incompatible with this task")
    _assert_preview_dir_empty(project_id)


@pytest.mark.asyncio
async def test_preview_inference_classification_without_active_deployment_returns_clear_error_and_cleans_temp_file(
    client: AsyncClient,
) -> None:
    project = await _create_default_task_project(client, name="preview-no-deployment", task_type="classification_single")
    project_id = project["id"]
    task_id = project["default_task_id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/preview-inference",
        data={"task_kind": "classification"},
        files={"file": ("preview.png", _png_bytes(), "image/png")},
    )

    assert_api_error(response, status_code=409, code="no_active_deployment", message="No active deployment is configured")
    _assert_preview_dir_empty(project_id)
