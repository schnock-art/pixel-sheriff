from __future__ import annotations

import json
import uuid

from httpx import AsyncClient
import pytest

from .api_test_helpers import (
    _create_detection_project_with_dataset_version,
    _create_detection_project_with_manifest,
    assert_api_error,
)

@pytest.mark.asyncio
async def test_project_model_create_builds_schema_valid_config_from_manifest(client: AsyncClient) -> None:
    project_id, manifest = await _create_detection_project_with_manifest(client, project_name="model-create")

    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    payload = created.json()
    config = payload["config"]

    assert payload["id"]
    assert config["schema_version"] == "1.0"
    assert config["source_dataset"]["task"] == "detection"
    assert config["source_dataset"]["num_classes"] == len(manifest["label_schema"]["class_order"])
    assert config["source_dataset"]["class_order"] == manifest["label_schema"]["class_order"]
    assert config["architecture"]["family"] == "retinanet"
    assert config["architecture"]["backbone"]["name"] == "resnet50"
    assert config["architecture"]["head"]["num_classes"] == len(manifest["label_schema"]["class_order"])
    assert config["outputs"]["primary"]["format"] == "coco_detections"
    assert config["export"]["onnx"]["enabled"] is True
    assert config["export"]["onnx"]["opset"] == 17
    assert config["export"]["onnx"]["dynamic_shapes"] == {"enabled": True, "batch": True, "height_width": False}

    detail = await client.get(f"/api/v1/projects/{project_id}/models/{payload['id']}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["project_id"] == project_id
    assert detail_payload["config_json"]["source_dataset"]["num_classes"] == len(manifest["label_schema"]["class_order"])


@pytest.mark.asyncio
async def test_project_model_create_allows_multi_label_classification_loss(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "multi-label-model", "task_type": "classification"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    category = (
        await client.post(
            f"/api/v1/projects/{project_id}/categories",
            json={"task_id": task_id, "name": "flower", "display_order": 1},
        )
    ).json()
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "category_ids": [category["id"]],
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 80},
            },
        },
    )
    assert annotation.status_code == 200

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "multi-label-v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": True}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created_dataset.status_code == 200

    created_model = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created_model.status_code == 200
    config = created_model.json()["config"]
    assert config["source_dataset"]["label_mode"] == "multi_label"
    assert config["loss"]["type"] == "classification_bce_with_logits"


@pytest.mark.asyncio
async def test_project_model_list_returns_summaries(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-list")

    first = await client.post(f"/api/v1/projects/{project_id}/models", json={"name": "retina-a"})
    second = await client.post(f"/api/v1/projects/{project_id}/models", json={"name": "retina-b"})
    assert first.status_code == 200
    assert second.status_code == 200

    listed = await client.get(f"/api/v1/projects/{project_id}/models")
    assert listed.status_code == 200
    rows = listed.json()

    assert len(rows) == 2
    names = {row["name"] for row in rows}
    assert names == {"retina-a", "retina-b"}
    for row in rows:
        assert row["task"] == "detection"
        assert row["backbone_name"] == "resnet50"
        assert row["num_classes"] == 1


@pytest.mark.asyncio
async def test_project_model_update_persists_valid_config(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-update-valid")

    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    created_payload = created.json()
    model_id = created_payload["id"]
    updated_config = created_payload["config"]
    updated_config["input"]["input_size"] = [512, 512]
    updated_config["architecture"]["backbone"]["name"] = "resnet34"
    updated_config["export"]["onnx"]["opset"] = 18

    update_response = await client.put(
        f"/api/v1/projects/{project_id}/models/{model_id}",
        json={"config_json": updated_config},
    )
    assert update_response.status_code == 200
    update_payload = update_response.json()
    assert update_payload["id"] == model_id
    assert update_payload["config_json"]["input"]["input_size"] == [512, 512]
    assert update_payload["config_json"]["architecture"]["backbone"]["name"] == "resnet34"
    assert update_payload["config_json"]["export"]["onnx"]["opset"] == 18

    detail = await client.get(f"/api/v1/projects/{project_id}/models/{model_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["config_json"]["input"]["input_size"] == [512, 512]
    assert detail_payload["config_json"]["architecture"]["backbone"]["name"] == "resnet34"
    assert detail_payload["config_json"]["export"]["onnx"]["opset"] == 18


@pytest.mark.asyncio
async def test_project_model_update_accepts_ssdlite_detection_config(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-update-ssdlite")

    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    created_payload = created.json()
    model_id = created_payload["id"]
    updated_config = created_payload["config"]
    updated_config["input"]["input_size"] = [320, 320]
    updated_config["architecture"] = {
        "family": "ssdlite320_mobilenet_v3_large",
        "framework": "torchvision",
        "precision": "fp32",
        "backbone": {"name": "mobilenet_v3_large", "pretrained": True},
        "neck": {"type": "none"},
        "head": {"type": "ssdlite", "num_classes": updated_config["source_dataset"]["num_classes"]},
    }
    updated_config["loss"] = {"type": "ssdlite_default"}
    updated_config["outputs"]["primary"] = {
        "name": "coco_detections",
        "type": "task_output",
        "task": "detection",
        "format": "coco_detections",
    }
    updated_config["export"]["onnx"]["output_names"] = ["coco_detections"]

    update_response = await client.put(
        f"/api/v1/projects/{project_id}/models/{model_id}",
        json={"config_json": updated_config},
    )
    assert update_response.status_code == 200
    update_payload = update_response.json()
    assert update_payload["config_json"]["input"]["input_size"] == [320, 320]
    assert update_payload["config_json"]["architecture"]["family"] == "ssdlite320_mobilenet_v3_large"
    assert update_payload["config_json"]["architecture"]["backbone"]["name"] == "mobilenet_v3_large"
    assert update_payload["config_json"]["loss"]["type"] == "ssdlite_default"


@pytest.mark.asyncio
async def test_project_model_update_rejects_ssdlite_detection_config_with_invalid_input_size(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-update-ssdlite-invalid-size")

    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    created_payload = created.json()
    model_id = created_payload["id"]
    updated_config = created_payload["config"]
    updated_config["input"]["input_size"] = [224, 224]
    updated_config["architecture"] = {
        "family": "ssdlite320_mobilenet_v3_large",
        "framework": "torchvision",
        "precision": "fp32",
        "backbone": {"name": "mobilenet_v3_large", "pretrained": True},
        "neck": {"type": "none"},
        "head": {"type": "ssdlite", "num_classes": updated_config["source_dataset"]["num_classes"]},
    }
    updated_config["loss"] = {"type": "ssdlite_default"}
    updated_config["outputs"]["primary"] = {
        "name": "coco_detections",
        "type": "task_output",
        "task": "detection",
        "format": "coco_detections",
    }
    updated_config["export"]["onnx"]["output_names"] = ["coco_detections"]

    update_response = await client.put(
        f"/api/v1/projects/{project_id}/models/{model_id}",
        json={"config_json": updated_config},
    )
    payload = assert_api_error(
        update_response,
        status_code=422,
        code="validation_error",
        message="Model config validation failed",
    )
    issues = payload["error"]["details"]["issues"]
    assert isinstance(issues, list)
    assert any(
        issue.get("path") == "input.input_size" and "requires input_size [320, 320]" in str(issue.get("message"))
        for issue in issues
    )


@pytest.mark.asyncio
async def test_project_model_update_returns_validation_error_for_invalid_config(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-update-invalid")

    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    created_payload = created.json()
    model_id = created_payload["id"]
    invalid_config = created_payload["config"]
    invalid_config["schema_version"] = "2.0"

    update_response = await client.put(
        f"/api/v1/projects/{project_id}/models/{model_id}",
        json={"config_json": invalid_config},
    )
    payload = assert_api_error(
        update_response,
        status_code=422,
        code="validation_error",
        message="Model config validation failed",
    )
    issues = payload["error"]["details"]["issues"]
    assert isinstance(issues, list)
    assert len(issues) >= 1
    first_issue = issues[0]
    assert isinstance(first_issue["path"], str)
    assert isinstance(first_issue["message"], str)


@pytest.mark.asyncio
async def test_project_model_update_returns_not_found_for_missing_model(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-update-missing")
    missing_model_id = str(uuid.uuid4())

    update_response = await client.put(
        f"/api/v1/projects/{project_id}/models/{missing_model_id}",
        json={"config_json": {}},
    )
    assert_api_error(
        update_response,
        status_code=404,
        code="model_not_found",
        message="Model not found in project",
    )


@pytest.mark.asyncio
async def test_project_model_export_generates_deterministic_artifact(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-export")
    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    model_id = created.json()["id"]

    export_a = await client.post(f"/api/v1/projects/{project_id}/models/{model_id}/exports")
    assert export_a.status_code == 200
    payload_a = export_a.json()
    assert payload_a["project_id"] == project_id
    assert payload_a["model_id"] == model_id
    assert payload_a["format"] == "onnx"
    assert len(payload_a["hash"]) == 64
    assert payload_a["export_uri"].endswith("/download")

    downloaded_a = await client.get(payload_a["export_uri"])
    assert downloaded_a.status_code == 200
    assert downloaded_a.headers["content-type"].startswith("application/json")
    artifact_a = json.loads(downloaded_a.content.decode("utf-8"))
    assert artifact_a["project_id"] == project_id
    assert artifact_a["model_id"] == model_id
    assert artifact_a["format"] == "onnx"
    assert artifact_a["source_config_hash"]

    export_b = await client.post(f"/api/v1/projects/{project_id}/models/{model_id}/exports")
    assert export_b.status_code == 200
    payload_b = export_b.json()
    assert payload_b["hash"] == payload_a["hash"]

    downloaded_b = await client.get(payload_b["export_uri"])
    assert downloaded_b.status_code == 200
    assert downloaded_b.content == downloaded_a.content


@pytest.mark.asyncio
async def test_project_model_export_returns_validation_error_when_export_disabled(client: AsyncClient) -> None:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name="model-export-disabled")
    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    model_id = created.json()["id"]
    disabled_config = created.json()["config"]
    disabled_config["export"]["onnx"]["enabled"] = False

    updated = await client.put(
        f"/api/v1/projects/{project_id}/models/{model_id}",
        json={"config_json": disabled_config},
    )
    assert updated.status_code == 200

    export = await client.post(f"/api/v1/projects/{project_id}/models/{model_id}/exports")
    assert_api_error(
        export,
        status_code=422,
        code="model_export_disabled",
        message="Model ONNX export is disabled",
    )


@pytest.mark.asyncio
async def test_model_create_with_explicit_dataset_version_id(client: AsyncClient) -> None:
    project_id, _task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="model-dvid-explicit"
    )

    resp = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": dataset_version_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"]
    # The config's source_dataset.manifest_id must reference the dataset version we passed in.
    assert payload["config"]["source_dataset"]["manifest_id"] == dataset_version_id
    assert payload["config"]["source_dataset"]["task"] == "detection"


@pytest.mark.asyncio
async def test_model_create_without_dataset_version_id_uses_active_version(client: AsyncClient) -> None:
    project_id, _task_id, dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="model-dvid-backwards-compat"
    )

    # No dataset_version_id provided — should fall back to the active/latest version.
    resp = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"]
    assert payload["config"]["source_dataset"]["manifest_id"] == dataset_version_id


@pytest.mark.asyncio
async def test_model_create_with_unknown_dataset_version_id_returns_404(client: AsyncClient) -> None:
    project_id, _task_id, _dataset_version_id = await _create_detection_project_with_dataset_version(
        client, project_name="model-dvid-not-found"
    )

    resp = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "dataset_version_not_found"
