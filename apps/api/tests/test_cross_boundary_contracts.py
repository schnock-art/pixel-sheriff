import json
from pathlib import Path

from httpx import AsyncClient
from jsonschema import Draft7Validator, Draft202012Validator
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DATASET_VERSION_SCHEMA = json.loads(
    (REPO_ROOT / "apps" / "web" / "src" / "lib" / "schemas" / "dataset_version_v2.schema.json").read_text(encoding="utf-8")
)
WEB_MODEL_CONFIG_SCHEMA = json.loads(
    (REPO_ROOT / "apps" / "web" / "src" / "schemas" / "model-config-1.0.schema.json").read_text(encoding="utf-8")
)


def _assert_valid_against_web_schema(*, validator, payload: dict, label: str) -> None:
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    assert not errors, f"{label} failed web schema validation: {[error.message for error in errors]}"


async def _create_task(
    client: AsyncClient, *, project_id: str, name: str, kind: str, label_mode: str | None = None
) -> dict:
    payload = {"name": name, "kind": kind}
    if label_mode is not None:
        payload["label_mode"] = label_mode
    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json=payload,
    )
    assert response.status_code == 200
    return response.json()


async def _create_category(client: AsyncClient, *, project_id: str, task_id: str, name: str) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": name},
    )
    assert response.status_code == 200
    return response.json()


async def _upload_asset(client: AsyncClient, *, project_id: str, filename: str, relative_path: str) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": relative_path},
        files={"file": (filename, b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    return response.json()


async def _save_classification_annotation(
    client: AsyncClient, *, project_id: str, task_id: str, asset_id: str, category_id: str
) -> None:
    response = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset_id,
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "category_ids": [category_id],
                "classification": {"category_ids": [category_id], "primary_category_id": category_id},
                "image_basis": {"width": 100, "height": 80},
            },
        },
    )
    assert response.status_code == 200


async def _save_bbox_annotation(
    client: AsyncClient, *, project_id: str, task_id: str, asset_id: str, category_id: str, bbox_id: str
) -> None:
    response = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset_id,
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "image_basis": {"width": 100, "height": 80},
                "objects": [{"id": bbox_id, "kind": "bbox", "category_id": category_id, "bbox": [10, 10, 20, 15]}],
            },
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_dataset_version_response_validates_against_web_schema(client: AsyncClient) -> None:
    validator = Draft202012Validator(WEB_DATASET_VERSION_SCHEMA)

    project = (await client.post("/api/v1/projects", json={"name": "cross-boundary-dataset", "task_type": "bbox"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="boat")
    asset = await _upload_asset(client, project_id=project_id, filename="sample.jpg", relative_path="boats/sample.jpg")
    await _save_bbox_annotation(client, project_id=project_id, task_id=task_id, asset_id=asset["id"], category_id=category["id"], bbox_id="bbox-1")

    created = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created.status_code == 200
    created_payload = created.json()
    _assert_valid_against_web_schema(validator=validator, payload=created_payload["version"], label="created dataset version")

    dataset_version_id = created_payload["version"]["dataset_version_id"]
    detail = await client.get(f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}")
    assert detail.status_code == 200
    _assert_valid_against_web_schema(validator=validator, payload=detail.json()["version"], label="dataset version detail")


@pytest.mark.asyncio
async def test_model_config_response_validates_against_web_schema(client: AsyncClient) -> None:
    validator = Draft7Validator(WEB_MODEL_CONFIG_SCHEMA)

    project = (await client.post("/api/v1/projects", json={"name": "cross-boundary-model", "task_type": "bbox"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="boat")
    asset = await _upload_asset(client, project_id=project_id, filename="sample.jpg", relative_path="boats/sample.jpg")
    await _save_bbox_annotation(client, project_id=project_id, task_id=task_id, asset_id=asset["id"], category_id=category["id"], bbox_id="bbox-1")

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {}},
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
    created_model_payload = created_model.json()
    _assert_valid_against_web_schema(validator=validator, payload=created_model_payload["config"], label="created model config")

    detail = await client.get(f"/api/v1/projects/{project_id}/models/{created_model_payload['id']}")
    assert detail.status_code == 200
    _assert_valid_against_web_schema(validator=validator, payload=detail.json()["config_json"], label="model detail config")


@pytest.mark.asyncio
async def test_dataset_preview_create_and_model_draft_flow_stays_consistent(client: AsyncClient) -> None:
    dataset_validator = Draft202012Validator(WEB_DATASET_VERSION_SCHEMA)
    model_validator = Draft7Validator(WEB_MODEL_CONFIG_SCHEMA)

    project = (await client.post("/api/v1/projects", json={"name": "preview-create-model", "task_type": "bbox"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="flower")

    uploaded_assets = []
    for index in range(4):
        asset = await _upload_asset(
            client,
            project_id=project_id,
            filename=f"sample_{index}.jpg",
            relative_path=f"flowers/sample_{index}.jpg",
        )
        uploaded_assets.append(asset)
        await _save_bbox_annotation(
            client,
            project_id=project_id,
            task_id=task_id,
            asset_id=asset["id"],
            category_id=category["id"],
            bbox_id=f"bbox-{index}",
        )

    preview_payload = {
        "task_id": task_id,
        "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": False}},
        "split": {
            "seed": 1337,
            "ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "stratify": {"enabled": False, "by": "label_primary", "strict_stratify": False},
        },
    }
    preview = await client.post(f"/api/v1/projects/{project_id}/datasets/versions/preview", json=preview_payload)
    assert preview.status_code == 200
    preview_json = preview.json()
    assert preview_json["counts"]["total"] == len(uploaded_assets)
    assert preview_json["class_names"][category["id"]] == "flower"

    created = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={**preview_payload, "name": "preview-backed-version", "set_active": True},
    )
    assert created.status_code == 200
    created_json = created.json()
    version = created_json["version"]
    _assert_valid_against_web_schema(validator=dataset_validator, payload=version, label="preview-backed dataset version")

    assert version["stats"]["asset_count"] == preview_json["counts"]["total"]
    assert version["stats"]["class_counts"] == preview_json["counts"]["class_counts"]
    assert version["stats"]["split_counts"] == preview_json["counts"]["split_counts"]

    model = await client.post(
        f"/api/v1/projects/{project_id}/models",
        json={"dataset_version_id": version["dataset_version_id"]},
    )
    assert model.status_code == 200
    model_json = model.json()
    _assert_valid_against_web_schema(validator=model_validator, payload=model_json["config"], label="model config from preview-backed version")
    assert model_json["config"]["source_dataset"]["manifest_id"] == version["dataset_version_id"]
    assert model_json["config"]["source_dataset"]["class_order"] == version["labels"]["label_schema"]["class_order"]


@pytest.mark.asyncio
async def test_task_scoped_labeling_active_dataset_and_experiment_creation_stay_aligned(client: AsyncClient) -> None:
    model_validator = Draft7Validator(WEB_MODEL_CONFIG_SCHEMA)

    project = (await client.post("/api/v1/projects", json={"name": "task-aware-flow", "task_type": "classification_single"})).json()
    project_id = project["id"]
    default_task_id = project["default_task_id"]
    secondary_task = await _create_task(
        client,
        project_id=project_id,
        name="secondary-bbox",
        kind="bbox",
    )
    secondary_task_id = secondary_task["id"]

    default_category = await _create_category(client, project_id=project_id, task_id=default_task_id, name="default-class")
    secondary_category = await _create_category(client, project_id=project_id, task_id=secondary_task_id, name="secondary-box")
    asset = await _upload_asset(client, project_id=project_id, filename="sample.jpg", relative_path="mixed/sample.jpg")

    await _save_classification_annotation(
        client,
        project_id=project_id,
        task_id=default_task_id,
        asset_id=asset["id"],
        category_id=default_category["id"],
    )
    await _save_bbox_annotation(
        client,
        project_id=project_id,
        task_id=secondary_task_id,
        asset_id=asset["id"],
        category_id=secondary_category["id"],
        bbox_id="bbox-secondary",
    )

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "secondary-task-version",
            "task_id": secondary_task_id,
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": True}},
            "split": {
                "seed": 7,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created_dataset.status_code == 200
    dataset_version_id = created_dataset.json()["version"]["dataset_version_id"]

    model = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert model.status_code == 200
    model_json = model.json()
    _assert_valid_against_web_schema(validator=model_validator, payload=model_json["config"], label="task-aware active model config")
    assert model_json["config"]["source_dataset"]["manifest_id"] == dataset_version_id
    assert model_json["config"]["source_dataset"]["task_id"] == secondary_task_id
    assert model_json["config"]["source_dataset"]["task"] == "detection"
    assert secondary_category["id"] in model_json["config"]["source_dataset"]["class_order"]
    assert default_category["id"] not in model_json["config"]["source_dataset"]["class_order"]

    created_experiment = await client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={"model_id": model_json["id"], "name": "task-aware-experiment"},
    )
    assert created_experiment.status_code == 200
    experiment_json = created_experiment.json()
    assert experiment_json["task_id"] == secondary_task_id
    assert experiment_json["config_json"]["dataset_version_id"] == dataset_version_id
    assert experiment_json["config_json"]["task_id"] == secondary_task_id
    assert experiment_json["config_json"]["task"] == "detection"
