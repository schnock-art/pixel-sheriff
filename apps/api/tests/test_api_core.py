from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import uuid
import zipfile

from httpx import AsyncClient
import pytest
import sheriff_api.routers.assets as assets_router
import sheriff_api.routers.datasets as datasets_router
import sheriff_api.routers.exports as exports_router

from .api_test_helpers import (
    _create_dataset_export,
    _create_default_task_project,
    _create_task_scoped_category,
    assert_api_error,
)

@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_crud_and_export_flow(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="demo")
    project_id = project["id"]
    task_id = project["default_task_id"]

    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="cat", display_order=1)
    assert isinstance(category["id"], str) and category["id"]
    assert uuid.UUID(category["id"])

    patched = (await client.patch(f"/api/v1/categories/{category['id']}", json={"name": "kitty"})).json()
    assert patched["id"] == category["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {"category_ids": [category["id"]]},
        },
    )

    filtered_assets = await client.get(f"/api/v1/projects/{project_id}/assets", params={"status": "approved"})
    assert len(filtered_assets.json()) == 1

    dataset_version_id, export = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="crud-flow",
        selection_filters={"include_statuses": ["approved"]},
    )
    assert export["hash"]
    assert export["export_uri"].startswith(f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export/download")

    archive = await client.get(export["export_uri"])
    assert archive.status_code == 200
    assert archive.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        names = set(bundle.namelist())
        assert "manifest.json" in names
        assert "coco_instances.json" in names
        assert any(name.startswith("assets/") for name in names)

        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        assert len(manifest["assets"]) == 1

        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))
        assert coco["categories"][0]["name"] == "kitty"
        assert coco["annotations"] == []


@pytest.mark.asyncio
async def test_asset_upload_and_content(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "upload-demo"})).json()
    project_id = project["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    uploaded_asset = upload.json()
    assert uploaded_asset["uri"].startswith("/api/v1/assets/")

    content = await client.get(uploaded_asset["uri"])
    assert content.status_code == 200
    assert content.content == b"fake-image-bytes"


@pytest.mark.asyncio
async def test_upload_rejects_unknown_project(client: AsyncClient) -> None:
    missing_project_id = str(uuid.uuid4())
    upload = await client.post(
        f"/api/v1/projects/{missing_project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert_api_error(upload, status_code=404, code="project_not_found", message="Project not found")


@pytest.mark.asyncio
async def test_get_project_returns_project_not_found_error_code(client: AsyncClient) -> None:
    missing_project_id = str(uuid.uuid4())
    response = await client.get(f"/api/v1/projects/{missing_project_id}")
    assert_api_error(response, status_code=404, code="project_not_found", message="Project not found")


@pytest.mark.asyncio
async def test_delete_project_returns_project_not_found_error_code(client: AsyncClient) -> None:
    missing_project_id = str(uuid.uuid4())
    response = await client.delete(f"/api/v1/projects/{missing_project_id}")
    assert_api_error(response, status_code=404, code="project_not_found", message="Project not found")


@pytest.mark.asyncio
async def test_patch_category_returns_category_not_found_error_code(client: AsyncClient) -> None:
    response = await client.patch("/api/v1/categories/999999", json={"name": "does-not-exist"})
    assert_api_error(response, status_code=404, code="category_not_found", message="Category not found")


@pytest.mark.asyncio
async def test_legacy_project_export_endpoints_return_gone_error_code(client: AsyncClient) -> None:
    project_id = str(uuid.uuid4())
    created = await client.post(f"/api/v1/projects/{project_id}/exports")
    assert_api_error(created, status_code=410, code="exports_legacy_gone", message=exports_router.LEGACY_MESSAGE)

    listed = await client.get(f"/api/v1/projects/{project_id}/exports")
    assert_api_error(listed, status_code=410, code="exports_legacy_gone", message=exports_router.LEGACY_MESSAGE)

    downloaded = await client.get(f"/api/v1/projects/{project_id}/exports/{uuid.uuid4().hex}/download")
    assert_api_error(downloaded, status_code=410, code="exports_legacy_gone", message=exports_router.LEGACY_MESSAGE)


@pytest.mark.asyncio
async def test_dataset_version_export_download_returns_export_file_not_found_error_code(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/projects/{uuid.uuid4()}/datasets/versions/{uuid.uuid4()}/export/download")
    assert_api_error(response, status_code=404, code="export_file_not_found", message="Export file not found")


@pytest.mark.asyncio
async def test_dataset_version_export_download_returns_export_path_invalid_error_code(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(_storage_uri: str) -> Path:
        raise ValueError("bad export relpath")

    monkeypatch.setattr(
        datasets_router.dataset_store,
        "get_export_artifact",
        lambda _project_id, _dataset_version_id: {"hash": "bad-hash"},
    )
    monkeypatch.setattr(datasets_router.storage, "resolve", _raise_value_error)
    response = await client.get(f"/api/v1/projects/{uuid.uuid4()}/datasets/versions/{uuid.uuid4()}/export/download")
    payload = assert_api_error(response, status_code=400, code="export_path_invalid", message="Invalid export path")
    assert payload["error"]["details"]["reason"] == "bad export relpath"


@pytest.mark.asyncio
async def test_annotation_upsert_rejects_asset_outside_project(client: AsyncClient) -> None:
    project_a = await _create_default_task_project(client, name="project-a")
    project_b = await _create_default_task_project(client, name="project-b")

    upload = await client.post(
        f"/api/v1/projects/{project_a['id']}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    wrong_project_upsert = await client.post(
        f"/api/v1/projects/{project_b['id']}/annotations",
        json={"asset_id": asset["id"], "task_id": project_b["default_task_id"], "status": "labeled", "payload_json": {"category_ids": []}},
    )
    assert_api_error(wrong_project_upsert, status_code=404, code="not_found", message="Asset not found in project")


@pytest.mark.asyncio
async def test_annotation_submit_stale_asset_context_returns_not_found(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="stale-submit")
    project_id = project["id"]
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    deleted = await client.delete(f"/api/v1/projects/{project_id}/assets/{asset_id}")
    assert deleted.status_code == 204

    stale_submit = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset_id, "task_id": project["default_task_id"], "status": "labeled", "payload_json": {"category_ids": []}},
    )
    assert_api_error(stale_submit, status_code=404, code="not_found", message="Asset not found in project")


@pytest.mark.asyncio
async def test_delete_asset_removes_asset_content_and_annotations(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="delete-asset")
    project_id = project["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "task_id": project["default_task_id"], "status": "approved", "payload_json": {"category_ids": []}},
    )
    assert annotation.status_code == 200

    deletion = await client.delete(f"/api/v1/projects/{project_id}/assets/{asset['id']}")
    assert deletion.status_code == 204

    content = await client.get(asset["uri"])
    assert content.status_code == 404

    assets = await client.get(f"/api/v1/projects/{project_id}/assets")
    assert assets.status_code == 200
    assert assets.json() == []

    annotations = await client.get(f"/api/v1/projects/{project_id}/annotations", params={"task_id": project["default_task_id"]})
    assert annotations.status_code == 200
    assert annotations.json() == []


@pytest.mark.asyncio
async def test_delete_project_removes_related_resources(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="delete-project")
    project_id = project["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "task_id": project["default_task_id"], "status": "approved", "payload_json": {"category_ids": []}},
    )
    assert annotation.status_code == 200

    deletion = await client.delete(f"/api/v1/projects/{project_id}")
    assert deletion.status_code == 204

    project_get = await client.get(f"/api/v1/projects/{project_id}")
    assert project_get.status_code == 404

    content = await client.get(asset["uri"])
    assert content.status_code == 404


@pytest.mark.asyncio
async def test_upload_validates_project_before_storage_write(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bytes]] = []

    def _record_write(path: str, content: bytes) -> None:
        calls.append((path, content))

    monkeypatch.setattr(assets_router.storage, "write_bytes", _record_write)

    missing_project_id = str(uuid.uuid4())
    upload = await client.post(
        f"/api/v1/projects/{missing_project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert_api_error(upload, status_code=404, code="project_not_found", message="Project not found")
    assert calls == []


@pytest.mark.asyncio
async def test_asset_upload_populates_dimensions_and_relative_path(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "upload-dimensions"})).json()
    project_id = project["id"]

    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\x00\x00\x00"
        b"\x04\x00\x01\xf61\xbcf\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    relative_path = "train/cats/sample.png"
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": relative_path},
        files={"file": ("sample.png", png_1x1, "image/png")},
    )
    assert upload.status_code == 200
    uploaded_asset = upload.json()
    assert uploaded_asset["width"] == 1
    assert uploaded_asset["height"] == 1
    assert uploaded_asset["metadata_json"]["relative_path"] == relative_path


@pytest.mark.asyncio
async def test_asset_upload_defaults_relative_path_to_filename(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "upload-relative-default"})).json()
    project_id = project["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("fallback-name.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    uploaded_asset = upload.json()
    assert uploaded_asset["metadata_json"]["relative_path"] == "fallback-name.jpg"


@pytest.mark.asyncio
async def test_validation_errors_use_structured_error_shape(client: AsyncClient) -> None:
    invalid_request = await client.post("/api/v1/projects", json={})
    payload = assert_api_error(
        invalid_request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
    )
    issues = payload["error"]["details"]["issues"]
    assert isinstance(issues, list)
    assert len(issues) > 0


@pytest.mark.asyncio
async def test_annotation_geometry_validation_rejects_out_of_bounds_bbox(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="bbox-validation", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="car")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 64, "height": 64},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [40, 10, 30, 10]},
                ],
            },
        },
    )
    assert_api_error(
        invalid,
        status_code=422,
        code="annotation_geometry_out_of_bounds",
        message="Geometry coordinates must be within image bounds",
    )


@pytest.mark.asyncio
async def test_annotation_mode_rejects_geometry_for_classification_project(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="classification-mode", task_type="classification_single")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="class-a")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 80, "height": 80},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 20]},
                ],
            },
        },
    )
    payload = assert_api_error(
        invalid,
        status_code=422,
        code="annotation_task_mode_mismatch",
        message="Task mode does not allow geometry objects",
    )
    assert payload["error"]["details"]["task_kind"] == "classification"


@pytest.mark.asyncio
async def test_annotation_mode_rejects_polygon_for_bbox_project(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="bbox-mode", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="class-b")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 80, "height": 80},
                "objects": [
                    {"id": "poly-1", "kind": "polygon", "category_id": category["id"], "segmentation": [[10, 10, 30, 10, 20, 20]]},
                ],
            },
        },
    )
    payload = assert_api_error(
        invalid,
        status_code=422,
        code="annotation_task_mode_mismatch",
        message="Task mode only allows bounding box objects",
    )
    assert payload["error"]["details"]["task_kind"] == "bbox"


@pytest.mark.asyncio
async def test_annotation_mode_rejects_bbox_for_segmentation_project(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="seg-mode", task_type="segmentation")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="class-c")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 80, "height": 80},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 20]},
                ],
            },
        },
    )
    payload = assert_api_error(
        invalid,
        status_code=422,
        code="annotation_task_mode_mismatch",
        message="Task mode only allows segmentation polygon objects",
    )
    assert payload["error"]["details"]["task_kind"] == "segmentation"


@pytest.mark.asyncio
async def test_annotation_upsert_accepts_bbox_geometry_payload(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="bbox-ok", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="truck")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 128, "height": 128},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 12, 30, 20]},
                ],
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    assert payload["version"] == "2.0"
    assert payload["category_ids"] == [category["id"]]
    assert len(payload["objects"]) == 1
    assert payload["objects"][0]["kind"] == "bbox"
    assert payload["image_basis"] == {"width": 128, "height": 128}


@pytest.mark.asyncio
async def test_annotation_upsert_preserves_prediction_review_metadata(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "prediction-review", "task_type": "classification_single"})).json()
    project_id = project["id"]
    category = (
        await client.post(
            f"/api/v1/projects/{project_id}/categories",
            json={"task_id": project["default_task_id"], "name": "truck"},
        )
    ).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": project["default_task_id"],
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "prediction_review": {
                    "origin_kind": "deployment_prediction",
                    "task": "classification",
                    "deployment_id": "dep-1",
                    "deployment_name": "cls-v1",
                    "selected_class_id": category["id"],
                    "selected_class_name": "truck",
                    "score": 0.91,
                },
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    assert payload["prediction_review"] == {
        "origin_kind": "deployment_prediction",
        "task": "classification",
        "deployment_id": "dep-1",
        "deployment_name": "cls-v1",
        "selected_class_id": category["id"],
        "selected_class_name": "truck",
        "score": 0.91,
    }


@pytest.mark.asyncio
async def test_annotation_upsert_accepts_deployment_prediction_bbox_provenance(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "bbox-prediction-review", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (
        await client.post(
            f"/api/v1/projects/{project_id}/categories",
            json={"task_id": project["default_task_id"], "name": "truck"},
        )
    ).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": project["default_task_id"],
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 128, "height": 128},
                "objects": [
                    {
                        "id": "bbox-1",
                        "kind": "bbox",
                        "category_id": category["id"],
                        "bbox": [10, 12, 30, 20],
                        "provenance": {
                            "origin_kind": "deployment_prediction",
                            "source_model": "detector-v1",
                            "confidence": 0.88,
                            "review_decision": "accepted",
                        },
                    },
                ],
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    assert payload["objects"][0]["provenance"] == {
        "origin_kind": "deployment_prediction",
        "source_model": "detector-v1",
        "confidence": 0.88,
        "review_decision": "accepted",
    }


@pytest.mark.asyncio
async def test_annotation_upsert_accepts_segmentation_geometry_payload(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="seg-ok", task_type="segmentation")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="person")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 128, "height": 128},
                "objects": [
                    {"id": "poly-1", "kind": "polygon", "category_id": category["id"], "segmentation": [[20, 20, 40, 20, 38, 36]]},
                ],
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    assert payload["version"] == "2.0"
    assert payload["category_ids"] == [category["id"]]
    assert len(payload["objects"]) == 1
    assert payload["objects"][0]["kind"] == "polygon"
    assert payload["image_basis"] == {"width": 128, "height": 128}


@pytest.mark.asyncio
async def test_export_includes_bbox_geometry_records(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="geometry-export-bbox", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="person")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    upsert = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 100},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 30, 40]},
                ],
            },
        },
    )
    assert upsert.status_code == 200

    _dataset_version_id, export = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="geometry-export-bbox",
        selection_filters={"include_statuses": ["approved"]},
    )
    archive = await client.get(export["export_uri"])
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))
        assert len(coco["annotations"]) == 1
        bbox_row = coco["annotations"][0]
        assert bbox_row["bbox"] == [10.0, 10.0, 30.0, 40.0]
        assert bbox_row["area"] == 1200.0
        assert "segmentation" not in bbox_row


@pytest.mark.asyncio
async def test_export_includes_segmentation_geometry_records(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="geometry-export-seg", task_type="segmentation")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="person")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    upsert = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 100},
                "objects": [
                    {"id": "poly-1", "kind": "polygon", "category_id": category["id"], "segmentation": [[10, 10, 40, 10, 40, 30, 10, 30]]},
                ],
            },
        },
    )
    assert upsert.status_code == 200

    _dataset_version_id, export = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="geometry-export-seg",
        selection_filters={"include_statuses": ["approved"]},
    )
    archive = await client.get(export["export_uri"])
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))
        assert len(coco["annotations"]) == 1
        poly_row = coco["annotations"][0]
        assert poly_row["segmentation"] == [[10.0, 10.0, 40.0, 10.0, 40.0, 30.0, 10.0, 30.0]]
        assert poly_row["area"] > 0


@pytest.mark.asyncio
async def test_export_detection_includes_negative_images_by_default(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="detection-negatives-default", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="Road Sign")

    positive = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("positive.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert positive.status_code == 200
    positive_asset = positive.json()

    negative = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("negative.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert negative.status_code == 200
    negative_asset = negative.json()

    positive_annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": positive_asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 100},
                "objects": [{"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 15]}],
            },
        },
    )
    assert positive_annotation.status_code == 200

    negative_annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": negative_asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 100},
                "objects": [],
            },
        },
    )
    assert negative_annotation.status_code == 200

    _dataset_version_id, export = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="detection-negatives-default",
        selection_filters={"include_statuses": ["approved"]},
    )
    archive = await client.get(export["export_uri"])
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))

        assert manifest["splits"]["generation"]["notes"] == "include_negative_images=true"
        assert len(manifest["assets"]) == 2
        assert {image["id"] for image in coco["images"]} == {positive_asset["id"], negative_asset["id"]}
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["image_id"] == positive_asset["id"]


@pytest.mark.asyncio
async def test_export_detection_excludes_negative_images_when_disabled(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="detection-negatives-off", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="Road Sign")

    positive = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("positive.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert positive.status_code == 200
    positive_asset = positive.json()

    negative = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("negative.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert negative.status_code == 200
    negative_asset = negative.json()

    positive_annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": positive_asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 100},
                "objects": [{"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 15]}],
            },
        },
    )
    assert positive_annotation.status_code == 200

    negative_annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": negative_asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 100},
                "objects": [],
            },
        },
    )
    assert negative_annotation.status_code == 200

    _dataset_version_id, export = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="detection-negatives-off",
        selection_filters={"include_statuses": ["approved"], "include_negative_images": False},
    )
    archive = await client.get(export["export_uri"])
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))

        assert manifest["splits"]["generation"]["notes"] == "include_negative_images=false"
        assert len(manifest["assets"]) == 1
        assert manifest["assets"][0]["asset_id"] == positive_asset["id"]
        assert {image["id"] for image in coco["images"]} == {positive_asset["id"]}
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["image_id"] == positive_asset["id"]


@pytest.mark.asyncio
async def test_export_normalizes_category_names_to_lowercase_slug(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="slug-classes")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="Rock Face", display_order=1)

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "task_id": task_id, "status": "approved", "payload_json": {"category_ids": [category["id"]]}},
    )
    assert annotation.status_code == 200

    _dataset_version_id, export = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="slug-classes",
        selection_filters={"include_statuses": ["approved"]},
    )
    archive = await client.get(export["export_uri"])
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))

        cls = manifest["label_schema"]["classes"][0]
        assert manifest["label_schema"]["rules"]["names_normalized"] == "lowercase_slug"
        assert cls["name"] == "rock_face"
        assert cls["display_name"] == "Rock Face"
        assert coco["categories"][0]["name"] == "rock_face"


@pytest.mark.asyncio
async def test_export_hash_is_stable_for_equivalent_geometry_object_order(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="geometry-hash", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="bike")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    payload_a = {
        "version": "2.0",
        "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
        "image_basis": {"width": 100, "height": 100},
        "objects": [
            {"id": "bbox-b", "kind": "bbox", "category_id": category["id"], "bbox": [60, 10, 20, 10]},
            {"id": "bbox-a", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 10]},
        ],
    }
    payload_b = {
        "version": "2.0",
        "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
        "image_basis": {"width": 100, "height": 100},
        "objects": [
            {"id": "bbox-a", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 10]},
            {"id": "bbox-b", "kind": "bbox", "category_id": category["id"], "bbox": [60, 10, 20, 10]},
        ],
    }

    upsert_a = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "task_id": task_id, "status": "approved", "payload_json": payload_a},
    )
    assert upsert_a.status_code == 200

    _dataset_version_id_a, export_a = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="geometry-hash-a",
        selection_filters={"include_statuses": ["approved"]},
    )
    hash_a = export_a["hash"]

    upsert_b = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "task_id": task_id, "status": "approved", "payload_json": payload_b},
    )
    assert upsert_b.status_code == 200

    _dataset_version_id_b, export_b = await _create_dataset_export(
        client,
        project_id=project_id,
        task_id=task_id,
        version_name="geometry-hash-b",
        selection_filters={"include_statuses": ["approved"]},
    )
    hash_b = export_b["hash"]

    assert hash_a == hash_b


@pytest.mark.asyncio
async def test_regression_classification_preserves_label_when_classification_block_empty(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="reg-classification", task_type="classification_single")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="mountain")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "category_id": category["id"],
                "category_ids": [category["id"]],
                "classification": {"category_ids": [], "primary_category_id": None},
                "objects": [],
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    # Regression expectation: label must persist even if classification block is empty.
    assert payload["category_ids"] == [category["id"]]
    assert payload["classification"]["primary_category_id"] == category["id"]


@pytest.mark.asyncio
async def test_regression_bbox_preserves_class_from_object_when_classification_block_empty(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="reg-bbox", task_type="bbox")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="lake")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [], "primary_category_id": None},
                "image_basis": {"width": 100, "height": 80},
                "objects": [{"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 15]}],
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    # Regression expectation: geometry class should backfill classification fields.
    assert payload["category_ids"] == [category["id"]]
    assert payload["classification"]["primary_category_id"] == category["id"]


@pytest.mark.asyncio
async def test_regression_segmentation_preserves_class_from_object_when_classification_block_empty(client: AsyncClient) -> None:
    project = await _create_default_task_project(client, name="reg-seg", task_type="segmentation")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_task_scoped_category(client, project_id=project_id, task_id=task_id, name="stone")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "labeled",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [], "primary_category_id": None},
                "image_basis": {"width": 100, "height": 80},
                "objects": [
                    {"id": "poly-1", "kind": "polygon", "category_id": category["id"], "segmentation": [[10, 10, 30, 10, 20, 25]]},
                ],
            },
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload_json"]
    # Regression expectation: geometry class should backfill classification fields.
    assert payload["category_ids"] == [category["id"]]
    assert payload["classification"]["primary_category_id"] == category["id"]
