import asyncio
import json
from io import BytesIO
from pathlib import Path
import uuid
import zipfile

from httpx import AsyncClient
import pytest
import sheriff_api.routers.assets as assets_router
import sheriff_api.routers.exports as exports_router
import sheriff_api.routers.models as models_router
from sheriff_api.config import get_settings


def assert_api_error(response, *, status_code: int, code: str, message: str | None = None) -> dict:
    assert response.status_code == status_code
    payload = response.json()
    assert "error" in payload
    error = payload["error"]
    assert error["code"] == code
    if message is not None:
        assert error["message"] == message
    details = error.get("details")
    assert isinstance(details, dict)
    assert details["request_path"]
    assert details["request_method"]
    return payload


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_crud_and_export_flow(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "demo"})).json()
    project_id = project["id"]

    category = (
        await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "cat", "display_order": 1})
    ).json()
    assert category["id"] > 0

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
        json={"asset_id": asset["id"], "status": "approved", "payload_json": {"category_ids": [category["id"]]}},
    )

    filtered_assets = await client.get(f"/api/v1/projects/{project_id}/assets", params={"status": "approved"})
    assert len(filtered_assets.json()) == 1

    export = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export.status_code == 200
    assert "manifest_json" in export.json()
    assert export.json()["manifest_json"]["label_schema"]["classes"][0]["name"] == "kitty"
    assert export.json()["export_uri"].startswith(f"/api/v1/projects/{project_id}/exports/")

    archive = await client.get(export.json()["export_uri"])
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
async def test_export_download_returns_export_file_not_found_error_code(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/projects/{uuid.uuid4()}/exports/{uuid.uuid4().hex}/download")
    assert_api_error(response, status_code=404, code="export_file_not_found", message="Export file not found")


@pytest.mark.asyncio
async def test_export_download_returns_export_path_invalid_error_code(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(_storage_uri: str) -> Path:
        raise ValueError("bad export relpath")

    monkeypatch.setattr(exports_router.storage, "resolve", _raise_value_error)
    response = await client.get(f"/api/v1/projects/{uuid.uuid4()}/exports/{uuid.uuid4().hex}/download")
    payload = assert_api_error(response, status_code=400, code="export_path_invalid", message="Invalid export path")
    assert payload["error"]["details"]["reason"] == "bad export relpath"


@pytest.mark.asyncio
async def test_annotation_upsert_rejects_asset_outside_project(client: AsyncClient) -> None:
    project_a = (await client.post("/api/v1/projects", json={"name": "project-a"})).json()
    project_b = (await client.post("/api/v1/projects", json={"name": "project-b"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_a['id']}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    wrong_project_upsert = await client.post(
        f"/api/v1/projects/{project_b['id']}/annotations",
        json={"asset_id": asset["id"], "status": "labeled", "payload_json": {"category_ids": []}},
    )
    assert_api_error(wrong_project_upsert, status_code=404, code="not_found", message="Asset not found in project")


@pytest.mark.asyncio
async def test_annotation_submit_stale_asset_context_returns_not_found(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "stale-submit"})).json()
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
        json={"asset_id": asset_id, "status": "labeled", "payload_json": {"category_ids": []}},
    )
    assert_api_error(stale_submit, status_code=404, code="not_found", message="Asset not found in project")


@pytest.mark.asyncio
async def test_delete_asset_removes_asset_content_and_annotations(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "delete-asset"})).json()
    project_id = project["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "status": "approved", "payload_json": {"category_ids": []}},
    )
    assert annotation.status_code == 200

    deletion = await client.delete(f"/api/v1/projects/{project_id}/assets/{asset['id']}")
    assert deletion.status_code == 204

    content = await client.get(asset["uri"])
    assert content.status_code == 404

    assets = await client.get(f"/api/v1/projects/{project_id}/assets")
    assert assets.status_code == 200
    assert assets.json() == []

    annotations = await client.get(f"/api/v1/projects/{project_id}/annotations")
    assert annotations.status_code == 200
    assert annotations.json() == []


@pytest.mark.asyncio
async def test_delete_project_removes_related_resources(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "delete-project"})).json()
    project_id = project["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "status": "approved", "payload_json": {"category_ids": []}},
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
    project = (await client.post("/api/v1/projects", json={"name": "bbox-validation", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "car"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
    project = (await client.post("/api/v1/projects", json={"name": "classification-mode", "task_type": "classification_single"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "class-a"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
        message="Project task mode does not allow geometry objects",
    )
    assert payload["error"]["details"]["task_type"] == "classification_single"


@pytest.mark.asyncio
async def test_annotation_mode_rejects_polygon_for_bbox_project(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "bbox-mode", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "class-b"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
        message="Project task mode only allows bounding box objects",
    )
    assert payload["error"]["details"]["task_type"] == "bbox"


@pytest.mark.asyncio
async def test_annotation_mode_rejects_bbox_for_segmentation_project(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "seg-mode", "task_type": "segmentation"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "class-c"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
        message="Project task mode only allows segmentation polygon objects",
    )
    assert payload["error"]["details"]["task_type"] == "segmentation"


@pytest.mark.asyncio
async def test_annotation_upsert_accepts_bbox_geometry_payload(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "bbox-ok", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "truck"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
async def test_annotation_upsert_accepts_segmentation_geometry_payload(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "seg-ok", "task_type": "segmentation"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "person"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
    project = (await client.post("/api/v1/projects", json={"name": "geometry-export-bbox", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "person"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    upsert = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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

    export = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export.status_code == 200

    archive = await client.get(export.json()["export_uri"])
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
    project = (await client.post("/api/v1/projects", json={"name": "geometry-export-seg", "task_type": "segmentation"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "person"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    upsert = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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

    export = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export.status_code == 200

    archive = await client.get(export.json()["export_uri"])
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        coco = json.loads(bundle.read("coco_instances.json").decode("utf-8"))
        assert len(coco["annotations"]) == 1
        poly_row = coco["annotations"][0]
        assert poly_row["segmentation"] == [[10.0, 10.0, 40.0, 10.0, 40.0, 30.0, 10.0, 30.0]]
        assert poly_row["area"] > 0


@pytest.mark.asyncio
async def test_export_detection_includes_negative_images_by_default(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "detection-negatives-default", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "Road Sign"})).json()

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

    export = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export.status_code == 200

    archive = await client.get(export.json()["export_uri"])
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
    project = (await client.post("/api/v1/projects", json={"name": "detection-negatives-off", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "Road Sign"})).json()

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

    export = await client.post(
        f"/api/v1/projects/{project_id}/exports",
        json={"selection_criteria_json": {"status": "approved", "include_negative_images": False}},
    )
    assert export.status_code == 200

    archive = await client.get(export.json()["export_uri"])
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
    project = (await client.post("/api/v1/projects", json={"name": "slug-classes"})).json()
    project_id = project["id"]
    category = (
        await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "Rock Face", "display_order": 1})
    ).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "status": "approved", "payload_json": {"category_ids": [category["id"]]}},
    )
    assert annotation.status_code == 200

    export = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export.status_code == 200

    archive = await client.get(export.json()["export_uri"])
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
    project = (await client.post("/api/v1/projects", json={"name": "geometry-hash", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "bike"})).json()

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
        json={"asset_id": asset["id"], "status": "approved", "payload_json": payload_a},
    )
    assert upsert_a.status_code == 200

    export_a = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export_a.status_code == 200
    hash_a = export_a.json()["hash"]

    upsert_b = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": asset["id"], "status": "approved", "payload_json": payload_b},
    )
    assert upsert_b.status_code == 200

    export_b = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export_b.status_code == 200
    hash_b = export_b.json()["hash"]

    assert hash_a == hash_b


@pytest.mark.asyncio
async def test_regression_classification_preserves_label_when_classification_block_empty(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "reg-classification", "task_type": "classification_single"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "mountain"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
    project = (await client.post("/api/v1/projects", json={"name": "reg-bbox", "task_type": "bbox"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "lake"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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
    project = (await client.post("/api/v1/projects", json={"name": "reg-seg", "task_type": "segmentation"})).json()
    project_id = project["id"]
    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "stone"})).json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    saved = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
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


async def _create_detection_project_with_manifest(client: AsyncClient, *, project_name: str) -> tuple[str, dict]:
    project = (await client.post("/api/v1/projects", json={"name": project_name, "task_type": "bbox"})).json()
    project_id = project["id"]

    category = (await client.post(f"/api/v1/projects/{project_id}/categories", json={"name": "boat"})).json()
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
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 80},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 15]},
                ],
            },
        },
    )
    assert annotation.status_code == 200

    export = await client.post(f"/api/v1/projects/{project_id}/exports", json={"selection_criteria_json": {"status": "approved"}})
    assert export.status_code == 200
    return project_id, export.json()["manifest_json"]


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
async def test_suggestions_batch_queue_and_asset_retrieval_contract(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "mal-batch"})).json()
    project_id = project["id"]
    first = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("first.jpg", b"first-image", "image/jpeg")},
    )
    second = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("second.jpg", b"second-image", "image/jpeg")},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_asset = first.json()["id"]
    second_asset = second.json()["id"]

    model = await client.post("/api/v1/models", json={"name": "mal-v1", "uri": "file:///tmp/mal-v1.onnx"})
    assert model.status_code == 200
    model_id = model.json()["id"]

    enqueued_payloads: list[dict] = []

    async def _enqueue(job_payload: dict) -> None:
        enqueued_payloads.append(job_payload)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(models_router.suggestion_queue, "enqueue_batch_job", _enqueue)
    queued = await client.post(
        f"/api/v1/projects/{project_id}/suggestions/batch",
        json={"model_id": model_id},
    )
    monkeypatch.undo()

    assert queued.status_code == 200
    queued_payload = queued.json()
    assert queued_payload["status"] == "queued"
    assert queued_payload["project_id"] == project_id
    assert queued_payload["queued"] == 2
    assert queued_payload["request_id"]
    assert len(enqueued_payloads) == 1
    assert enqueued_payloads[0]["job_type"] == "suggest_batch"
    assert enqueued_payloads[0]["project_id"] == project_id
    assert enqueued_payloads[0]["model_id"] == model_id
    assert sorted(enqueued_payloads[0]["asset_ids"]) == sorted([first_asset, second_asset])

    first_rows = await client.get(f"/api/v1/assets/{first_asset}/suggestions")
    second_rows = await client.get(f"/api/v1/assets/{second_asset}/suggestions")
    assert first_rows.status_code == 200
    assert second_rows.status_code == 200

    first_payload = first_rows.json()
    second_payload = second_rows.json()
    assert len(first_payload) == 1
    assert len(second_payload) == 1
    assert first_payload[0]["model_id"] == model_id
    assert second_payload[0]["model_id"] == model_id
    assert first_payload[0]["status"] == "pending"
    assert second_payload[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_suggestion_accept_reject_lifecycle_contract(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "mal-lifecycle"})).json()
    project_id = project["id"]
    asset_a = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("first.jpg", b"first-image", "image/jpeg")},
    )
    asset_b = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("second.jpg", b"second-image", "image/jpeg")},
    )
    assert asset_a.status_code == 200
    assert asset_b.status_code == 200
    asset_a_id = asset_a.json()["id"]
    asset_b_id = asset_b.json()["id"]

    model = await client.post("/api/v1/models", json={"name": "mal-v2", "uri": "file:///tmp/mal-v2.onnx"})
    assert model.status_code == 200
    model_id = model.json()["id"]

    async def _enqueue(_job_payload: dict) -> None:
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(models_router.suggestion_queue, "enqueue_batch_job", _enqueue)
    queued = await client.post(
        f"/api/v1/projects/{project_id}/suggestions/batch",
        json={"model_id": model_id},
    )
    monkeypatch.undo()
    assert queued.status_code == 200

    rows_a = await client.get(f"/api/v1/assets/{asset_a_id}/suggestions")
    rows_b = await client.get(f"/api/v1/assets/{asset_b_id}/suggestions")
    assert rows_a.status_code == 200
    assert rows_b.status_code == 200
    suggestion_a = rows_a.json()[0]
    suggestion_b = rows_b.json()[0]

    accepted = await client.post(
        f"/api/v1/projects/{project_id}/suggestions/{suggestion_a['id']}/accept",
        json={"annotation_payload": {"category_ids": [1]}},
    )
    assert accepted.status_code == 200
    accepted_payload = accepted.json()
    assert accepted_payload["status"] == "accepted"
    assert accepted_payload["payload_json"]["status"] == "accepted"
    assert accepted_payload["payload_json"]["annotation_payload"]["category_ids"] == [1]

    rejected = await client.post(
        f"/api/v1/projects/{project_id}/suggestions/{suggestion_b['id']}/reject",
        json={"reason": "low confidence"},
    )
    assert rejected.status_code == 200
    rejected_payload = rejected.json()
    assert rejected_payload["status"] == "rejected"
    assert rejected_payload["payload_json"]["status"] == "rejected"
    assert rejected_payload["payload_json"]["reason"] == "low confidence"

    refreshed_a = await client.get(f"/api/v1/assets/{asset_a_id}/suggestions")
    refreshed_b = await client.get(f"/api/v1/assets/{asset_b_id}/suggestions")
    assert refreshed_a.status_code == 200
    assert refreshed_b.status_code == 200
    assert refreshed_a.json()[0]["status"] == "accepted"
    assert refreshed_b.json()[0]["status"] == "rejected"


async def _create_project_model(client: AsyncClient, *, project_name: str) -> tuple[str, str]:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name=project_name)
    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    return project_id, created.json()["id"]


def _seed_experiment_run_artifacts(
    *,
    project_id: str,
    experiment_id: str,
    attempt: int = 1,
    metrics_rows: list[dict] | None = None,
    include_onnx: bool = False,
    onnx_status: str = "exported",
) -> None:
    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    run_dir = experiment_dir / "runs" / str(attempt)
    run_dir.mkdir(parents=True, exist_ok=True)

    if metrics_rows is None:
        metrics_rows = [
            {"attempt": attempt, "epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.5},
            {"attempt": attempt, "epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "val_accuracy": 0.65},
            {"attempt": attempt, "epoch": 3, "train_loss": 0.6, "val_loss": 0.5, "val_accuracy": 0.72},
        ]

    metrics_content = "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in metrics_rows)
    (run_dir / "metrics.jsonl").write_text(metrics_content, encoding="utf-8")

    evaluation_payload = {
        "schema_version": "1",
        "task": "classification",
        "computed_at": "2025-01-01T00:00:00Z",
        "split": "val",
        "num_samples": 4,
        "classes": {
            "class_order": [1, 2],
            "class_names": ["one", "two"],
            "id_to_index": {"1": 0, "2": 1},
        },
        "overall": {
            "accuracy": 0.75,
            "macro_f1": 0.73,
            "macro_precision": 0.72,
            "macro_recall": 0.74,
        },
        "per_class": [
            {"class_index": 0, "class_id": 1, "name": "one", "precision": 0.8, "recall": 0.67, "f1": 0.73, "support": 3},
            {"class_index": 1, "class_id": 2, "name": "two", "precision": 0.67, "recall": 1.0, "f1": 0.8, "support": 1},
        ],
        "confusion_matrix": {
            "matrix": [[2, 1], [0, 1]],
            "normalized_by": "none",
            "labels": {"axis": "true_rows_pred_cols"},
        },
        "samples": {
            "misclassified": [
                {
                    "asset_id": "asset-1",
                    "relative_path": "assets/a1.jpg",
                    "true_class_index": 0,
                    "pred_class_index": 1,
                    "confidence": 0.95,
                    "margin": 0.70,
                }
            ],
            "lowest_confidence_correct": [],
            "highest_confidence_wrong": [
                {
                    "asset_id": "asset-1",
                    "relative_path": "assets/a1.jpg",
                    "true_class_index": 0,
                    "pred_class_index": 1,
                    "confidence": 0.95,
                    "margin": 0.70,
                }
            ],
        },
    }
    predictions_rows = [
        {"asset_id": "asset-0", "relative_path": "assets/a0.jpg", "true_class_index": 0, "pred_class_index": 0, "confidence": 0.81, "margin": 0.52},
        {"asset_id": "asset-1", "relative_path": "assets/a1.jpg", "true_class_index": 0, "pred_class_index": 1, "confidence": 0.95, "margin": 0.70},
        {"asset_id": "asset-2", "relative_path": "assets/a2.jpg", "true_class_index": 0, "pred_class_index": 0, "confidence": 0.55, "margin": 0.11},
        {"asset_id": "asset-3", "relative_path": "assets/a3.jpg", "true_class_index": 1, "pred_class_index": 1, "confidence": 0.60, "margin": 0.22},
    ]
    predictions_content = "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in predictions_rows)
    predictions_meta = {
        "schema_version": "1",
        "attempt": attempt,
        "num_samples": len(predictions_rows),
        "task": "classification",
        "split": "val",
        "computed_at": "2025-01-01T00:00:00Z",
    }

    for target in [run_dir / "evaluation.json", experiment_dir / "evaluation.json"]:
        target.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")
    for target in [run_dir / "predictions.jsonl", experiment_dir / "predictions.jsonl"]:
        target.write_text(predictions_content, encoding="utf-8")
    for target in [run_dir / "predictions.meta.json", experiment_dir / "predictions.meta.json"]:
        target.write_text(json.dumps(predictions_meta, indent=2, sort_keys=True), encoding="utf-8")

    runtime_payload = {
        "device_selected": "cuda",
        "cuda_available": True,
        "mps_available": False,
        "amp_enabled": True,
        "torch_version": "2.x",
        "torchvision_version": "0.x",
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }
    for target in [run_dir / "runtime.json", experiment_dir / "runtime.json"]:
        target.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True), encoding="utf-8")

    (run_dir / "training.log").write_text(
        "epoch=1 train_loss=0.90 val_loss=0.80 val_accuracy=0.50\n"
        "epoch=2 train_loss=0.70 val_loss=0.60 val_accuracy=0.65\n",
        encoding="utf-8",
    )

    if include_onnx:
        onnx_dir = run_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        if onnx_status == "exported":
            (onnx_dir / "model.onnx").write_bytes(b"fake-onnx-binary-content")
        (onnx_dir / "onnx.metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": onnx_status,
                    "attempt": attempt,
                    "input_shape": [3, 224, 224],
                    "class_order": ["one", "two"],
                    "class_names": ["one", "two"],
                    "preprocess": {
                        "resize": {"width": 224, "height": 224},
                        "normalization": {"type": "imagenet"},
                    },
                    "validation": {"status": "passed" if onnx_status == "exported" else "failed"},
                    "error": None if onnx_status == "exported" else "onnxruntime validation failed",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    status_path = experiment_dir / "status.json"
    status_payload = {}
    if status_path.exists():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload.update(
        {
            "status": "completed",
            "current_run_attempt": attempt,
            "last_completed_attempt": attempt,
            "active_job_id": None,
            "error": None,
        }
    )
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")


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

    export_rows = await client.get(f"/api/v1/projects/{project_id}/exports")
    assert export_rows.status_code == 200
    exports_payload = export_rows.json()
    assert len(exports_payload) == 1
    content_hash = exports_payload[0]["hash"]

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
    assert payload["from_byte"] == 0
    assert payload["to_byte"] > 0


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
    experiment_id = created.json()["id"]

    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    assert experiment_dir.exists()

    deletion = await client.delete(f"/api/v1/projects/{project_id}")
    assert deletion.status_code == 204
    assert experiment_dir.exists() is False
