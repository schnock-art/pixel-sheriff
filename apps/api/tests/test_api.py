import json
from io import BytesIO
import uuid
import zipfile

from httpx import AsyncClient
import pytest
import sheriff_api.routers.assets as assets_router


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
    assert export.json()["manifest_json"]["categories"][0]["name"] == "kitty"
    assert export.json()["export_uri"].startswith(f"/api/v1/projects/{project_id}/exports/")

    archive = await client.get(export.json()["export_uri"])
    assert archive.status_code == 200
    assert archive.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(BytesIO(archive.content), "r") as bundle:
        names = set(bundle.namelist())
        assert "manifest.json" in names
        assert "annotations.json" in names
        assert any(name.startswith("images/") for name in names)

        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        assert manifest["counts"]["assets"] == 1

        annotations = json.loads(bundle.read("annotations.json").decode("utf-8"))
        assert annotations["categories"][0]["name"] == "kitty"


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
