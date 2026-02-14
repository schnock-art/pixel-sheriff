import json
import asyncio
from io import BytesIO
import uuid
import zipfile

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio

from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.main import app


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def reset_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_health() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_crud_and_export_flow() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
async def test_asset_upload_and_content() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
async def test_upload_rejects_unknown_project() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        missing_project_id = str(uuid.uuid4())
        upload = await client.post(
            f"/api/v1/projects/{missing_project_id}/assets/upload",
            files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
        )
        assert upload.status_code == 404
        assert upload.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_annotation_upsert_rejects_asset_outside_project() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
        assert wrong_project_upsert.status_code == 404
        assert wrong_project_upsert.json()["detail"] == "Asset not found in project"


@pytest.mark.asyncio
async def test_delete_asset_removes_asset_content_and_annotations() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
async def test_delete_project_removes_related_resources() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
