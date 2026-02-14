from httpx import ASGITransport, AsyncClient
import pytest

from sheriff_api.main import app


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

        asset = (
            await client.post(
                f"/api/v1/projects/{project_id}/assets",
                json={"uri": "assets/x.jpg", "mime_type": "image/jpeg", "checksum": "abc"},
            )
        ).json()

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
