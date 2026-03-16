from __future__ import annotations

from httpx import AsyncClient
import pytest
import sheriff_api.routers.models as models_router

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
