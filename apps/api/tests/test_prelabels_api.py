from __future__ import annotations

from httpx import AsyncClient
import pytest

import sheriff_api.routers.prelabels as prelabels_router
import sheriff_api.routers.video_imports as video_imports_router
from sheriff_api.db.models import PrelabelProposal, PrelabelSession
from sheriff_api.db.session import SessionLocal


async def _create_project(client: AsyncClient, *, name: str) -> dict:
    response = await client.post("/api/v1/projects", json={"name": name, "task_type": "bbox"})
    assert response.status_code == 200
    return response.json()


async def _create_category(client: AsyncClient, *, project_id: str, task_id: str, name: str) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": name},
    )
    assert response.status_code == 200
    return response.json()


async def _create_sequence_with_frame(client: AsyncClient, *, project_id: str, task_id: str, name: str) -> tuple[dict, dict]:
    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={"task_id": task_id, "name": name, "fps": 2},
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    uploaded = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "0", "timestamp_seconds": "0.0"},
        files={"file": ("frame_000001.jpg", b"frame-a", "image/jpeg")},
    )
    assert uploaded.status_code == 200
    return sequence, uploaded.json()


@pytest.mark.asyncio
async def test_video_import_prelabel_config_creates_session_and_passes_job_payload(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued: dict[str, object] = {}

    async def fake_enqueue(payload: dict[str, object]) -> None:
        enqueued["payload"] = payload

    monkeypatch.setattr(video_imports_router.media_queue, "enqueue_extract_video_job", fake_enqueue)

    project = await _create_project(client, name="video-prelabels")
    project_id = project["id"]
    task_id = project["default_task_id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/video-imports",
        data={
            "task_id": task_id,
            "fps": "2",
            "max_frames": "12",
            "name": "clip-session",
            "prelabel_config": '{"source_type":"florence2","prompts":["person"],"frame_sampling":{"mode":"every_n_frames","value":3},"confidence_threshold":0.3,"max_detections_per_frame":5}',
        },
        files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["prelabel_session_id"], str)
    assert enqueued["payload"]["prelabel_session_id"] == payload["prelabel_session_id"]

    session_response = await client.get(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{payload['prelabel_session_id']}"
    )
    assert session_response.status_code == 200
    session = session_response.json()["session"]
    assert session["sequence_id"] == payload["sequence"]["id"]
    assert session["source_type"] == "florence2"


@pytest.mark.asyncio
async def test_accept_prelabel_proposals_merges_once_and_is_idempotent(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_enqueue_existing_sequence_assets_for_session(*args, **kwargs) -> dict:
        return {"status": "queued", "enqueued": 0}

    monkeypatch.setattr(
        prelabels_router,
        "enqueue_existing_sequence_assets_for_session",
        fake_enqueue_existing_sequence_assets_for_session,
    )

    project = await _create_project(client, name="accept-prelabels")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="person")
    sequence, asset = await _create_sequence_with_frame(client, project_id=project_id, task_id=task_id, name="cam-a")

    created = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels",
        json={
            "sequence_id": sequence["id"],
            "source_type": "florence2",
            "prompts": ["person"],
            "frame_sampling": {"mode": "every_n_frames", "value": 1},
            "confidence_threshold": 0.25,
            "max_detections_per_frame": 5,
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session"]["id"]

    async with SessionLocal() as db:
        session = await db.get(PrelabelSession, session_id)
        assert session is not None
        proposal = PrelabelProposal(
            session_id=session_id,
            asset_id=asset["id"],
            project_id=project_id,
            task_id=task_id,
            category_id=category["id"],
            label_text="person",
            prompt_text="person",
            confidence=0.88,
            bbox_json=[10.0, 12.0, 40.0, 24.0],
            status="pending",
        )
        db.add(proposal)
        await db.commit()
        proposal_id = proposal.id

    first_accept = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/accept",
        json={"proposal_ids": [proposal_id]},
    )
    assert first_accept.status_code == 200
    assert first_accept.json()["updated"] == 1

    second_accept = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/accept",
        json={"proposal_ids": [proposal_id]},
    )
    assert second_accept.status_code == 200
    assert second_accept.json()["updated"] == 0

    annotations = await client.get(f"/api/v1/projects/{project_id}/annotations", params={"task_id": task_id})
    assert annotations.status_code == 200
    payload = annotations.json()[0]["payload_json"]
    assert payload["classification"]["category_ids"] == [category["id"]]
    assert len(payload["objects"]) == 1
    assert payload["objects"][0]["provenance"]["proposal_id"] == proposal_id
    assert payload["objects"][0]["provenance"]["review_decision"] == "accepted"


@pytest.mark.asyncio
async def test_annotation_save_marks_prelabel_proposal_as_edited(client: AsyncClient) -> None:
    project = await _create_project(client, name="edit-prelabels")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category_a = await _create_category(client, project_id=project_id, task_id=task_id, name="person")
    category_b = await _create_category(client, project_id=project_id, task_id=task_id, name="helmet")
    sequence, asset = await _create_sequence_with_frame(client, project_id=project_id, task_id=task_id, name="cam-b")

    async with SessionLocal() as db:
        session = PrelabelSession(
            project_id=project_id,
            task_id=task_id,
            sequence_id=sequence["id"],
            source_type="florence2",
            source_ref="microsoft/Florence-2-base-ft",
            prompts_json=["person"],
            sampling_mode="every_n_frames",
            sampling_value=1.0,
            confidence_threshold=0.25,
            max_detections_per_frame=5,
            live_mode=False,
            status="running",
        )
        db.add(session)
        await db.flush()
        proposal = PrelabelProposal(
            session_id=session.id,
            asset_id=asset["id"],
            project_id=project_id,
            task_id=task_id,
            category_id=category_a["id"],
            label_text="person",
            prompt_text="person",
            confidence=0.82,
            bbox_json=[5.0, 6.0, 32.0, 18.0],
            status="pending",
        )
        db.add(proposal)
        await db.commit()
        session_id = session.id
        proposal_id = proposal.id

    response = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset["id"],
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {
                    "category_ids": [category_b["id"]],
                    "primary_category_id": category_b["id"],
                },
                "objects": [
                    {
                        "id": "edited-object",
                        "kind": "bbox",
                        "category_id": category_b["id"],
                        "bbox": [7, 8, 28, 20],
                        "provenance": {
                            "origin_kind": "ai_prelabel",
                            "session_id": session_id,
                            "proposal_id": proposal_id,
                            "source_model": "microsoft/Florence-2-base-ft",
                            "prompt_text": "person",
                            "confidence": 0.82,
                            "review_decision": "edited",
                        },
                    }
                ],
            },
        },
    )
    assert response.status_code == 200

    async with SessionLocal() as db:
        proposal = await db.get(PrelabelProposal, proposal_id)
        assert proposal is not None
        assert proposal.status == "edited"
        assert proposal.reviewed_category_id == category_b["id"]
        assert proposal.reviewed_bbox_json == [7.0, 8.0, 28.0, 20.0]
        assert proposal.promoted_annotation_id == response.json()["id"]
        assert proposal.promoted_object_id == "edited-object"
