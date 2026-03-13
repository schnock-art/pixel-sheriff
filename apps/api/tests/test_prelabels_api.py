from __future__ import annotations

import json
import httpx
from httpx import AsyncClient
import pytest
from sqlalchemy import select
import struct
import zlib

import sheriff_api.routers.prelabels as prelabels_router
import sheriff_api.routers.video_imports as video_imports_router
import sheriff_api.services.prelabel_adapters as prelabel_adapters
import sheriff_api.services.prelabels as prelabels_service
from sheriff_api.db.models import Asset, AssetSequence, PrelabelProposal, PrelabelSession, Task
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


async def _upload_sequence_frame(
    client: AsyncClient,
    *,
    project_id: str,
    sequence_id: str,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    file_name: str | None = None,
    content: bytes | None = None,
    mime_type: str = "image/png",
) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence_id}/frames",
        data={"frame_index": str(frame_index), "timestamp_seconds": str(timestamp_seconds)},
        files={"file": (file_name or f"frame_{frame_index + 1:06d}.png", content or _png_bytes(), mime_type)},
    )
    assert response.status_code == 200
    return response.json()


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


async def _create_sequence_with_frame(
    client: AsyncClient,
    *,
    project_id: str,
    task_id: str,
    name: str,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    file_name: str | None = None,
    content: bytes | None = None,
    mime_type: str = "image/png",
) -> tuple[dict, dict]:
    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={"task_id": task_id, "name": name, "fps": 2},
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]
    uploaded = await _upload_sequence_frame(
        client,
        project_id=project_id,
        sequence_id=sequence["id"],
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        file_name=file_name,
        content=content,
        mime_type=mime_type,
    )
    return sequence, uploaded


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
async def test_prelabel_source_status_reports_florence_device(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_warmup(payload: dict[str, object]) -> dict[str, object]:
        assert payload["model_name"] == "microsoft/Florence-2-base-ft"
        return {"device_selected": "cuda", "warmed": True}

    monkeypatch.setattr(prelabels_service.inference_client, "warmup_florence", fake_warmup)

    project = await _create_project(client, name="prelabel-source-status")
    project_id = project["id"]
    task_id = project["default_task_id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/source-status",
        json={
            "source_type": "florence2",
            "prompts": ["person"],
            "frame_sampling": {"mode": "every_n_frames", "value": 1},
            "confidence_threshold": 0.25,
            "max_detections_per_frame": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source_type"] == "florence2"
    assert payload["source_label"] == "Florence-2"
    assert payload["device_selected"] == "cuda"


@pytest.mark.asyncio
async def test_prelabel_source_status_retries_transient_florence_warmup_failures(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    request = httpx.Request("POST", "http://trainer/infer/florence/warmup")
    response = httpx.Response(
        503,
        request=request,
        content=json.dumps({"detail": {"code": "florence_load_failed", "message": "transient"}}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )

    async def fake_warmup(payload: dict[str, object]) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        assert payload["model_name"] == "microsoft/Florence-2-base-ft"
        if attempts == 1:
            raise httpx.HTTPStatusError("temporary failure", request=request, response=response)
        return {"device_selected": "cuda", "warmed": True}

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(prelabels_service.inference_client, "warmup_florence", fake_warmup)
    monkeypatch.setattr(prelabels_service.asyncio, "sleep", fake_sleep)

    project = await _create_project(client, name="prelabel-source-status-retry")
    project_id = project["id"]
    task_id = project["default_task_id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/source-status",
        json={
            "source_type": "florence2",
            "prompts": ["person"],
            "frame_sampling": {"mode": "every_n_frames", "value": 1},
            "confidence_threshold": 0.25,
            "max_detections_per_frame": 5,
        },
    )
    assert response.status_code == 200
    assert attempts == 2
    assert response.json()["device_selected"] == "cuda"


@pytest.mark.asyncio
async def test_webcam_live_prelabels_use_florence2_adapter_for_stream_frames(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued_payloads: list[dict[str, object]] = []
    adapter_calls: dict[str, object] = {}

    async def fake_enqueue(self, payload: dict[str, object]) -> None:
        enqueued_payloads.append(payload)

    class FakeFlorenceAdapter:
        name = "fake-florence"

        async def warmup(self) -> None:
            return None

        async def detect(
            self,
            *,
            asset_storage_uri: str,
            prompts: list[str],
            threshold: float,
            max_detections: int,
        ) -> list[prelabels_service.DetectionResult]:
            adapter_calls["asset_storage_uri"] = asset_storage_uri
            adapter_calls["prompts"] = prompts
            adapter_calls["threshold"] = threshold
            adapter_calls["max_detections"] = max_detections
            return [
                prelabels_service.DetectionResult(
                    label_text="person",
                    score=0.91,
                    bbox_xyxy=(10.0, 20.0, 40.0, 70.0),
                    raw={"source": "fake-florence"},
                )
            ]

    def fake_florence_factory(*, model_name: str):
        adapter_calls["model_name"] = model_name
        return FakeFlorenceAdapter()

    monkeypatch.setattr(prelabels_service.PrelabelQueue, "enqueue_asset_job", fake_enqueue)
    monkeypatch.setitem(prelabels_service.PRELABEL_ADAPTER_REGISTRY, "florence2", fake_florence_factory)

    project = await _create_project(client, name="webcam-live-florence")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="person")

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={
            "task_id": task_id,
            "name": "cam-live",
            "fps": 2,
            "prelabel_config": {
                "source_type": "florence2",
                "prompts": ["person"],
                "frame_sampling": {"mode": "every_n_frames", "value": 1},
                "confidence_threshold": 0.25,
                "max_detections_per_frame": 5,
            },
        },
    )
    assert created.status_code == 200
    created_payload = created.json()
    sequence = created_payload["sequence"]
    session_id = created_payload["prelabel_session_id"]
    assert isinstance(session_id, str)

    uploaded = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "0", "timestamp_seconds": "0.0"},
        files={"file": ("frame_000001.jpg", b"frame-a", "image/jpeg")},
    )
    assert uploaded.status_code == 200
    asset = uploaded.json()

    assert len(enqueued_payloads) == 1
    assert enqueued_payloads[0]["job_type"] == "prelabel_asset"
    assert enqueued_payloads[0]["session_id"] == session_id
    assert enqueued_payloads[0]["asset_id"] == asset["id"]

    result = await prelabels_service.process_prelabel_asset_job(dict(enqueued_payloads[0]))
    assert result["generated_proposals"] == 1
    assert adapter_calls["model_name"] == "microsoft/Florence-2-base-ft"
    assert adapter_calls["prompts"] == ["person"]
    assert adapter_calls["threshold"] == 0.25
    assert adapter_calls["max_detections"] == 5

    session_response = await client.get(f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}")
    assert session_response.status_code == 200
    session = session_response.json()["session"]
    assert session["source_type"] == "florence2"
    assert session["generated_proposals"] == 1

    proposals_response = await client.get(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/proposals",
        params={"asset_id": asset["id"]},
    )
    assert proposals_response.status_code == 200
    proposals = proposals_response.json()["items"]
    assert len(proposals) == 1
    assert proposals[0]["category_id"] == category["id"]
    assert proposals[0]["prompt_text"] == "person"


@pytest.mark.asyncio
async def test_webcam_live_prelabel_jobs_are_enqueued_after_asset_commit(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued_payloads: list[dict[str, object]] = []
    observed_visibility: dict[str, bool] = {}

    async def fake_enqueue(self, payload: dict[str, object]) -> None:
        enqueued_payloads.append(payload)
        async with SessionLocal() as verify_db:
            session = await verify_db.get(PrelabelSession, str(payload["session_id"]))
            asset = await verify_db.get(Asset, str(payload["asset_id"]))
            sequence = await verify_db.get(AssetSequence, session.sequence_id if session is not None else None)
            task = await verify_db.get(Task, session.task_id if session is not None else None)
            observed_visibility["session"] = session is not None
            observed_visibility["asset"] = asset is not None
            observed_visibility["sequence"] = sequence is not None
            observed_visibility["task"] = task is not None

    monkeypatch.setattr(prelabels_service.PrelabelQueue, "enqueue_asset_job", fake_enqueue)

    project = await _create_project(client, name="webcam-live-commit-order")
    project_id = project["id"]
    task_id = project["default_task_id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={
            "task_id": task_id,
            "name": "cam-commit-order",
            "fps": 2,
            "prelabel_config": {
                "source_type": "florence2",
                "prompts": ["person"],
                "frame_sampling": {"mode": "every_n_frames", "value": 1},
                "confidence_threshold": 0.25,
                "max_detections_per_frame": 5,
            },
        },
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    uploaded = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "0", "timestamp_seconds": "0.0"},
        files={"file": ("frame_000001.jpg", b"frame-a", "image/jpeg")},
    )
    assert uploaded.status_code == 200

    assert len(enqueued_payloads) == 1
    assert observed_visibility == {
        "session": True,
        "asset": True,
        "sequence": True,
        "task": True,
    }


@pytest.mark.asyncio
async def test_webcam_frame_uploads_use_unique_storage_paths_even_with_same_browser_filename(
    client: AsyncClient,
) -> None:
    project = await _create_project(client, name="webcam-frame-storage-paths")
    project_id = project["id"]
    task_id = project["default_task_id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={"task_id": task_id, "name": "cam-paths", "fps": 2},
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    first = await _upload_sequence_frame(
        client,
        project_id=project_id,
        sequence_id=sequence["id"],
        frame_index=0,
        file_name="frame.jpg",
        mime_type="image/jpeg",
        content=_png_bytes(),
    )
    second = await _upload_sequence_frame(
        client,
        project_id=project_id,
        sequence_id=sequence["id"],
        frame_index=1,
        file_name="frame.jpg",
        mime_type="image/jpeg",
        content=_png_bytes(rgba=(192, 64, 32, 255)),
    )

    assert first["metadata_json"]["storage_uri"].endswith("/frame_000001.jpg")
    assert second["metadata_json"]["storage_uri"].endswith("/frame_000002.jpg")
    assert first["metadata_json"]["storage_uri"] != second["metadata_json"]["storage_uri"]


@pytest.mark.asyncio
async def test_florence_alias_matching_maps_common_variant_labels_to_task_categories(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued_payloads: list[dict[str, object]] = []

    async def fake_enqueue(self, payload: dict[str, object]) -> None:
        enqueued_payloads.append(payload)

    async def fake_florence_detect(self, payload: dict[str, object]) -> dict[str, object]:
        assert payload["model_name"] == "microsoft/Florence-2-base-ft"
        return {
            "device_selected": "cpu",
            "boxes": [
                {"label_text": "person", "score": 0.91, "bbox": [10.0, 12.0, 60.0, 90.0]},
                {"label_text": "glasses", "score": 0.84, "bbox": [18.0, 26.0, 48.0, 42.0]},
                {"label_text": "eyes", "score": 0.76, "bbox": [20.0, 24.0, 40.0, 34.0]},
                {"label_text": "lips", "score": 0.72, "bbox": [24.0, 48.0, 42.0, 58.0]},
            ],
        }

    monkeypatch.setattr(prelabels_service.PrelabelQueue, "enqueue_asset_job", fake_enqueue)
    monkeypatch.setattr(prelabel_adapters.InferenceClient, "florence_detect", fake_florence_detect)

    project = await _create_project(client, name="florence-alias-matching")
    project_id = project["id"]
    task_id = project["default_task_id"]
    categories = {
        name: await _create_category(client, project_id=project_id, task_id=task_id, name=name)
        for name in ("human", "glass", "eye", "mouth")
    }

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={
            "task_id": task_id,
            "name": "cam-aliases",
            "fps": 2,
            "prelabel_config": {
                "source_type": "florence2",
                "prompts": ["human", "glass", "eye", "mouth"],
                "frame_sampling": {"mode": "every_n_frames", "value": 1},
                "confidence_threshold": 0.25,
                "max_detections_per_frame": 10,
            },
        },
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]
    session_id = created.json()["prelabel_session_id"]
    assert isinstance(session_id, str)

    asset = await _upload_sequence_frame(
        client,
        project_id=project_id,
        sequence_id=sequence["id"],
        content=_png_bytes(120, 96),
    )

    assert len(enqueued_payloads) == 1
    result = await prelabels_service.process_prelabel_asset_job(dict(enqueued_payloads[0]))
    assert result["generated_proposals"] == 4
    assert result["skipped_unmatched"] == 0

    session_response = await client.get(f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}")
    assert session_response.status_code == 200
    debug_by_label = {row["label_text"]: row for row in session_response.json()["session"]["debug_detections"]}
    assert debug_by_label["person"]["resolved_category_name"] == "human"
    assert debug_by_label["person"]["status"] == "matched"
    assert debug_by_label["glasses"]["resolved_category_name"] == "glass"
    assert debug_by_label["glasses"]["status"] == "matched"
    assert debug_by_label["eyes"]["resolved_category_name"] == "eye"
    assert debug_by_label["lips"]["resolved_category_name"] == "mouth"

    proposals_response = await client.get(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/proposals",
        params={"asset_id": asset["id"]},
    )
    assert proposals_response.status_code == 200
    proposals = proposals_response.json()["items"]
    assert len(proposals) == 4
    by_prompt = {proposal["prompt_text"]: proposal for proposal in proposals}
    assert by_prompt["person"]["category_id"] == categories["human"]["id"]
    assert by_prompt["glasses"]["category_id"] == categories["glass"]["id"]
    assert by_prompt["eyes"]["category_id"] == categories["eye"]["id"]
    assert by_prompt["lips"]["category_id"] == categories["mouth"]["id"]


@pytest.mark.asyncio
async def test_florence_adapter_skips_malformed_rows_and_clamps_valid_boxes(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued_payloads: list[dict[str, object]] = []

    async def fake_enqueue(self, payload: dict[str, object]) -> None:
        enqueued_payloads.append(payload)

    async def fake_florence_detect(self, payload: dict[str, object]) -> dict[str, object]:
        assert payload["model_name"] == "microsoft/Florence-2-base-ft"
        return {
            "device_selected": "cpu",
            "boxes": [
                {"label_text": "cat", "score": 0.93, "bbox": [110.0, -5.0, 20.0, 90.0]},
                {"label_text": "cat", "score": 0.81, "bbox": [12.0, 12.0, 12.0, 40.0]},
                {"label_text": "dog", "score": float("inf"), "bbox": [1.0, 2.0, 30.0, 45.0]},
                {"label_text": "", "score": 0.77, "bbox": [2.0, 3.0, 10.0, 15.0]},
                {"label_text": "dog", "score": 0.55, "bbox": [1.0, 2.0, 3.0]},
                {"label_text": "horse", "score": 0.66, "bbox": [5.0, 6.0, 25.0, 30.0]},
            ],
        }

    monkeypatch.setattr(prelabels_service.PrelabelQueue, "enqueue_asset_job", fake_enqueue)
    monkeypatch.setattr(prelabel_adapters.InferenceClient, "florence_detect", fake_florence_detect)

    project = await _create_project(client, name="florence-normalization")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="Cat")

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={
            "task_id": task_id,
            "name": "cam-normalize",
            "fps": 2,
            "prelabel_config": {
                "source_type": "florence2",
                "prompts": ["cat", "dog"],
                "frame_sampling": {"mode": "every_n_frames", "value": 1},
                "confidence_threshold": 0.25,
                "max_detections_per_frame": 6,
            },
        },
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]
    session_id = created.json()["prelabel_session_id"]
    assert isinstance(session_id, str)

    asset = await _upload_sequence_frame(
        client,
        project_id=project_id,
        sequence_id=sequence["id"],
        content=_png_bytes(100, 80),
    )

    assert len(enqueued_payloads) == 1
    result = await prelabels_service.process_prelabel_asset_job(dict(enqueued_payloads[0]))
    assert result["generated_proposals"] == 1
    assert result["skipped_unmatched"] == 1

    session_response = await client.get(f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}")
    assert session_response.status_code == 200
    debug_by_label = {row["label_text"]: row for row in session_response.json()["session"]["debug_detections"]}
    assert debug_by_label["cat"]["status"] == "matched"
    assert debug_by_label["cat"]["resolved_category_name"] == "Cat"
    assert debug_by_label["horse"]["status"] == "unmatched"
    assert debug_by_label["horse"]["resolved_category_name"] is None

    proposals_response = await client.get(
        f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/proposals",
        params={"asset_id": asset["id"]},
    )
    assert proposals_response.status_code == 200
    proposals = proposals_response.json()["items"]
    assert len(proposals) == 1
    assert proposals[0]["category_id"] == category["id"]
    assert proposals[0]["prompt_text"] == "cat"
    assert proposals[0]["bbox"] == [20.0, 0.0, 80.0, 80.0]


@pytest.mark.asyncio
async def test_close_input_keeps_live_sessions_running_until_queued_jobs_finish(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued_payloads: list[dict[str, object]] = []

    async def fake_enqueue(self, payload: dict[str, object]) -> None:
        enqueued_payloads.append(payload)

    class FakeFlorenceAdapter:
        name = "fake-florence"

        async def warmup(self) -> None:
            return None

        async def detect(
            self,
            *,
            asset_storage_uri: str,
            prompts: list[str],
            threshold: float,
            max_detections: int,
        ) -> list[prelabels_service.DetectionResult]:
            return [
                prelabels_service.DetectionResult(
                    label_text="person",
                    score=0.89,
                    bbox_xyxy=(8.0, 10.0, 36.0, 44.0),
                    raw={"asset_storage_uri": asset_storage_uri, "prompts": prompts, "threshold": threshold, "max_detections": max_detections},
                )
            ]

    monkeypatch.setattr(prelabels_service.PrelabelQueue, "enqueue_asset_job", fake_enqueue)
    monkeypatch.setitem(prelabels_service.PRELABEL_ADAPTER_REGISTRY, "florence2", lambda *, model_name: FakeFlorenceAdapter())

    project = await _create_project(client, name="webcam-close-input")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id=project_id, task_id=task_id, name="person")

    session_ids: list[str] = []
    for index, name in enumerate(("dock-a", "dock-b")):
        created = await client.post(
            f"/api/v1/projects/{project_id}/webcam-sessions",
            json={
                "task_id": task_id,
                "name": name,
                "fps": 2,
                "prelabel_config": {
                    "source_type": "florence2",
                    "prompts": ["person"],
                    "frame_sampling": {"mode": "every_n_frames", "value": 1},
                    "confidence_threshold": 0.25,
                    "max_detections_per_frame": 5,
                },
            },
        )
        assert created.status_code == 200
        created_payload = created.json()
        session_id = created_payload["prelabel_session_id"]
        assert isinstance(session_id, str)
        session_ids.append(session_id)

        sequence = created_payload["sequence"]
        uploaded = await _upload_sequence_frame(
            client,
            project_id=project_id,
            sequence_id=sequence["id"],
            frame_index=0,
            timestamp_seconds=index / 2,
            file_name=f"frame_{index + 1:06d}.png",
            content=_png_bytes(64, 48),
            mime_type="image/png",
        )
        assert isinstance(uploaded["id"], str)

    assert len(enqueued_payloads) == 2

    for session_id in session_ids:
        close_response = await client.post(
            f"/api/v1/projects/{project_id}/tasks/{task_id}/prelabels/{session_id}/close-input",
        )
        assert close_response.status_code == 200
        assert close_response.json()["session"]["status"] == "running"

    for payload in enqueued_payloads:
        result = await prelabels_service.process_prelabel_asset_job(dict(payload))
        assert result["status"] == "completed"

    async with SessionLocal() as db:
        for session_id in session_ids:
            session = await db.get(PrelabelSession, session_id)
            assert session is not None
            assert session.status == "completed"
            proposals = list(
                (
                    await db.execute(
                        select(PrelabelProposal).where(PrelabelProposal.session_id == session_id)
                    )
                ).scalars().all()
            )
            assert len(proposals) == 1
            assert proposals[0].category_id == category["id"]


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
