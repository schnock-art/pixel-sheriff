from __future__ import annotations

import json
import subprocess
from pathlib import Path

from httpx import AsyncClient
import pytest
import sheriff_api.routers.video_imports as video_imports_router
import sheriff_api.services.video_frames as video_frames
from sheriff_api.config import get_settings
from sheriff_api.db.session import SessionLocal
from sheriff_api.services.storage import LocalStorage
from sheriff_api.services.video_frames import VideoFrameExtractionError, extract_video_sequence_job


async def _create_project(client: AsyncClient, *, name: str) -> dict:
    response = await client.post("/api/v1/projects", json={"name": name})
    assert response.status_code == 200
    return response.json()


async def _create_video_import(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    project_id: str,
    task_id: str | None,
    name: str,
) -> tuple[dict, dict[str, object]]:
    enqueued: dict[str, object] = {}

    async def fake_enqueue(payload: dict[str, object]) -> None:
        enqueued["payload"] = payload

    monkeypatch.setattr(video_imports_router.media_queue, "enqueue_extract_video_job", fake_enqueue)
    response = await client.post(
        f"/api/v1/projects/{project_id}/video-imports",
        data={"task_id": task_id, "fps": "2", "max_frames": "8", "name": name},
        files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    return response.json()["sequence"], enqueued["payload"]


def _completed_process(*, args: list[str], returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.asyncio
async def test_extract_video_sequence_job_persists_ready_frames(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = await _create_project(client, name="extract-ready")
    sequence, payload = await _create_video_import(
        client,
        monkeypatch,
        project_id=project["id"],
        task_id=project["default_task_id"],
        name="extract-ready",
    )

    def fake_run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "ffprobe":
            return _completed_process(
                args=args,
                returncode=0,
                stdout=json.dumps(
                    {
                        "streams": [{"width": 640, "height": 360, "avg_frame_rate": "30/1", "duration": "2.0"}],
                        "format": {"duration": "2.0"},
                    }
                ),
            )

        output_pattern = Path(args[-1])
        output_pattern.parent.mkdir(parents=True, exist_ok=True)
        (output_pattern.parent / "frame_000001.jpg").write_bytes(b"frame-a")
        (output_pattern.parent / "frame_000002.jpg").write_bytes(b"frame-b")
        return _completed_process(args=args, returncode=0)

    monkeypatch.setattr(video_frames, "_run_command", fake_run_command)

    result = await extract_video_sequence_job(
        payload,
        session_factory=SessionLocal,
        storage=LocalStorage(get_settings().storage_root),
    )

    assert result["status"] == "ready"
    assert result["frame_count"] == 2

    detail = await client.get(f"/api/v1/projects/{project['id']}/sequences/{sequence['id']}")
    assert detail.status_code == 200
    payload_json = detail.json()

    assert payload_json["status"] == "ready"
    assert payload_json["frame_count"] == 2
    assert [asset["frame_index"] for asset in payload_json["assets"]] == [0, 1]
    assert [asset["source_kind"] for asset in payload_json["assets"]] == ["video_frame", "video_frame"]

    source_video_path = Path(get_settings().storage_root) / str(payload["video_storage_uri"])
    assert not source_video_path.exists()


@pytest.mark.asyncio
async def test_extract_video_sequence_job_marks_failed_and_removes_partial_assets(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = await _create_project(client, name="extract-failed")
    sequence, payload = await _create_video_import(
        client,
        monkeypatch,
        project_id=project["id"],
        task_id=project["default_task_id"],
        name="extract-failed",
    )

    def fake_run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "ffprobe":
            return _completed_process(
                args=args,
                returncode=0,
                stdout=json.dumps(
                    {
                        "streams": [{"width": 640, "height": 360, "avg_frame_rate": "30/1", "duration": "2.0"}],
                        "format": {"duration": "2.0"},
                    }
                ),
            )
        return _completed_process(args=args, returncode=1, stderr="ffmpeg exploded")

    monkeypatch.setattr(video_frames, "_run_command", fake_run_command)

    with pytest.raises(VideoFrameExtractionError, match="ffmpeg exploded"):
        await extract_video_sequence_job(
            payload,
            session_factory=SessionLocal,
            storage=LocalStorage(get_settings().storage_root),
        )

    detail = await client.get(f"/api/v1/projects/{project['id']}/sequences/{sequence['id']}")
    assert detail.status_code == 200
    payload_json = detail.json()

    assert payload_json["status"] == "failed"
    assert payload_json["error_message"] == "ffmpeg exploded"
    assert payload_json["frame_count"] == 0
    assert payload_json["assets"] == []

    source_video_path = Path(get_settings().storage_root) / str(payload["video_storage_uri"])
    assert not source_video_path.exists()
