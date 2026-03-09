import asyncio

from sheriff_worker.main import Worker
from sheriff_worker.jobs import extract_frames
from sheriff_worker.queues.broker import InMemoryBroker


def test_worker_jobs() -> None:
    broker = InMemoryBroker()
    worker = Worker(broker)

    broker.enqueue("extract_frames", {"video_uri": "video.mp4", "fps": 2})
    assert worker.tick()["frames_extracted"] == 2

    broker.enqueue("build_export_zip", {"dataset_version_id": "dv1"})
    assert worker.tick()["export_uri"].endswith("dv1.zip")

    broker.enqueue("inference_suggest", {"project_id": "p1"})
    assert worker.tick()["status"] == "queued"


def test_async_extract_frames_job_delegates_to_video_service(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_extract(payload: dict[str, object]) -> dict[str, object]:
        captured["payload"] = payload
        return {"status": "ready", "sequence_id": "seq-1", "frame_count": 4}

    monkeypatch.setattr(extract_frames, "extract_video_sequence_job", fake_extract)

    result = asyncio.run(extract_frames.run_async({"sequence_id": "seq-1", "project_id": "project-1"}))
    assert result == {"status": "ready", "sequence_id": "seq-1", "frame_count": 4}
    assert captured["payload"] == {"sequence_id": "seq-1", "project_id": "project-1"}
