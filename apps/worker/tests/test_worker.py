from sheriff_worker.main import Worker
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
