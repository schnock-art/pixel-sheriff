from sheriff_worker.jobs import build_export_zip, extract_frames, inference_suggest
from sheriff_worker.queues.broker import InMemoryBroker

HANDLERS = {
    "extract_frames": extract_frames.run,
    "build_export_zip": build_export_zip.run,
    "inference_suggest": inference_suggest.run,
}


class Worker:
    def __init__(self, broker: InMemoryBroker) -> None:
        self.broker = broker

    def tick(self) -> dict | None:
        job = self.broker.pop()
        if not job:
            return None
        return HANDLERS[job["job_name"]](job["payload"])
