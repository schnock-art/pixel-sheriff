from __future__ import annotations

import asyncio
import json
import logging
import os

from redis.asyncio import Redis

from sheriff_worker.jobs import build_export_zip, extract_frames, inference_suggest
from sheriff_worker.queues.broker import InMemoryBroker

logger = logging.getLogger(__name__)

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


async def run_redis_worker() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    queue_key = os.getenv("MEDIA_QUEUE_KEY", "pixel_sheriff:media_jobs:v1")
    redis = Redis.from_url(redis_url, decode_responses=True)
    logger.info("Media worker listening on %s", queue_key)
    try:
        while True:
            item = await redis.blpop(queue_key, timeout=5)
            if item is None:
                continue

            _queue_name, payload_raw = item
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                logger.exception("Ignoring invalid media job payload: %s", payload_raw)
                continue

            job_type = str(payload.get("job_type") or "").strip()
            if job_type != "extract_video_frames":
                logger.warning("Ignoring unknown media job type: %s", job_type)
                continue

            try:
                result = await extract_frames.run_async(payload)
                logger.info("Completed media job %s", result)
            except Exception:
                logger.exception("Media job failed")
    finally:
        await redis.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run_redis_worker())
