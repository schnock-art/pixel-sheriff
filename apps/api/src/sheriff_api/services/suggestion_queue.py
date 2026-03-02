from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from sheriff_api.config import get_settings


class SuggestionQueue:
    def __init__(self, *, redis_url: str | None = None, queue_key: str | None = None) -> None:
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._queue_key = queue_key or settings.suggestion_queue_key

    async def enqueue_batch_job(self, job_payload: dict[str, Any]) -> None:
        redis = Redis.from_url(self._redis_url, decode_responses=True)
        try:
            await redis.rpush(self._queue_key, json.dumps(job_payload, separators=(",", ":")))
        finally:
            await redis.aclose()

