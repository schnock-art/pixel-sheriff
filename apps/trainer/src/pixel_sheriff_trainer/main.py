from __future__ import annotations

import asyncio
import os
from pathlib import Path

from redis.asyncio import Redis

from pixel_sheriff_trainer.jobs import parse_train_job
from pixel_sheriff_trainer.runner import TrainRunner


async def worker_loop() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    storage_root = os.getenv("STORAGE_ROOT", "/app/data")
    if not Path(storage_root).is_absolute() and Path("/app/data").exists():
        # In containers, relative storage roots can silently write to ephemeral paths.
        print(f"[trainer] warning: relative STORAGE_ROOT={storage_root}; falling back to /app/data", flush=True)
        storage_root = "/app/data"
    queue_key = os.getenv("JOB_QUEUE_KEY", "pixel_sheriff:train_jobs:v1")
    timeout_seconds = max(1, int(os.getenv("TRAINER_POLL_SECONDS", "5")))

    print(f"[trainer] boot redis={redis_url} queue={queue_key} storage={storage_root}", flush=True)
    redis = Redis.from_url(redis_url, decode_responses=True)
    runner = TrainRunner(storage_root)

    try:
        while True:
            popped = await redis.blpop(queue_key, timeout=timeout_seconds)
            if popped is None:
                continue
            _queue_name, raw_payload = popped
            try:
                job = parse_train_job(raw_payload)
            except Exception as exc:
                print(f"[trainer] invalid job payload: {exc}", flush=True)
                continue
            result = runner.process(job)
            print(
                f"[trainer] processed job_id={job.job_id} project={job.project_id} experiment={job.experiment_id} attempt={job.attempt} result={result}",
                flush=True,
            )
    finally:
        await redis.aclose()


def main() -> None:
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
