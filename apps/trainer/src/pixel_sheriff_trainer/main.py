from __future__ import annotations

import asyncio
import os
from pathlib import Path

from redis.asyncio import Redis
import uvicorn

from pixel_sheriff_trainer.inference.app import create_app
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


async def inference_loop() -> None:
    host = os.getenv("TRAINER_INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("TRAINER_INFERENCE_PORT", "8020"))
    app = create_app()
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config=config)
    await server.serve()


async def service_loop() -> None:
    worker_task = asyncio.create_task(worker_loop(), name="trainer-worker")
    inference_task = asyncio.create_task(inference_loop(), name="trainer-inference")
    done, pending = await asyncio.wait({worker_task, inference_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exc = task.exception()
        if exc is not None:
            raise exc


def main() -> None:
    asyncio.run(service_loop())


if __name__ == "__main__":
    main()
