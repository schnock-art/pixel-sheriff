from collections import deque


class InMemoryBroker:
    def __init__(self) -> None:
        self.queue: deque[dict] = deque()

    def enqueue(self, job_name: str, payload: dict) -> dict:
        job = {"job_name": job_name, "payload": payload}
        self.queue.append(job)
        return job

    def pop(self) -> dict | None:
        return self.queue.popleft() if self.queue else None
