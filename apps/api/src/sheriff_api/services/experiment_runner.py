from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import random
from typing import Any

from sheriff_api.services.experiment_store import ExperimentStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _metric_name_for_task(task: str) -> str:
    if task == "detection":
        return "val_map"
    if task == "segmentation":
        return "val_iou"
    return "val_accuracy"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ExperimentRunnerManager:
    def __init__(self, store: ExperimentStore, *, sleep_seconds: float = 0.3) -> None:
        self._store = store
        self._sleep_seconds = sleep_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    def _key(self, project_id: str, experiment_id: str) -> str:
        return f"{project_id}:{experiment_id}"

    def subscribe(self, project_id: str, experiment_id: str) -> asyncio.Queue[dict[str, Any]]:
        key = self._key(project_id, experiment_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.setdefault(key, set()).add(queue)
        return queue

    def unsubscribe(self, project_id: str, experiment_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        key = self._key(project_id, experiment_id)
        subscribers = self._subscribers.get(key)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(key, None)

    def publish(self, project_id: str, experiment_id: str, event: dict[str, Any]) -> None:
        key = self._key(project_id, experiment_id)
        subscribers = self._subscribers.get(key)
        if not subscribers:
            return
        for queue in list(subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    def start(self, project_id: str, experiment_id: str) -> bool:
        key = self._key(project_id, experiment_id)
        existing = self._tasks.get(key)
        if existing and not existing.done():
            return False
        self._store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=False)
        task = asyncio.create_task(self._run(project_id, experiment_id), name=f"experiment-run:{key}")
        self._tasks[key] = task
        return True

    def _task_metric_value(self, *, task: str, progress: float, rng: random.Random) -> float:
        jitter = rng.uniform(-0.015, 0.015)
        if task == "detection":
            return _clamp(0.18 + (0.55 * progress) + jitter, 0.01, 0.99)
        if task == "segmentation":
            return _clamp(0.22 + (0.60 * progress) + jitter, 0.01, 0.99)
        return _clamp(0.48 + (0.50 * progress) + jitter, 0.01, 0.99)

    async def _run(self, project_id: str, experiment_id: str) -> None:
        key = self._key(project_id, experiment_id)
        try:
            record = self._store.get(project_id, experiment_id)
            if record is None:
                return

            config = record.get("config_json")
            if not isinstance(config, dict):
                config = {}
            task = str(config.get("task", "classification"))
            epochs = max(1, int(config.get("epochs", 30) or 30))
            advanced = config.get("advanced")
            seed = 1337
            if isinstance(advanced, dict):
                try:
                    seed = int(advanced.get("seed", 1337))
                except (TypeError, ValueError):
                    seed = 1337
            rng = random.Random(seed)

            metric_name = _metric_name_for_task(task)
            best_loss: float | None = None
            best_metric: float | None = None
            best_metric_epoch: int | None = None

            self.publish(project_id, experiment_id, {"type": "status", "status": "running"})

            for epoch in range(1, epochs + 1):
                if self._store.is_cancel_requested(project_id=project_id, experiment_id=experiment_id):
                    self._store.set_status(project_id=project_id, experiment_id=experiment_id, status="canceled")
                    self.publish(project_id, experiment_id, {"type": "status", "status": "canceled"})
                    self.publish(project_id, experiment_id, {"type": "done", "status": "canceled"})
                    return

                await asyncio.sleep(self._sleep_seconds)

                progress = epoch / float(epochs)
                train_loss = _clamp((1.2 * (1.0 - progress)) + rng.uniform(-0.03, 0.03), 0.03, 4.0)
                val_loss = _clamp((1.0 * (1.0 - (progress * 0.92))) + rng.uniform(-0.025, 0.025), 0.03, 4.0)
                primary_metric = self._task_metric_value(task=task, progress=progress, rng=rng)

                metric_row: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "created_at": _utc_now_iso(),
                }
                metric_row[metric_name] = round(primary_metric, 6)
                self._store.append_metric(project_id=project_id, experiment_id=experiment_id, metric_row=metric_row)

                changed_checkpoint_rows: list[dict[str, Any]] = []
                checkpoints: dict[str, dict[str, Any]] = {
                    "best_metric": {
                        "kind": "best_metric",
                        "epoch": None,
                        "metric_name": metric_name,
                        "value": None,
                        "updated_at": None,
                    },
                    "best_loss": {
                        "kind": "best_loss",
                        "epoch": None,
                        "metric_name": "val_loss",
                        "value": None,
                        "updated_at": None,
                    },
                    "latest": {
                        "kind": "latest",
                        "epoch": None,
                        "metric_name": metric_name,
                        "value": None,
                        "updated_at": None,
                    },
                }

                existing = self._store.get(project_id, experiment_id, metrics_limit=1)
                if existing and isinstance(existing.get("checkpoints"), list):
                    for row in existing["checkpoints"]:
                        if not isinstance(row, dict):
                            continue
                        kind = str(row.get("kind", ""))
                        if kind in checkpoints:
                            checkpoints[kind] = dict(checkpoints[kind], **row)

                now = _utc_now_iso()
                checkpoints["latest"] = {
                    "kind": "latest",
                    "epoch": epoch,
                    "metric_name": metric_name,
                    "value": round(primary_metric, 6),
                    "updated_at": now,
                }
                changed_checkpoint_rows.append(checkpoints["latest"])

                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    checkpoints["best_loss"] = {
                        "kind": "best_loss",
                        "epoch": epoch,
                        "metric_name": "val_loss",
                        "value": round(val_loss, 6),
                        "updated_at": now,
                    }
                    changed_checkpoint_rows.append(checkpoints["best_loss"])

                if best_metric is None or primary_metric > best_metric:
                    best_metric = primary_metric
                    best_metric_epoch = epoch
                    checkpoints["best_metric"] = {
                        "kind": "best_metric",
                        "epoch": epoch,
                        "metric_name": metric_name,
                        "value": round(primary_metric, 6),
                        "updated_at": now,
                    }
                    changed_checkpoint_rows.append(checkpoints["best_metric"])

                checkpoint_list = [checkpoints["best_metric"], checkpoints["best_loss"], checkpoints["latest"]]
                self._store.set_checkpoints(project_id=project_id, experiment_id=experiment_id, checkpoints=checkpoint_list)

                summary_json = {
                    "best_metric_name": metric_name,
                    "best_metric_value": round(best_metric, 6) if best_metric is not None else None,
                    "best_epoch": best_metric_epoch,
                    "last_epoch": epoch,
                }
                self._store.set_summary(project_id=project_id, experiment_id=experiment_id, summary_json=summary_json)

                self.publish(project_id, experiment_id, {"type": "metric", **metric_row})
                for checkpoint in changed_checkpoint_rows:
                    self.publish(project_id, experiment_id, {"type": "checkpoint", **checkpoint})

            self._store.set_cancel_requested(project_id=project_id, experiment_id=experiment_id, cancel_requested=False)
            self._store.set_status(project_id=project_id, experiment_id=experiment_id, status="completed")
            self.publish(project_id, experiment_id, {"type": "status", "status": "completed"})
            self.publish(project_id, experiment_id, {"type": "done", "status": "completed"})
        except Exception as exc:
            self._store.set_status(project_id=project_id, experiment_id=experiment_id, status="failed")
            self.publish(project_id, experiment_id, {"type": "status", "status": "failed"})
            self.publish(project_id, experiment_id, {"type": "done", "status": "failed", "message": str(exc)})
        finally:
            self._tasks.pop(key, None)

