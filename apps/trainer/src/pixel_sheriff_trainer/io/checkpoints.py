from __future__ import annotations

import json
import queue
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.time import utc_now_iso


def default_checkpoints() -> list[dict[str, Any]]:
    return [
        {
            "kind": "best_metric",
            "epoch": None,
            "metric_name": None,
            "value": None,
            "updated_at": None,
            "uri": None,
            "status": "pending",
            "error": None,
        },
        {
            "kind": "best_loss",
            "epoch": None,
            "metric_name": "val_loss",
            "value": None,
            "updated_at": None,
            "uri": None,
            "status": "pending",
            "error": None,
        },
        {
            "kind": "latest",
            "epoch": None,
            "metric_name": None,
            "value": None,
            "updated_at": None,
            "uri": None,
            "status": "pending",
            "error": None,
        },
    ]


def read_checkpoints(storage: ExperimentStorage, *, project_id: str, experiment_id: str, attempt: int) -> list[dict[str, Any]]:
    path = storage.checkpoints_path(project_id, experiment_id, attempt)
    if not path.exists():
        return default_checkpoints()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_checkpoints()
    if not isinstance(payload, list):
        return default_checkpoints()
    return payload


def _write_checkpoints(storage: ExperimentStorage, *, project_id: str, experiment_id: str, attempt: int, rows: list[dict[str, Any]]) -> None:
    out_path = storage.checkpoints_path(project_id, experiment_id, attempt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def _upsert_checkpoint_row(rows: list[dict[str, Any]], updated_row: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(updated_row.get("kind"))
    replaced = False
    for index, row in enumerate(rows):
        if str(row.get("kind")) == kind:
            rows[index] = updated_row
            replaced = True
            break
    if not replaced:
        rows.append(updated_row)
    return rows


def _latest_epoch_files_sorted(ckpt_dir: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("latest_epoch_*.pt"):
        match = re.match(r"latest_epoch_(\d+)\.pt$", path.name)
        if not match:
            continue
        try:
            rows.append((int(match.group(1)), path))
        except ValueError:
            continue
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def save_checkpoint(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    kind: str,
    epoch: int,
    metric_name: str | None,
    value: float | None,
    state_dict: dict[str, Any],
    keep_last: int = 1,
) -> dict[str, Any]:
    import torch

    ckpt_dir = storage.checkpoints_dir(project_id, experiment_id, attempt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if kind == "latest":
        versioned_path = ckpt_dir / f"latest_epoch_{int(epoch)}.pt"
        torch.save(state_dict, versioned_path)
        path = ckpt_dir / "latest.pt"
        shutil.copy2(versioned_path, path)

        keep_count = max(1, int(keep_last))
        for _, stale_path in _latest_epoch_files_sorted(ckpt_dir)[keep_count:]:
            try:
                stale_path.unlink()
            except OSError:
                pass
    else:
        path = ckpt_dir / f"{kind}.pt"
        torch.save(state_dict, path)

    uri = str(path.relative_to(storage.root)).replace("\\", "/")

    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    updated_row = {
        "kind": kind,
        "epoch": int(epoch),
        "metric_name": metric_name,
        "value": float(value) if isinstance(value, (int, float)) else None,
        "updated_at": utc_now_iso(),
        "uri": uri,
        "status": "ok",
        "error": None,
    }
    rows = _upsert_checkpoint_row(rows, updated_row)
    _write_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt, rows=rows)
    return updated_row


def record_checkpoint_error(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    kind: str,
    epoch: int,
    metric_name: str | None,
    value: float | None,
    error: str,
) -> dict[str, Any]:
    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    previous_row = next((row for row in rows if str(row.get("kind")) == kind), {})
    updated_row = {
        "kind": kind,
        "epoch": int(epoch),
        "metric_name": metric_name,
        "value": float(value) if isinstance(value, (int, float)) else None,
        "updated_at": utc_now_iso(),
        "uri": previous_row.get("uri"),
        "status": "error",
        "error": str(error),
    }
    rows = _upsert_checkpoint_row(rows, updated_row)
    _write_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt, rows=rows)
    return updated_row


def _checkpoint_priority(kind: str) -> int:
    priorities = {
        "best_metric": 0,
        "best_loss": 1,
        "latest": 2,
    }
    return priorities.get(kind, 99)


def _remove_checkpoint_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    path.unlink()


def compact_completed_checkpoints(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
) -> list[str]:
    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    epoch_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") != "ok":
            continue
        epoch = row.get("epoch")
        uri = str(row.get("uri") or "")
        if not isinstance(epoch, int) or epoch < 1 or not uri:
            continue
        epoch_groups.setdefault(epoch, []).append(row)

    compacted_kinds: list[str] = []
    for epoch, group in epoch_groups.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda row: (_checkpoint_priority(str(row.get("kind") or "")), str(row.get("kind") or "")))
        canonical_row = ordered[0]
        canonical_uri = str(canonical_row.get("uri") or "")
        if not canonical_uri:
            continue
        canonical_path = storage.resolve(canonical_uri)
        if not canonical_path.exists() or not canonical_path.is_file():
            continue

        for duplicate_row in ordered[1:]:
            duplicate_kind = str(duplicate_row.get("kind") or "")
            duplicate_uri = str(duplicate_row.get("uri") or "")
            if duplicate_uri and duplicate_uri != canonical_uri:
                duplicate_path = storage.resolve(duplicate_uri)
                if duplicate_kind == "latest":
                    _remove_checkpoint_file(storage.checkpoints_dir(project_id, experiment_id, attempt) / f"latest_epoch_{epoch}.pt")
                _remove_checkpoint_file(duplicate_path)
            elif duplicate_kind == "latest":
                _remove_checkpoint_file(storage.checkpoints_dir(project_id, experiment_id, attempt) / f"latest_epoch_{epoch}.pt")

            duplicate_row["uri"] = canonical_uri
            duplicate_row["updated_at"] = utc_now_iso()
            compacted_kinds.append(duplicate_kind)

    if compacted_kinds:
        _write_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt, rows=rows)
    return compacted_kinds


@dataclass(frozen=True)
class CheckpointWriteJob:
    kind: str
    epoch: int
    metric_name: str | None
    value: float | None
    state_dict: dict[str, Any]


@dataclass(frozen=True)
class CheckpointWriteResult:
    kind: str
    epoch: int
    ok: bool
    row: dict[str, Any] | None
    error: str | None = None
    dropped: bool = False


EnqueueDisposition = Literal["queued", "dropped"]


class AsyncCheckpointWriter:
    def __init__(
        self,
        storage: ExperimentStorage,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        keep_last: int,
        max_queue_size: int = 8,
    ) -> None:
        self._storage = storage
        self._project_id = project_id
        self._experiment_id = experiment_id
        self._attempt = attempt
        self._keep_last = max(1, int(keep_last))
        self._jobs: queue.Queue[CheckpointWriteJob | object] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._results: queue.Queue[CheckpointWriteResult] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._sentinel: object = object()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker_loop, name="checkpoint-writer", daemon=True)
        self._thread.start()

    def enqueue(
        self,
        *,
        kind: str,
        epoch: int,
        metric_name: str | None,
        value: float | None,
        state_dict: dict[str, Any],
    ) -> EnqueueDisposition:
        job = CheckpointWriteJob(
            kind=kind,
            epoch=int(epoch),
            metric_name=metric_name,
            value=float(value) if isinstance(value, (int, float)) else None,
            state_dict=state_dict,
        )
        if kind == "latest":
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                self._results.put(
                    CheckpointWriteResult(
                        kind=kind,
                        epoch=int(epoch),
                        ok=False,
                        row=None,
                        dropped=True,
                        error="queue_full_latest_dropped",
                    )
                )
                return "dropped"
            return "queued"

        # best_* writes are loss-sensitive metadata and should not be dropped.
        self._jobs.put(job)
        return "queued"

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is self._sentinel:
                self._jobs.task_done()
                break
            assert isinstance(job, CheckpointWriteJob)
            try:
                row = save_checkpoint(
                    self._storage,
                    project_id=self._project_id,
                    experiment_id=self._experiment_id,
                    attempt=self._attempt,
                    kind=job.kind,
                    epoch=job.epoch,
                    metric_name=job.metric_name,
                    value=job.value,
                    state_dict=job.state_dict,
                    keep_last=self._keep_last,
                )
            except Exception as exc:
                row = record_checkpoint_error(
                    self._storage,
                    project_id=self._project_id,
                    experiment_id=self._experiment_id,
                    attempt=self._attempt,
                    kind=job.kind,
                    epoch=job.epoch,
                    metric_name=job.metric_name,
                    value=job.value,
                    error=str(exc),
                )
                self._results.put(
                    CheckpointWriteResult(
                        kind=job.kind,
                        epoch=job.epoch,
                        ok=False,
                        row=row,
                        error=str(exc),
                    )
                )
            else:
                self._results.put(
                    CheckpointWriteResult(
                        kind=job.kind,
                        epoch=job.epoch,
                        ok=True,
                        row=row,
                    )
                )
            finally:
                self._jobs.task_done()

    def drain_results(self) -> list[CheckpointWriteResult]:
        rows: list[CheckpointWriteResult] = []
        while True:
            try:
                rows.append(self._results.get_nowait())
            except queue.Empty:
                break
        return rows

    def flush_and_stop(self) -> None:
        if self._thread is None:
            return
        self._jobs.join()
        self._jobs.put(self._sentinel)
        self._jobs.join()
        self._thread.join(timeout=5.0)
        self._thread = None
