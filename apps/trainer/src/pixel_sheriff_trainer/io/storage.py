from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pixel_sheriff_trainer.utils.time import utc_now_iso


def default_status() -> dict[str, Any]:
    return {
        "status": "draft",
        "cancel_requested": False,
        "current_run_attempt": None,
        "last_completed_attempt": None,
        "active_job_id": None,
        "error": None,
        "updated_at": utc_now_iso(),
    }


class ExperimentStorage:
    def __init__(self, storage_root: str) -> None:
        self.root = Path(storage_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_uri: str) -> Path:
        root = self.root.resolve()
        candidate = (root / relative_uri).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Resolved path escapes storage root")
        return candidate

    def experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        return self.resolve(f"experiments/{project_id}/{experiment_id}")

    def records_path(self, project_id: str) -> Path:
        return self.resolve(f"experiments/{project_id}/records.json")

    def status_path(self, project_id: str, experiment_id: str) -> Path:
        return self.experiment_dir(project_id, experiment_id) / "status.json"

    def run_dir(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.experiment_dir(project_id, experiment_id) / "runs" / str(attempt)

    def run_json_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "run.json"

    def events_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "events.jsonl"

    def events_meta_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "events.meta.json"

    def metrics_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "metrics.jsonl"

    def checkpoints_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "checkpoints.json"

    def checkpoints_dir(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "checkpoints"

    def runtime_path(self, project_id: str, experiment_id: str, attempt: int | None = None) -> Path:
        if isinstance(attempt, int) and attempt >= 1:
            return self.run_dir(project_id, experiment_id, attempt) / "runtime.json"
        return self.experiment_dir(project_id, experiment_id) / "runtime.json"

    def training_log_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self.run_dir(project_id, experiment_id, attempt) / "training.log"

    def evaluation_path(self, project_id: str, experiment_id: str, attempt: int | None = None) -> Path:
        if isinstance(attempt, int) and attempt >= 1:
            return self.run_dir(project_id, experiment_id, attempt) / "evaluation.json"
        return self.experiment_dir(project_id, experiment_id) / "evaluation.json"

    def predictions_path(self, project_id: str, experiment_id: str, attempt: int | None = None) -> Path:
        if isinstance(attempt, int) and attempt >= 1:
            return self.run_dir(project_id, experiment_id, attempt) / "predictions.jsonl"
        return self.experiment_dir(project_id, experiment_id) / "predictions.jsonl"

    def predictions_meta_path(self, project_id: str, experiment_id: str, attempt: int | None = None) -> Path:
        if isinstance(attempt, int) and attempt >= 1:
            return self.run_dir(project_id, experiment_id, attempt) / "predictions.meta.json"
        return self.experiment_dir(project_id, experiment_id) / "predictions.meta.json"

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        return payload

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _read_records(self, project_id: str) -> list[dict[str, Any]]:
        payload = self._read_json(self.records_path(project_id), [])
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    def _write_records(self, project_id: str, rows: list[dict[str, Any]]) -> None:
        self._write_json(self.records_path(project_id), rows)

    def read_status(self, project_id: str, experiment_id: str) -> dict[str, Any]:
        row = self._read_json(self.status_path(project_id, experiment_id), default_status())
        if not isinstance(row, dict):
            return default_status()
        merged = default_status()
        merged.update(row)
        return merged

    def write_status(self, project_id: str, experiment_id: str, status_row: dict[str, Any]) -> dict[str, Any]:
        row = default_status()
        row.update(status_row)
        row["updated_at"] = utc_now_iso()
        self._write_json(self.status_path(project_id, experiment_id), row)
        return row

    def patch_status(self, project_id: str, experiment_id: str, **fields: Any) -> dict[str, Any]:
        current = self.read_status(project_id, experiment_id)
        current.update(fields)
        return self.write_status(project_id, experiment_id, current)

    def update_record(self, project_id: str, experiment_id: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any] | None:
        records = self._read_records(project_id)
        for row in records:
            if str(row.get("id")) != experiment_id:
                continue
            mutator(row)
            row["updated_at"] = utc_now_iso()
            self._write_records(project_id, records)
            return row
        return None

    def set_experiment_status(self, project_id: str, experiment_id: str, status: str, *, error: str | None = None) -> None:
        self.update_record(project_id, experiment_id, lambda row: row.__setitem__("status", status))
        status_row = self.read_status(project_id, experiment_id)
        status_row["status"] = status
        status_row["error"] = error
        if status in {"completed", "failed", "canceled"}:
            current_attempt = status_row.get("current_run_attempt")
            if isinstance(current_attempt, int) and current_attempt >= 1:
                status_row["last_completed_attempt"] = current_attempt
            status_row["active_job_id"] = None
        self.write_status(project_id, experiment_id, status_row)

    def set_summary(self, project_id: str, experiment_id: str, summary: dict[str, Any]) -> None:
        self.update_record(project_id, experiment_id, lambda row: row.__setitem__("summary_json", summary))

    def is_cancel_requested(self, project_id: str, experiment_id: str) -> bool:
        status_row = self.read_status(project_id, experiment_id)
        return bool(status_row.get("cancel_requested", False))

    def set_run_started(self, project_id: str, experiment_id: str, attempt: int) -> None:
        run_path = self.run_json_path(project_id, experiment_id, attempt)
        run_row = self._read_json(run_path, {})
        if not isinstance(run_row, dict):
            run_row = {}
        run_row["started_at"] = utc_now_iso()
        self._write_json(run_path, run_row)

    def set_run_ended(self, project_id: str, experiment_id: str, attempt: int) -> None:
        run_path = self.run_json_path(project_id, experiment_id, attempt)
        run_row = self._read_json(run_path, {})
        if not isinstance(run_row, dict):
            run_row = {}
        run_row["ended_at"] = utc_now_iso()
        self._write_json(run_path, run_row)
