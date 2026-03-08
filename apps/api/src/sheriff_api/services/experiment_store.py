from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import uuid
import shutil

from sheriff_api.services.storage import LocalStorage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_summary() -> dict[str, Any]:
    return {
        "best_metric_name": None,
        "best_metric_value": None,
        "best_epoch": None,
        "last_epoch": None,
    }


def _default_checkpoints() -> list[dict[str, Any]]:
    return [
        {"kind": "best_metric", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None, "status": "pending", "error": None},
        {"kind": "best_loss", "epoch": None, "metric_name": "val_loss", "value": None, "updated_at": None, "uri": None, "status": "pending", "error": None},
        {"kind": "latest", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None, "status": "pending", "error": None},
    ]


class ExperimentStore:
    def __init__(self, storage_root: str) -> None:
        self._storage = LocalStorage(storage_root)
        self._storage.root.mkdir(parents=True, exist_ok=True)

    def _records_path(self, project_id: str) -> Path:
        return self._storage.resolve(f"experiments/{project_id}/records.json")

    def _experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        return self._storage.resolve(f"experiments/{project_id}/{experiment_id}")

    def _config_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "config.json"

    def _status_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "status.json"

    def _legacy_metrics_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "metrics.jsonl"

    def _legacy_checkpoints_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "checkpoints.json"

    def _runs_dir(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "runs"

    def _run_dir(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._runs_dir(project_id, experiment_id) / str(attempt)

    def _run_json_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "run.json"

    def _events_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "events.jsonl"

    def _events_meta_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "events.meta.json"

    def _metrics_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "metrics.jsonl"

    def _checkpoints_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "checkpoints.json"

    def _evaluation_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "evaluation.json"

    def _predictions_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "predictions.jsonl"

    def _predictions_meta_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "predictions.meta.json"

    def _runtime_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "runtime.json"

    def _training_log_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "training.log"

    def _onnx_dir(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._run_dir(project_id, experiment_id, attempt) / "onnx"

    def _onnx_model_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._onnx_dir(project_id, experiment_id, attempt) / "model.onnx"

    def _onnx_metadata_path(self, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._onnx_dir(project_id, experiment_id, attempt) / "onnx.metadata.json"

    def _latest_evaluation_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "evaluation.json"

    def _latest_predictions_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "predictions.jsonl"

    def _latest_predictions_meta_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "predictions.meta.json"

    def _latest_runtime_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "runtime.json"

    def _status_default(self) -> dict[str, Any]:
        return {
            "status": "draft",
            "cancel_requested": False,
            "current_run_attempt": None,
            "last_completed_attempt": None,
            "active_job_id": None,
            "error": None,
            "updated_at": _utc_now_iso(),
        }

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
        payload = self._read_json(self._records_path(project_id), [])
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _write_records(self, project_id: str, records: list[dict[str, Any]]) -> None:
        self._write_json(self._records_path(project_id), records)

    def _read_status(self, project_id: str, experiment_id: str) -> dict[str, Any]:
        payload = self._read_json(self._status_path(project_id, experiment_id), self._status_default())
        if not isinstance(payload, dict):
            return self._status_default()
        merged = dict(self._status_default())
        merged.update(payload)
        return merged

    def _write_status(self, project_id: str, experiment_id: str, status_row: dict[str, Any]) -> dict[str, Any]:
        next_row = dict(self._status_default())
        next_row.update(status_row)
        next_row["updated_at"] = _utc_now_iso()
        self._write_json(self._status_path(project_id, experiment_id), next_row)
        return next_row

    def _line_count(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for _ in handle:
                    count += 1
        except OSError:
            return 0
        return count

    def _read_events_meta(self, project_id: str, experiment_id: str, attempt: int) -> dict[str, Any]:
        meta_default = {"line_count": 0, "updated_at": None}
        payload = self._read_json(self._events_meta_path(project_id, experiment_id, attempt), meta_default)
        if not isinstance(payload, dict):
            return dict(meta_default)
        if not isinstance(payload.get("line_count"), int):
            payload["line_count"] = 0
        return payload

    def _write_events_meta(self, project_id: str, experiment_id: str, attempt: int, line_count: int) -> None:
        self._write_json(
            self._events_meta_path(project_id, experiment_id, attempt),
            {"line_count": int(max(0, line_count)), "updated_at": _utc_now_iso()},
        )

    def _resolve_attempt(self, project_id: str, experiment_id: str, attempt: int | None = None) -> int | None:
        if isinstance(attempt, int) and attempt >= 1:
            return attempt
        status_row = self._read_status(project_id, experiment_id)
        current_attempt = status_row.get("current_run_attempt")
        if isinstance(current_attempt, int) and current_attempt >= 1:
            return current_attempt
        return None

    def _read_checkpoints_for_attempt(self, project_id: str, experiment_id: str, attempt: int | None) -> list[dict[str, Any]]:
        if isinstance(attempt, int) and attempt >= 1:
            payload = self._read_json(self._checkpoints_path(project_id, experiment_id, attempt), _default_checkpoints())
            if isinstance(payload, list):
                return payload
            return _default_checkpoints()
        payload = self._read_json(self._legacy_checkpoints_path(project_id, experiment_id), _default_checkpoints())
        if isinstance(payload, list):
            return payload
        return _default_checkpoints()

    def _update_record(
        self,
        *,
        project_id: str,
        experiment_id: str,
        mutator: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any] | None:
        records = self._read_records(project_id)
        for record in records:
            if str(record.get("id")) != experiment_id:
                continue
            mutator(record)
            record["updated_at"] = _utc_now_iso()
            self._write_records(project_id, records)
            return record
        return None

    def list_by_project(self, project_id: str, *, model_id: str | None = None) -> list[dict[str, Any]]:
        records = self._read_records(project_id)
        if model_id:
            records = [row for row in records if str(row.get("model_id")) == model_id]
        hydrated: list[dict[str, Any]] = []
        for row in records:
            experiment_id = str(row.get("id") or "")
            if not experiment_id:
                continue
            status_row = self._read_status(project_id, experiment_id)
            merged = dict(row)
            merged["status"] = str(status_row.get("status", row.get("status", "draft")))
            merged["current_run_attempt"] = status_row.get("current_run_attempt")
            merged["last_completed_attempt"] = status_row.get("last_completed_attempt")
            merged["active_job_id"] = status_row.get("active_job_id")
            merged["error"] = status_row.get("error")
            hydrated.append(merged)
        return sorted(hydrated, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def get_index_record(self, project_id: str, experiment_id: str) -> dict[str, Any] | None:
        for row in self._read_records(project_id):
            if str(row.get("id")) == experiment_id:
                return row
        return None

    def get(
        self,
        project_id: str,
        experiment_id: str,
        *,
        metrics_limit: int | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any] | None:
        index_record = self.get_index_record(project_id, experiment_id)
        if index_record is None:
            return None

        config = self._read_json(self._config_path(project_id, experiment_id), index_record.get("config_json", {}))
        status_row = self._read_status(project_id, experiment_id)
        resolved_attempt = self._resolve_attempt(project_id, experiment_id, attempt)
        checkpoints = self._read_checkpoints_for_attempt(project_id, experiment_id, resolved_attempt)
        metrics = self.read_metrics(project_id, experiment_id, limit=metrics_limit, attempt=resolved_attempt)

        payload = dict(index_record)
        payload["config_json"] = config if isinstance(config, dict) else {}
        payload["status"] = str(status_row.get("status", index_record.get("status", "draft")))
        payload["current_run_attempt"] = status_row.get("current_run_attempt")
        payload["last_completed_attempt"] = status_row.get("last_completed_attempt")
        payload["active_job_id"] = status_row.get("active_job_id")
        payload["error"] = status_row.get("error")
        payload["checkpoints"] = checkpoints
        payload["metrics"] = metrics
        return payload

    def create(
        self,
        *,
        project_id: str,
        model_id: str,
        task_id: str | None,
        name: str,
        config_json: dict[str, Any],
        status: str = "draft",
    ) -> dict[str, Any]:
        records = self._read_records(project_id)
        timestamp = _utc_now_iso()
        experiment_id = str(uuid.uuid4())
        record = {
            "id": experiment_id,
            "project_id": project_id,
            "task_id": task_id,
            "model_id": model_id,
            "name": name,
            "created_at": timestamp,
            "updated_at": timestamp,
            "status": status,
            "config_json": config_json,
            "summary_json": _default_summary(),
            "artifacts_json": {},
        }
        records.append(record)
        self._write_records(project_id, records)

        self._write_json(self._config_path(project_id, experiment_id), config_json)
        self._write_status(
            project_id,
            experiment_id,
            {
                "status": status,
                "cancel_requested": False,
                "current_run_attempt": None,
                "last_completed_attempt": None,
                "active_job_id": None,
                "error": None,
                "updated_at": timestamp,
            },
        )
        self._write_json(self._legacy_checkpoints_path(project_id, experiment_id), _default_checkpoints())
        metrics_path = self._legacy_metrics_path(project_id, experiment_id)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")

        return self.get(project_id, experiment_id) or record

    def update(
        self,
        *,
        project_id: str,
        experiment_id: str,
        name: str | None = None,
        config_json: dict[str, Any] | None = None,
        selected_checkpoint_kind: str | None = None,
    ) -> dict[str, Any] | None:
        def mutator(record: dict[str, Any]) -> None:
            if isinstance(name, str):
                record["name"] = name
            if isinstance(config_json, dict):
                record["config_json"] = config_json
            if isinstance(selected_checkpoint_kind, str):
                artifacts = record.get("artifacts_json")
                if not isinstance(artifacts, dict):
                    artifacts = {}
                artifacts["selected_checkpoint_kind"] = selected_checkpoint_kind
                record["artifacts_json"] = artifacts

        updated = self._update_record(project_id=project_id, experiment_id=experiment_id, mutator=mutator)
        if updated is None:
            return None
        if isinstance(config_json, dict):
            self._write_json(self._config_path(project_id, experiment_id), config_json)
        return self.get(project_id, experiment_id)

    def set_status(
        self,
        *,
        project_id: str,
        experiment_id: str,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        updated = self._update_record(
            project_id=project_id,
            experiment_id=experiment_id,
            mutator=lambda record: record.__setitem__("status", status),
        )
        if updated is None:
            return None

        status_row = self._read_status(project_id, experiment_id)
        status_row["status"] = status
        status_row["error"] = error
        if status in {"completed", "failed", "canceled"}:
            attempt = status_row.get("current_run_attempt")
            if isinstance(attempt, int) and attempt >= 1:
                status_row["last_completed_attempt"] = attempt
            status_row["active_job_id"] = None
        self._write_status(project_id, experiment_id, status_row)
        return self.get(project_id, experiment_id)

    def get_status_row(self, project_id: str, experiment_id: str) -> dict[str, Any]:
        return self._read_status(project_id, experiment_id)

    def set_cancel_requested(self, *, project_id: str, experiment_id: str, cancel_requested: bool) -> dict[str, Any]:
        status_row = self._read_status(project_id, experiment_id)
        status_row["cancel_requested"] = bool(cancel_requested)
        return self._write_status(project_id, experiment_id, status_row)

    def is_cancel_requested(self, *, project_id: str, experiment_id: str) -> bool:
        status_row = self._read_status(project_id, experiment_id)
        return bool(status_row.get("cancel_requested", False))

    def read_metrics(
        self,
        project_id: str,
        experiment_id: str,
        *,
        limit: int | None = None,
        attempt: int | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(attempt, int) and attempt >= 1:
            path = self._metrics_path(project_id, experiment_id, attempt)
        else:
            path = self._legacy_metrics_path(project_id, experiment_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        rows.append(parsed)
        except OSError:
            return []

        if isinstance(limit, int) and limit > 0:
            return rows[-limit:]
        return rows

    def append_metric(self, *, project_id: str, experiment_id: str, attempt: int, metric_row: dict[str, Any]) -> None:
        path = self._metrics_path(project_id, experiment_id, attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_row, sort_keys=True))
            handle.write("\n")

    def _list_attempts(self, project_id: str, experiment_id: str) -> list[int]:
        runs_dir = self._runs_dir(project_id, experiment_id)
        if not runs_dir.exists():
            return []
        attempts: list[int] = []
        try:
            for child in runs_dir.iterdir():
                if not child.is_dir():
                    continue
                try:
                    attempt = int(child.name)
                except ValueError:
                    continue
                if attempt >= 1:
                    attempts.append(attempt)
        except OSError:
            return []
        return sorted(set(attempts), reverse=True)

    def _candidate_attempts(self, project_id: str, experiment_id: str) -> list[int]:
        status_row = self._read_status(project_id, experiment_id)
        attempts: list[int] = []
        for key in ("last_completed_attempt", "current_run_attempt"):
            value = status_row.get(key)
            if isinstance(value, int) and value >= 1:
                attempts.append(value)
        attempts.extend(self._list_attempts(project_id, experiment_id))
        ordered: list[int] = []
        seen: set[int] = set()
        for attempt in attempts:
            if attempt in seen:
                continue
            seen.add(attempt)
            ordered.append(attempt)
        return ordered

    def _latest_attempt_with_file(self, project_id: str, experiment_id: str, filename: str) -> int | None:
        for attempt in self._candidate_attempts(project_id, experiment_id):
            path = self._run_dir(project_id, experiment_id, attempt) / filename
            if path.exists() and path.is_file():
                return attempt
        return None

    def latest_attempt_with_evaluation(self, project_id: str, experiment_id: str) -> int | None:
        return self._latest_attempt_with_file(project_id, experiment_id, "evaluation.json")

    def latest_attempt_with_runtime(self, project_id: str, experiment_id: str) -> int | None:
        return self._latest_attempt_with_file(project_id, experiment_id, "runtime.json")

    def latest_attempt_with_onnx(self, project_id: str, experiment_id: str) -> int | None:
        for attempt in self._candidate_attempts(project_id, experiment_id):
            model_path = self._onnx_model_path(project_id, experiment_id, attempt)
            metadata_path = self._onnx_metadata_path(project_id, experiment_id, attempt)
            if (model_path.exists() and model_path.is_file()) or (metadata_path.exists() and metadata_path.is_file()):
                return attempt
        return None

    def get_onnx_path(self, project_id: str, experiment_id: str, attempt: int, *, file_name: str = "model.onnx") -> Path:
        normalized = file_name.strip().lower()
        if normalized == "model.onnx":
            return self._onnx_model_path(project_id, experiment_id, attempt)
        if normalized == "onnx.metadata.json":
            return self._onnx_metadata_path(project_id, experiment_id, attempt)
        raise ValueError(f"Unsupported ONNX artifact file: {file_name}")

    def get_latest_onnx(self, project_id: str, experiment_id: str) -> dict[str, Any] | None:
        attempt = self.latest_attempt_with_onnx(project_id, experiment_id)
        if not isinstance(attempt, int) or attempt < 1:
            return None
        model_path = self._onnx_model_path(project_id, experiment_id, attempt)
        metadata_path = self._onnx_metadata_path(project_id, experiment_id, attempt)
        if not model_path.exists() and not metadata_path.exists():
            return None
        return {
            "attempt": attempt,
            "model_path": model_path if model_path.exists() else None,
            "metadata_path": metadata_path if metadata_path.exists() else None,
        }

    def read_evaluation(self, project_id: str, experiment_id: str, *, attempt: int | None = None) -> tuple[int, dict[str, Any]] | None:
        resolved_attempt = attempt if isinstance(attempt, int) and attempt >= 1 else self.latest_attempt_with_evaluation(project_id, experiment_id)
        if isinstance(resolved_attempt, int) and resolved_attempt >= 1:
            path = self._evaluation_path(project_id, experiment_id, resolved_attempt)
            payload = self._read_json(path, None)
            if isinstance(payload, dict):
                return resolved_attempt, payload

        fallback_payload = self._read_json(self._latest_evaluation_path(project_id, experiment_id), None)
        if isinstance(fallback_payload, dict):
            status_row = self._read_status(project_id, experiment_id)
            fallback_attempt = status_row.get("last_completed_attempt")
            if not isinstance(fallback_attempt, int) or fallback_attempt < 1:
                fallback_attempt = status_row.get("current_run_attempt")
            if not isinstance(fallback_attempt, int) or fallback_attempt < 1:
                fallback_attempt = 1
            return int(fallback_attempt), fallback_payload
        return None

    def read_runtime(self, project_id: str, experiment_id: str, *, attempt: int | None = None) -> tuple[int, dict[str, Any]] | None:
        resolved_attempt = attempt if isinstance(attempt, int) and attempt >= 1 else self.latest_attempt_with_runtime(project_id, experiment_id)
        if isinstance(resolved_attempt, int) and resolved_attempt >= 1:
            payload = self._read_json(self._runtime_path(project_id, experiment_id, resolved_attempt), None)
            if isinstance(payload, dict):
                return resolved_attempt, payload

        fallback_payload = self._read_json(self._latest_runtime_path(project_id, experiment_id), None)
        if isinstance(fallback_payload, dict):
            status_row = self._read_status(project_id, experiment_id)
            fallback_attempt = status_row.get("last_completed_attempt")
            if not isinstance(fallback_attempt, int) or fallback_attempt < 1:
                fallback_attempt = status_row.get("current_run_attempt")
            if not isinstance(fallback_attempt, int) or fallback_attempt < 1:
                fallback_attempt = 1
            return int(fallback_attempt), fallback_payload
        return None

    def read_training_log_chunk(
        self,
        project_id: str,
        experiment_id: str,
        *,
        from_byte: int = 0,
        max_bytes: int = 65536,
        attempt: int | None = None,
    ) -> dict[str, Any] | None:
        resolved_attempt = attempt if isinstance(attempt, int) and attempt >= 1 else self._latest_attempt_with_file(project_id, experiment_id, "training.log")
        if not isinstance(resolved_attempt, int) or resolved_attempt < 1:
            return None
        path = self._training_log_path(project_id, experiment_id, resolved_attempt)
        if not path.exists() or not path.is_file():
            return None
        try:
            file_size = path.stat().st_size
        except OSError:
            return None

        safe_max_bytes = max(1, min(int(max_bytes), 512 * 1024))
        requested_start = max(0, int(from_byte))
        start = requested_start
        if start > file_size:
            start = 0
        read_limit = max(0, file_size - start)
        to_read = min(safe_max_bytes, read_limit)
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                payload = handle.read(to_read)
        except OSError:
            return None

        return {
            "attempt": resolved_attempt,
            "from_byte": int(start),
            "to_byte": int(start + len(payload)),
            "content": payload.decode("utf-8", errors="replace"),
        }

    def read_predictions(
        self,
        project_id: str,
        experiment_id: str,
        *,
        attempt: int | None = None,
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None] | None:
        resolved_attempt = attempt if isinstance(attempt, int) and attempt >= 1 else self._latest_attempt_with_file(project_id, experiment_id, "predictions.jsonl")
        if isinstance(resolved_attempt, int) and resolved_attempt >= 1:
            predictions_path = self._predictions_path(project_id, experiment_id, resolved_attempt)
            meta = self._read_json(self._predictions_meta_path(project_id, experiment_id, resolved_attempt), None)
            rows: list[dict[str, Any]] = []
            if predictions_path.exists() and predictions_path.is_file():
                try:
                    with predictions_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                parsed = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(parsed, dict):
                                rows.append(parsed)
                except OSError:
                    return None
                return resolved_attempt, rows, meta if isinstance(meta, dict) else None

        fallback_path = self._latest_predictions_path(project_id, experiment_id)
        if not fallback_path.exists() or not fallback_path.is_file():
            return None
        fallback_meta = self._read_json(self._latest_predictions_meta_path(project_id, experiment_id), None)
        rows: list[dict[str, Any]] = []
        try:
            with fallback_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        rows.append(parsed)
        except OSError:
            return None
        fallback_attempt = fallback_meta.get("attempt") if isinstance(fallback_meta, dict) else None
        if not isinstance(fallback_attempt, int) or fallback_attempt < 1:
            fallback_attempt = self._read_status(project_id, experiment_id).get("last_completed_attempt")
        if not isinstance(fallback_attempt, int) or fallback_attempt < 1:
            fallback_attempt = 1
        return int(fallback_attempt), rows, fallback_meta if isinstance(fallback_meta, dict) else None

    def set_checkpoints(self, *, project_id: str, experiment_id: str, attempt: int, checkpoints: list[dict[str, Any]]) -> None:
        self._write_json(self._checkpoints_path(project_id, experiment_id, attempt), checkpoints)

    def set_summary(self, *, project_id: str, experiment_id: str, summary_json: dict[str, Any]) -> dict[str, Any] | None:
        updated = self._update_record(
            project_id=project_id,
            experiment_id=experiment_id,
            mutator=lambda record: record.__setitem__("summary_json", summary_json),
        )
        if updated is None:
            return None
        return self.get(project_id, experiment_id)

    def init_run_attempt(
        self,
        *,
        project_id: str,
        experiment_id: str,
        job_id: str,
        dataset_export: dict[str, Any],
        task: str,
        model_family: str,
    ) -> dict[str, Any] | None:
        index_record = self.get_index_record(project_id, experiment_id)
        if index_record is None:
            return None

        status_row = self._read_status(project_id, experiment_id)
        previous_attempt = status_row.get("current_run_attempt")
        attempt = (previous_attempt if isinstance(previous_attempt, int) and previous_attempt >= 1 else 0) + 1

        run_dir = self._run_dir(project_id, experiment_id, attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self._write_json(
            self._run_json_path(project_id, experiment_id, attempt),
            {
                "attempt": attempt,
                "job_id": job_id,
                "dataset_export": dataset_export,
                "task": task,
                "model_family": model_family,
                "started_at": None,
                "ended_at": None,
            },
        )
        self._events_path(project_id, experiment_id, attempt).write_text("", encoding="utf-8")
        self._write_events_meta(project_id, experiment_id, attempt, 0)
        self._metrics_path(project_id, experiment_id, attempt).write_text("", encoding="utf-8")
        self._write_json(self._checkpoints_path(project_id, experiment_id, attempt), _default_checkpoints())

        def _record_mutator(record: dict[str, Any]) -> None:
            record["status"] = "queued"
            artifacts = record.get("artifacts_json")
            if not isinstance(artifacts, dict):
                artifacts = {}
            artifacts["last_dataset_export"] = dataset_export
            record["artifacts_json"] = artifacts

        self._update_record(
            project_id=project_id,
            experiment_id=experiment_id,
            mutator=_record_mutator,
        )
        status_row.update(
            {
                "status": "queued",
                "cancel_requested": False,
                "current_run_attempt": attempt,
                "active_job_id": job_id,
                "error": None,
            }
        )
        self._write_status(project_id, experiment_id, status_row)
        return self.get(project_id, experiment_id, attempt=attempt)

    def set_run_started_at(self, *, project_id: str, experiment_id: str, attempt: int) -> None:
        run_path = self._run_json_path(project_id, experiment_id, attempt)
        run_row = self._read_json(run_path, {})
        if not isinstance(run_row, dict):
            run_row = {}
        run_row["started_at"] = _utc_now_iso()
        self._write_json(run_path, run_row)

    def set_run_ended_at(self, *, project_id: str, experiment_id: str, attempt: int) -> None:
        run_path = self._run_json_path(project_id, experiment_id, attempt)
        run_row = self._read_json(run_path, {})
        if not isinstance(run_row, dict):
            run_row = {}
        run_row["ended_at"] = _utc_now_iso()
        self._write_json(run_path, run_row)

    def append_event(self, *, project_id: str, experiment_id: str, attempt: int, event: dict[str, Any]) -> int:
        path = self._events_path(project_id, experiment_id, attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

        meta = self._read_events_meta(project_id, experiment_id, attempt)
        next_line = int(meta.get("line_count") or 0) + 1
        self._write_events_meta(project_id, experiment_id, attempt, next_line)
        return next_line

    def read_events(
        self,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        from_line: int = 0,
    ) -> list[dict[str, Any]]:
        path = self._events_path(project_id, experiment_id, attempt)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        start = max(0, int(from_line))
        try:
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    if idx <= start:
                        continue
                    parsed: Any
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    rows.append({"line": idx, "attempt": attempt, "event": parsed})
        except OSError:
            return []
        return rows

    def get_event_line_count(self, *, project_id: str, experiment_id: str, attempt: int) -> int:
        meta = self._read_events_meta(project_id, experiment_id, attempt)
        line_count = meta.get("line_count")
        if isinstance(line_count, int) and line_count >= 0:
            return line_count
        line_count = self._line_count(self._events_path(project_id, experiment_id, attempt))
        self._write_events_meta(project_id, experiment_id, attempt, line_count)
        return line_count

    def events_path(self, *, project_id: str, experiment_id: str, attempt: int) -> Path:
        return self._events_path(project_id, experiment_id, attempt)

    def run_metadata(self, *, project_id: str, experiment_id: str, attempt: int) -> dict[str, Any]:
        payload = self._read_json(self._run_json_path(project_id, experiment_id, attempt), {})
        if isinstance(payload, dict):
            return payload
        return {}

    def delete_project_tree(self, project_id: str) -> None:
        try:
            self._storage.delete_tree(f"experiments/{project_id}")
        except ValueError:
            return

    def delete(self, *, project_id: str, experiment_id: str) -> bool:
        records = self._read_records(project_id)
        next_records = [row for row in records if str(row.get("id")) != experiment_id]
        if len(next_records) == len(records):
            return False

        experiment_dir = self._experiment_dir(project_id, experiment_id)
        trashed_dir: Path | None = None
        if experiment_dir.exists():
            trash_root = self._storage.resolve(f"experiments/{project_id}/.trash")
            trash_root.mkdir(parents=True, exist_ok=True)
            trashed_dir = trash_root / f"{experiment_id}-{uuid.uuid4()}"
            experiment_dir.rename(trashed_dir)

        try:
            self._write_records(project_id, next_records)
        except Exception:
            if trashed_dir is not None and trashed_dir.exists() and not experiment_dir.exists():
                trashed_dir.rename(experiment_dir)
            raise

        if trashed_dir is not None and trashed_dir.exists():
            shutil.rmtree(trashed_dir, ignore_errors=False)
        return True
