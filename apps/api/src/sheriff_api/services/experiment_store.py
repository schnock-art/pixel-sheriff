from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import uuid

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
        {"kind": "best_metric", "epoch": None, "metric_name": None, "value": None, "updated_at": None},
        {"kind": "best_loss", "epoch": None, "metric_name": "val_loss", "value": None, "updated_at": None},
        {"kind": "latest", "epoch": None, "metric_name": None, "value": None, "updated_at": None},
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

    def _metrics_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "metrics.jsonl"

    def _checkpoints_path(self, project_id: str, experiment_id: str) -> Path:
        return self._experiment_dir(project_id, experiment_id) / "checkpoints.json"

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
        return sorted(records, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def get_index_record(self, project_id: str, experiment_id: str) -> dict[str, Any] | None:
        for row in self._read_records(project_id):
            if str(row.get("id")) == experiment_id:
                return row
        return None

    def get(self, project_id: str, experiment_id: str, *, metrics_limit: int | None = None) -> dict[str, Any] | None:
        index_record = self.get_index_record(project_id, experiment_id)
        if index_record is None:
            return None

        config = self._read_json(self._config_path(project_id, experiment_id), index_record.get("config_json", {}))
        status_row = self._read_json(self._status_path(project_id, experiment_id), {})
        checkpoints = self._read_json(self._checkpoints_path(project_id, experiment_id), _default_checkpoints())
        metrics = self.read_metrics(project_id, experiment_id, limit=metrics_limit)

        payload = dict(index_record)
        payload["config_json"] = config if isinstance(config, dict) else {}
        payload["status"] = status_row.get("status", index_record.get("status", "draft"))
        payload["checkpoints"] = checkpoints if isinstance(checkpoints, list) else _default_checkpoints()
        payload["metrics"] = metrics
        return payload

    def create(
        self,
        *,
        project_id: str,
        model_id: str,
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
        self._write_json(
            self._status_path(project_id, experiment_id),
            {"status": status, "cancel_requested": False, "updated_at": timestamp},
        )
        self._write_json(self._checkpoints_path(project_id, experiment_id), _default_checkpoints())
        metrics_path = self._metrics_path(project_id, experiment_id)
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

    def set_status(self, *, project_id: str, experiment_id: str, status: str) -> dict[str, Any] | None:
        updated = self._update_record(
            project_id=project_id,
            experiment_id=experiment_id,
            mutator=lambda record: record.__setitem__("status", status),
        )
        if updated is None:
            return None

        status_path = self._status_path(project_id, experiment_id)
        status_row = self._read_json(status_path, {})
        if not isinstance(status_row, dict):
            status_row = {}
        status_row["status"] = status
        status_row["updated_at"] = _utc_now_iso()
        self._write_json(status_path, status_row)
        return self.get(project_id, experiment_id)

    def set_cancel_requested(self, *, project_id: str, experiment_id: str, cancel_requested: bool) -> None:
        status_path = self._status_path(project_id, experiment_id)
        status_row = self._read_json(status_path, {})
        if not isinstance(status_row, dict):
            status_row = {}
        status_row["cancel_requested"] = bool(cancel_requested)
        status_row["updated_at"] = _utc_now_iso()
        self._write_json(status_path, status_row)

    def is_cancel_requested(self, *, project_id: str, experiment_id: str) -> bool:
        status_row = self._read_json(self._status_path(project_id, experiment_id), {})
        if not isinstance(status_row, dict):
            return False
        return bool(status_row.get("cancel_requested", False))

    def read_metrics(self, project_id: str, experiment_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        path = self._metrics_path(project_id, experiment_id)
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

    def append_metric(self, *, project_id: str, experiment_id: str, metric_row: dict[str, Any]) -> None:
        path = self._metrics_path(project_id, experiment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_row, sort_keys=True))
            handle.write("\n")

    def set_checkpoints(self, *, project_id: str, experiment_id: str, checkpoints: list[dict[str, Any]]) -> None:
        self._write_json(self._checkpoints_path(project_id, experiment_id), checkpoints)

    def set_summary(self, *, project_id: str, experiment_id: str, summary_json: dict[str, Any]) -> dict[str, Any] | None:
        updated = self._update_record(
            project_id=project_id,
            experiment_id=experiment_id,
            mutator=lambda record: record.__setitem__("summary_json", summary_json),
        )
        if updated is None:
            return None
        return self.get(project_id, experiment_id)

    def delete_project_tree(self, project_id: str) -> None:
        try:
            self._storage.delete_tree(f"experiments/{project_id}")
        except ValueError:
            return

