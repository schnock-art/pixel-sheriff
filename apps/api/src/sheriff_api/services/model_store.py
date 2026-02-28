from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelStore:
    """Temporary file-backed model store.

    TODO: replace with DB-backed project model table once migrations are in place.
    """

    def __init__(self, storage_root: str) -> None:
        self._root = Path(storage_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _records_path(self, project_id: str) -> Path:
        return self._root / "models" / project_id / "records.json"

    def _read_records(self, project_id: str) -> list[dict[str, Any]]:
        path = self._records_path(project_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _write_records(self, project_id: str, records: list[dict[str, Any]]) -> None:
        path = self._records_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    def list_by_project(self, project_id: str) -> list[dict[str, Any]]:
        records = self._read_records(project_id)
        return sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get(self, project_id: str, model_id: str) -> dict[str, Any] | None:
        for record in self._read_records(project_id):
            if str(record.get("id")) == model_id:
                return record
        return None

    def create(self, *, project_id: str, name: str, config_json: dict[str, Any]) -> dict[str, Any]:
        records = self._read_records(project_id)
        timestamp = _utc_now_iso()
        record = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "name": name,
            "config_json": config_json,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        records.append(record)
        self._write_records(project_id, records)
        return record

