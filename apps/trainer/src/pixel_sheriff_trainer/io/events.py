from __future__ import annotations

import json
from typing import Any

from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.time import utc_now_iso


class EventLog:
    def __init__(self, storage: ExperimentStorage) -> None:
        self.storage = storage

    def append(self, project_id: str, experiment_id: str, attempt: int, event: dict[str, Any]) -> int:
        payload = dict(event)
        payload.setdefault("attempt", attempt)
        payload.setdefault("ts", utc_now_iso())

        path = self.storage.events_path(project_id, experiment_id, attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

        meta_path = self.storage.events_meta_path(project_id, experiment_id, attempt)
        current_meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                current_meta = {}
        line_count = int(current_meta.get("line_count") or 0) + 1
        meta_path.write_text(
            json.dumps({"line_count": line_count, "updated_at": utc_now_iso()}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return line_count

