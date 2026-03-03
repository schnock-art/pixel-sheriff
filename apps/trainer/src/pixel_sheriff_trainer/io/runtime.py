from __future__ import annotations

import json
from typing import Any

from pixel_sheriff_trainer.io.storage import ExperimentStorage


def write_runtime_info(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    payload: dict[str, Any],
) -> None:
    run_path = storage.runtime_path(project_id, experiment_id, attempt)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    latest_path = storage.runtime_path(project_id, experiment_id, None)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
