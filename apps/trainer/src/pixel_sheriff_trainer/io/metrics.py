from __future__ import annotations

import json
from typing import Any

from pixel_sheriff_trainer.io.storage import ExperimentStorage


def append_metric(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    metric_row: dict[str, Any],
) -> None:
    path = storage.metrics_path(project_id, experiment_id, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric_row, sort_keys=True))
        handle.write("\n")

