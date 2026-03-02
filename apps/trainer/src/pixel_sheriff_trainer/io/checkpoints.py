from __future__ import annotations

from typing import Any

from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.time import utc_now_iso


def default_checkpoints() -> list[dict[str, Any]]:
    return [
        {"kind": "best_metric", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None},
        {"kind": "best_loss", "epoch": None, "metric_name": "val_loss", "value": None, "updated_at": None, "uri": None},
        {"kind": "latest", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None},
    ]


def read_checkpoints(storage: ExperimentStorage, *, project_id: str, experiment_id: str, attempt: int) -> list[dict[str, Any]]:
    path = storage.checkpoints_path(project_id, experiment_id, attempt)
    if not path.exists():
        return default_checkpoints()
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_checkpoints()
    if not isinstance(payload, list):
        return default_checkpoints()
    return payload


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
) -> dict[str, Any]:
    import torch

    ckpt_dir = storage.checkpoints_dir(project_id, experiment_id, attempt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{kind}.pt"
    path = ckpt_dir / filename
    torch.save(state_dict, path)
    uri = str(path.relative_to(storage.root)).replace("\\", "/")

    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    updated_at = utc_now_iso()
    updated_row = {
        "kind": kind,
        "epoch": int(epoch),
        "metric_name": metric_name,
        "value": float(value) if isinstance(value, (int, float)) else None,
        "updated_at": updated_at,
        "uri": uri,
    }

    replaced = False
    for index, row in enumerate(rows):
        if str(row.get("kind")) == kind:
            rows[index] = updated_row
            replaced = True
            break
    if not replaced:
        rows.append(updated_row)

    import json

    out_path = storage.checkpoints_path(project_id, experiment_id, attempt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return updated_row

