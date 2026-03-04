from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_doc(project_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "project_id": project_id,
        "active_deployment_id": None,
        "items": [],
    }


class DeploymentStore:
    def __init__(self, storage_root: str) -> None:
        self._root = Path(storage_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self._root / "deployments" / project_id / "deployments.json"

    def _read_doc(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        if not path.exists():
            return _default_doc(project_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _default_doc(project_id)
        if not isinstance(payload, dict):
            return _default_doc(project_id)
        items = payload.get("items")
        if not isinstance(items, list):
            items = []
        active_deployment_id = payload.get("active_deployment_id")
        if not isinstance(active_deployment_id, str):
            active_deployment_id = None
        return {
            "schema_version": "1",
            "project_id": project_id,
            "active_deployment_id": active_deployment_id,
            "items": [item for item in items if isinstance(item, dict)],
        }

    def _write_doc(self, project_id: str, payload: dict[str, Any]) -> None:
        path = self._path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def list(self, project_id: str) -> dict[str, Any]:
        doc = self._read_doc(project_id)
        items = sorted(doc["items"], key=lambda item: str(item.get("created_at", "")), reverse=True)
        return {"active_deployment_id": doc.get("active_deployment_id"), "items": items}

    def get(self, project_id: str, deployment_id: str) -> dict[str, Any] | None:
        doc = self._read_doc(project_id)
        for item in doc["items"]:
            if str(item.get("deployment_id")) == deployment_id:
                return item
        return None

    def create(
        self,
        *,
        project_id: str,
        name: str,
        task_id: str | None,
        task: str,
        device_preference: str,
        source: dict[str, Any],
        model_key: str,
        is_active: bool = False,
    ) -> dict[str, Any]:
        doc = self._read_doc(project_id)
        timestamp = _utc_now_iso()
        item = {
            "deployment_id": str(uuid.uuid4()),
            "task_id": task_id,
            "name": name,
            "task": task,
            "provider": "onnxruntime",
            "device_preference": device_preference,
            "model_key": model_key,
            "source": source,
            "status": "available",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        doc["items"].append(item)
        if is_active:
            doc["active_deployment_id"] = item["deployment_id"]
        self._write_doc(project_id, doc)
        return item

    def patch(
        self,
        *,
        project_id: str,
        deployment_id: str,
        name: str | None = None,
        device_preference: str | None = None,
        status: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any] | None:
        doc = self._read_doc(project_id)
        target: dict[str, Any] | None = None
        for item in doc["items"]:
            if str(item.get("deployment_id")) != deployment_id:
                continue
            target = item
            break
        if target is None:
            return None

        if isinstance(name, str):
            target["name"] = name
        if isinstance(device_preference, str):
            target["device_preference"] = device_preference
        if isinstance(status, str):
            target["status"] = status
            if status == "archived" and doc.get("active_deployment_id") == deployment_id:
                doc["active_deployment_id"] = None
        if is_active is True:
            doc["active_deployment_id"] = deployment_id
        if is_active is False and doc.get("active_deployment_id") == deployment_id:
            doc["active_deployment_id"] = None
        target["updated_at"] = _utc_now_iso()
        self._write_doc(project_id, doc)
        return target
