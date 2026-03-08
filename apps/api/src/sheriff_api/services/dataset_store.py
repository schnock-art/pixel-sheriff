from __future__ import annotations

from datetime import datetime, timezone
from importlib import resources
import json
from pathlib import Path
from typing import Any
import uuid

from jsonschema import Draft202012Validator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DatasetStoreValidationError(ValueError):
    def __init__(self, issues: list[dict[str, str]]) -> None:
        super().__init__("Dataset version validation failed")
        self.issues = issues


class DatasetStore:
    def __init__(self, storage_root: str, *, schema_path: Path | None = None) -> None:
        self._root = Path(storage_root)
        self._root.mkdir(parents=True, exist_ok=True)
        schema_json = self._load_schema_json(schema_path)
        self._validator = Draft202012Validator(schema_json)

    @staticmethod
    def _load_schema_json(schema_path: Path | None) -> dict[str, Any]:
        if isinstance(schema_path, Path):
            return json.loads(schema_path.read_text(encoding="utf-8"))

        resolved = Path(__file__).resolve()
        candidates = [
            resolved.parents[5] / "packages" / "contracts" / "schemas" / "dataset_version_v2.schema.json",
            resolved.parents[1] / "schemas_json" / "dataset_version_v2.schema.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))

        resource = resources.files("sheriff_api").joinpath("schemas_json/dataset_version_v2.schema.json")
        return json.loads(resource.read_text(encoding="utf-8"))

    def _path(self, project_id: str) -> Path:
        return self._root / "datasets" / project_id / "datasets.json"

    def _default_doc(self, project_id: str) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "project_id": project_id,
            "active_dataset_version_id": None,
            "items": [],
            "meta": {
                "archived_ids": [],
                "export_artifacts": {},
            },
        }

    def _read_doc(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        if not path.exists():
            return self._default_doc(project_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_doc(project_id)
        if not isinstance(payload, dict):
            return self._default_doc(project_id)

        items = payload.get("items")
        meta = payload.get("meta")
        archived_ids = meta.get("archived_ids") if isinstance(meta, dict) else []
        export_artifacts = meta.get("export_artifacts") if isinstance(meta, dict) else {}
        return {
            "schema_version": "1",
            "project_id": project_id,
            "active_dataset_version_id": payload.get("active_dataset_version_id")
            if isinstance(payload.get("active_dataset_version_id"), str)
            else None,
            "items": [item for item in items if isinstance(item, dict)] if isinstance(items, list) else [],
            "meta": {
                "archived_ids": [item for item in archived_ids if isinstance(item, str)] if isinstance(archived_ids, list) else [],
                "export_artifacts": export_artifacts if isinstance(export_artifacts, dict) else {},
            },
        }

    def _write_doc(self, project_id: str, payload: dict[str, Any]) -> None:
        path = self._path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _validate_dataset_version(self, payload: dict[str, Any]) -> None:
        errors = sorted(self._validator.iter_errors(payload), key=lambda err: err.path)
        if not errors:
            return
        issues: list[dict[str, str]] = []
        for error in errors[:64]:
            path = "$"
            if error.path:
                path = "$." + ".".join(str(item) for item in error.path)
            issues.append({"path": path, "message": error.message})
        raise DatasetStoreValidationError(issues)

    def list_versions(self, project_id: str, task_id: str | None = None) -> dict[str, Any]:
        doc = self._read_doc(project_id)
        archived_ids = set(doc["meta"]["archived_ids"])
        active_id = doc.get("active_dataset_version_id")
        items = sorted(doc["items"], key=lambda item: str(item.get("created_at", "")), reverse=True)
        if isinstance(task_id, str) and task_id.strip():
            items = [item for item in items if str(item.get("task_id") or "") == task_id.strip()]
        return {
            "active_dataset_version_id": active_id,
            "items": [
                {
                    "version": item,
                    "is_archived": str(item.get("dataset_version_id")) in archived_ids,
                    "is_active": str(item.get("dataset_version_id")) == active_id,
                }
                for item in items
            ],
        }

    def get_version(self, project_id: str, dataset_version_id: str) -> dict[str, Any] | None:
        doc = self._read_doc(project_id)
        archived_ids = set(doc["meta"]["archived_ids"])
        for item in doc["items"]:
            if str(item.get("dataset_version_id")) != dataset_version_id:
                continue
            return {
                "version": item,
                "is_archived": dataset_version_id in archived_ids,
                "is_active": dataset_version_id == doc.get("active_dataset_version_id"),
            }
        return None

    def create_version(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        doc = self._read_doc(project_id)
        version = dict(payload)
        version.setdefault("schema_version", "2.0")
        version.setdefault("dataset_version_id", str(uuid.uuid4()))
        version.setdefault("project_id", project_id)
        version.setdefault("created_at", _utc_now_iso())
        self._validate_dataset_version(version)
        doc["items"].append(version)
        self._write_doc(project_id, doc)
        return version

    def set_active(self, project_id: str, dataset_version_id: str | None) -> None:
        doc = self._read_doc(project_id)
        if dataset_version_id is None:
            doc["active_dataset_version_id"] = None
            self._write_doc(project_id, doc)
            return

        exists = any(str(item.get("dataset_version_id")) == dataset_version_id for item in doc["items"])
        if not exists:
            raise KeyError("dataset_version_not_found")
        doc["active_dataset_version_id"] = dataset_version_id
        self._write_doc(project_id, doc)

    def archive_version(self, project_id: str, dataset_version_id: str, archived: bool = True) -> None:
        doc = self._read_doc(project_id)
        exists = any(str(item.get("dataset_version_id")) == dataset_version_id for item in doc["items"])
        if not exists:
            raise KeyError("dataset_version_not_found")
        archived_ids = [item for item in doc["meta"]["archived_ids"] if item != dataset_version_id]
        if archived:
            archived_ids.append(dataset_version_id)
        doc["meta"]["archived_ids"] = sorted(set(archived_ids))
        self._write_doc(project_id, doc)

    def delete_version(self, project_id: str, dataset_version_id: str) -> None:
        doc = self._read_doc(project_id)
        original_count = len(doc["items"])
        doc["items"] = [item for item in doc["items"] if str(item.get("dataset_version_id")) != dataset_version_id]
        if len(doc["items"]) == original_count:
            raise KeyError("dataset_version_not_found")
        doc["meta"]["archived_ids"] = [item for item in doc["meta"]["archived_ids"] if item != dataset_version_id]
        artifacts = doc["meta"]["export_artifacts"]
        if isinstance(artifacts, dict):
            artifacts.pop(dataset_version_id, None)
        if doc.get("active_dataset_version_id") == dataset_version_id:
            doc["active_dataset_version_id"] = None
        self._write_doc(project_id, doc)

    def set_export_artifact(self, project_id: str, dataset_version_id: str, artifact: dict[str, Any]) -> None:
        doc = self._read_doc(project_id)
        exists = any(str(item.get("dataset_version_id")) == dataset_version_id for item in doc["items"])
        if not exists:
            raise KeyError("dataset_version_not_found")
        export_artifacts = doc["meta"].get("export_artifacts")
        if not isinstance(export_artifacts, dict):
            export_artifacts = {}
        export_artifacts[dataset_version_id] = dict(artifact)
        doc["meta"]["export_artifacts"] = export_artifacts
        self._write_doc(project_id, doc)

    def get_export_artifact(self, project_id: str, dataset_version_id: str) -> dict[str, Any] | None:
        doc = self._read_doc(project_id)
        export_artifacts = doc["meta"].get("export_artifacts")
        if not isinstance(export_artifacts, dict):
            return None
        artifact = export_artifacts.get(dataset_version_id)
        if isinstance(artifact, dict):
            return artifact
        return None
