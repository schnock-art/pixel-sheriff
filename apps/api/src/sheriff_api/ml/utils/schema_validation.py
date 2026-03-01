from __future__ import annotations

from typing import Any

try:
    from sheriff_api.services.model_config_factory import collect_model_config_issues
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    if exc.name != "jsonschema":
        raise
    collect_model_config_issues = None  # type: ignore[assignment]


def validate_model_config_schema(config: dict[str, Any]) -> None:
    if collect_model_config_issues is None:
        _validate_model_config_schema_fallback(config)
        return

    issues = collect_model_config_issues(config)
    if not issues:
        return

    rendered: list[str] = []
    for issue in issues[:12]:
        path = issue.get("path", "$")
        message = issue.get("message", "invalid")
        rendered.append(f"{path}: {message}")
    raise ValueError("; ".join(rendered))


def _validate_model_config_schema_fallback(config: dict[str, Any]) -> None:
    required_top_level = [
        "schema_version",
        "name",
        "created_at",
        "source_dataset",
        "input",
        "architecture",
        "loss",
        "outputs",
        "export",
    ]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Fallback schema validation failed; missing keys: {', '.join(missing)}")
    if str(config.get("schema_version")) != "1.0":
        raise ValueError("Fallback schema validation failed; schema_version must be '1.0'")
