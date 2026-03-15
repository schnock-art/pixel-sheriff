from __future__ import annotations

from typing import Any


AUGMENTATION_PROFILES = ("none", "light", "medium", "heavy", "custom")
AUGMENTATION_STEP_TYPES = ("horizontal_flip", "vertical_flip", "color_jitter", "rotate")
COLOR_JITTER_DEFAULTS = {
    "brightness": 0.15,
    "contrast": 0.15,
    "saturation": 0.1,
    "hue": 0.05,
}


def normalize_training_task(task: Any) -> str:
    normalized = str(task or "classification").strip().lower()
    if normalized in {"classification", "classification_single"}:
        return "classification"
    if normalized in {"detection", "bbox"}:
        return "detection"
    if normalized == "segmentation":
        return "segmentation"
    return "classification"


def task_default_augmentation_profile(task: Any) -> str:
    normalized = normalize_training_task(task)
    if normalized == "classification":
        return "light"
    return "none"


def preset_augmentation_steps(profile: Any) -> list[dict[str, Any]]:
    normalized = str(profile or "none").strip().lower()
    if normalized == "light":
        return [{"type": "horizontal_flip", "p": 0.5, "params": {}}]
    if normalized == "medium":
        return [
            {"type": "horizontal_flip", "p": 0.5, "params": {}},
            {"type": "color_jitter", "p": 1.0, "params": dict(COLOR_JITTER_DEFAULTS)},
        ]
    if normalized == "heavy":
        return [
            {"type": "horizontal_flip", "p": 0.5, "params": {}},
            {"type": "color_jitter", "p": 1.0, "params": dict(COLOR_JITTER_DEFAULTS)},
            {"type": "rotate", "p": 1.0, "params": {"degrees": 8.0}},
        ]
    return []


def _normalize_step(raw_step: Any) -> dict[str, Any] | None:
    if not isinstance(raw_step, dict):
        return None
    step_type = str(raw_step.get("type") or "").strip().lower()
    if step_type not in AUGMENTATION_STEP_TYPES:
        return None
    raw_probability = raw_step.get("p", 1.0)
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError):
        probability = 1.0
    params = raw_step.get("params")
    normalized_params = dict(params) if isinstance(params, dict) else {}
    return {
        "type": step_type,
        "p": max(0.0, min(1.0, probability)),
        "params": normalized_params,
    }


def summarize_augmentation_steps(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return "custom"
    segments: list[str] = []
    for raw_step in steps:
        step = _normalize_step(raw_step)
        if step is None:
            continue
        label = f"{step['type']}@{step['p']:.2f}"
        params = step["params"]
        if step["type"] == "rotate" and "degrees" in params:
            label = f"{label}({params['degrees']})"
        elif step["type"] == "color_jitter":
            parts: list[str] = []
            for key in ("brightness", "contrast", "saturation", "hue"):
                if key in params:
                    parts.append(f"{key}={params[key]}")
            if parts:
                label = f"{label}({', '.join(parts)})"
        segments.append(label)
    if not segments:
        return "custom"
    return f"custom: {', '.join(segments)}"


def effective_augmentation_metadata(config_json: dict[str, Any]) -> dict[str, Any]:
    task = normalize_training_task(config_json.get("task"))
    raw_profile = config_json.get("augmentation_profile")
    profile = str(raw_profile or task_default_augmentation_profile(task)).strip().lower()
    if profile not in AUGMENTATION_PROFILES:
        profile = task_default_augmentation_profile(task)

    spec_version = config_json.get("augmentation_spec_version")
    if spec_version == 1:
        if profile == "custom":
            steps = config_json.get("augmentation_steps")
            return {
                "augmentation": "custom",
                "augmentation_mode": "custom",
                "augmentation_summary": summarize_augmentation_steps(steps),
            }
        return {
            "augmentation": profile,
            "augmentation_mode": profile,
            "augmentation_summary": profile,
        }

    if task != "classification":
        return {
            "augmentation": "none",
            "augmentation_mode": "none",
            "augmentation_summary": "none",
        }

    legacy_profile = profile if profile in {"none", "light", "medium", "heavy"} else "none"
    return {
        "augmentation": legacy_profile,
        "augmentation_mode": legacy_profile,
        "augmentation_summary": legacy_profile,
    }
