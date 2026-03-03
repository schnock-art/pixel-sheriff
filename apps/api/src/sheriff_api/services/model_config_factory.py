from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


class ManifestConfigError(ValueError):
    pass


class ModelConfigValidationError(ValueError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_schema() -> dict[str, Any]:
    resolved = Path(__file__).resolve()
    repo_root_candidate: Path | None = None
    try:
        repo_root_candidate = resolved.parents[5] / "ModelConfig_schema.json"
    except IndexError:
        repo_root_candidate = None

    candidates = [
        Path.cwd() / "src" / "sheriff_api" / "schemas" / "model_config_schema.json",
        resolved.parents[1] / "schemas" / "model_config_schema.json",
    ]
    if repo_root_candidate is not None:
        candidates.append(repo_root_candidate)
    for candidate in candidates:
        if not candidate.exists():
            continue
        raw = candidate.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        return json.loads(raw)
    raise ModelConfigValidationError("ModelConfig schema file is missing or empty")


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, int):
            result.append(str(item))
            continue
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _normalize_task(raw_task: Any) -> str:
    if not isinstance(raw_task, str):
        raise ManifestConfigError("manifest.tasks.primary is required")
    value = raw_task.strip().lower()
    mapping = {
        "classification": "classification",
        "classification_single": "classification",
        "detection": "detection",
        "bbox": "detection",
        "segmentation": "segmentation",
    }
    normalized = mapping.get(value)
    if normalized is None:
        raise ManifestConfigError(f"Unsupported manifest task: {raw_task}")
    return normalized


def _resolve_classes(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    label_schema = manifest.get("label_schema")
    if not isinstance(label_schema, dict):
        raise ManifestConfigError("manifest.label_schema is required")

    class_order = _as_str_list(label_schema.get("class_order"))
    if not class_order:
        raise ManifestConfigError("manifest.label_schema.class_order must contain at least one class id")

    classes_raw = label_schema.get("classes")
    if not isinstance(classes_raw, list):
        raise ManifestConfigError("manifest.label_schema.classes must be an array")

    class_name_by_id: dict[str, str] = {}
    for row in classes_raw:
        if not isinstance(row, dict):
            continue
        class_id = row.get("id")
        if isinstance(class_id, int):
            class_id = str(class_id)
        if not isinstance(class_id, str):
            continue
        if isinstance(row.get("name"), str) and row["name"].strip():
            class_name_by_id[class_id] = row["name"].strip()
            continue
        if isinstance(row.get("display_name"), str) and row["display_name"].strip():
            class_name_by_id[class_id] = row["display_name"].strip()

    class_names: list[str] = []
    for class_id in class_order:
        class_name = class_name_by_id.get(class_id)
        if class_name is None:
            raise ManifestConfigError(f"Class id {class_id} from class_order is missing in label_schema.classes")
        class_names.append(class_name)

    return class_order, class_names


def _normalize_input(manifest: dict[str, Any]) -> dict[str, Any]:
    training_defaults = manifest.get("training_defaults")
    if not isinstance(training_defaults, dict):
        training_defaults = {}

    input_defaults = training_defaults.get("input")
    if not isinstance(input_defaults, dict):
        input_defaults = {}

    normalization_defaults = training_defaults.get("normalization")
    if not isinstance(normalization_defaults, dict):
        normalization_defaults = {}

    raw_size = input_defaults.get("recommended_size")
    input_size = [640, 640]
    if (
        isinstance(raw_size, list)
        and len(raw_size) == 2
        and isinstance(raw_size[0], int)
        and isinstance(raw_size[1], int)
        and raw_size[0] > 0
        and raw_size[1] > 0
    ):
        input_size = [raw_size[0], raw_size[1]]

    resize_policy = input_defaults.get("resize_policy")
    if resize_policy not in {"letterbox", "stretch", "longest_side_pad"}:
        resize_policy = "letterbox"

    normalization_type = normalization_defaults.get("type")
    if normalization_type not in {"imagenet", "none", "custom"}:
        normalization_type = "imagenet"

    normalization: dict[str, Any] = {"type": normalization_type}

    mean = normalization_defaults.get("mean")
    if isinstance(mean, list) and len(mean) == 3 and all(isinstance(v, (int, float)) for v in mean):
        normalization["mean"] = [float(v) for v in mean]

    std = normalization_defaults.get("std")
    if isinstance(std, list) and len(std) == 3 and all(isinstance(v, (int, float)) for v in std):
        normalization["std"] = [float(v) for v in std]

    if normalization_type == "custom" and ("mean" not in normalization or "std" not in normalization):
        raise ManifestConfigError("manifest.training_defaults.normalization for custom type requires mean and std")

    return {
        "image_channels": 3,
        "input_size": input_size,
        "resize_policy": resize_policy,
        "normalization": normalization,
    }


def _defaults_for_task(task: str, num_classes: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    if task == "classification":
        return (
            {
                "family": "resnet_classifier",
                "framework": "torchvision",
                "precision": "fp32",
                "backbone": {"name": "resnet18", "pretrained": True},
                "neck": {"type": "none"},
                "head": {"type": "linear", "num_classes": num_classes},
            },
            {"type": "classification_cross_entropy"},
            {
                "name": "classification_logits",
                "type": "task_output",
                "task": task,
                "format": "classification_logits",
            },
            "classification_logits",
        )

    if task == "detection":
        return (
            {
                "family": "retinanet",
                "framework": "torchvision",
                "precision": "fp32",
                "backbone": {"name": "resnet50", "pretrained": True},
                "neck": {"type": "fpn", "fpn_channels": 256},
                "head": {"type": "retinanet", "num_classes": num_classes},
            },
            {"type": "retinanet_default"},
            {
                "name": "coco_detections",
                "type": "task_output",
                "task": task,
                "format": "coco_detections",
            },
            "coco_detections",
        )

    return (
        {
            "family": "deeplabv3",
            "framework": "torchvision",
            "precision": "fp32",
            "backbone": {"name": "resnet50", "pretrained": True},
            "neck": {"type": "none"},
            "head": {"type": "deeplabv3_head", "num_classes": num_classes},
        },
        {"type": "deeplabv3_default"},
        {
            "name": "coco_segmentation",
            "type": "task_output",
            "task": task,
            "format": "coco_segmentation",
        },
        "coco_segmentation",
    )


def build_default_model_config(
    *,
    model_name: str,
    dataset_manifest_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict):
        raise ManifestConfigError("manifest.tasks is required")
    task = _normalize_task(tasks.get("primary"))

    class_order, class_names = _resolve_classes(manifest)
    num_classes = len(class_order)
    if num_classes < 1:
        raise ManifestConfigError("At least one class is required to build a model config")

    input_spec = _normalize_input(manifest)
    architecture, loss, primary_output, primary_output_name = _defaults_for_task(task, num_classes)

    return {
        "schema_version": "1.0",
        "name": model_name,
        "created_at": _utc_now_iso(),
        "source_dataset": {
            "manifest_id": dataset_manifest_id,
            "task": task,
            "num_classes": num_classes,
            "class_order": class_order,
            "class_names": class_names,
        },
        "input": input_spec,
        "architecture": architecture,
        "loss": loss,
        "outputs": {
            "primary": primary_output,
            "aux": [],
        },
        "export": {
            "onnx": {
                "enabled": True,
                "opset": 17,
                "dynamic_shapes": {
                    "enabled": True,
                    "batch": True,
                    "height_width": False,
                },
                "output_names": [primary_output_name],
            }
        },
    }


def validate_model_config(config: dict[str, Any]) -> None:
    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda err: list(err.absolute_path))
    if not errors:
        return

    rendered_errors: list[str] = []
    for error in errors[:8]:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        rendered_errors.append(f"{path}: {error.message}")
    raise ModelConfigValidationError("; ".join(rendered_errors))


def collect_model_config_issues(config: dict[str, Any]) -> list[dict[str, str]]:
    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda err: list(err.absolute_path))
    issues: list[dict[str, str]] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        issues.append({"path": path, "message": error.message})
    return issues
