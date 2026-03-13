from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pixel_sheriff_ml.model_factory import build_classifier_model
from pixel_sheriff_trainer.io.checkpoints import read_checkpoints
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.torchvision_cache import configure_torchvision_cache
from pixel_sheriff_trainer.utils.time import utc_now_iso

if TYPE_CHECKING:
    import torch.nn as nn


DEFAULT_INPUT_SHAPE = (3, 224, 224)


@dataclass(frozen=True)
class OnnxExportResult:
    status: str
    attempt: int
    model_uri: str | None
    metadata_uri: str
    error: str | None
    validation: dict[str, Any] | None


def _should_retry_with_legacy_onnx_export(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return (
        "guardondatadependentsymnode" in lowered
        or "torch.export" in lowered
        or "batched_nms" in lowered
        or "symbolic_shapes" in lowered
    )


def _export_with_fallback(
    model: "nn.Module",
    example_input: torch.Tensor,
    model_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> dict[str, Any]:
    export_mode = "torch_export"
    fallback_reason: str | None = None
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (example_input,),
                model_path,
                export_params=True,
                opset_version=17,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=True,
            )
    except Exception as exc:
        if not _should_retry_with_legacy_onnx_export(exc):
            raise
        fallback_reason = str(exc)
        export_mode = "legacy"
        with torch.no_grad():
            torch.onnx.export(
                model,
                (example_input,),
                model_path,
                export_params=True,
                opset_version=17,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
    return {
        "mode": export_mode,
        "fallback_reason": fallback_reason,
    }


def _input_shape_from_model(model_config: dict[str, Any]) -> tuple[int, int, int]:
    input_cfg = model_config.get("input")
    if not isinstance(input_cfg, dict):
        return DEFAULT_INPUT_SHAPE
    raw_size = input_cfg.get("input_size")
    if not isinstance(raw_size, list):
        return DEFAULT_INPUT_SHAPE
    size_values = [int(value) for value in raw_size if isinstance(value, int) and value > 0]
    if len(size_values) == 2:
        width, height = size_values
        return (3, height, width)
    if len(size_values) == 3:
        channels, height, width = size_values
        return (channels, height, width)
    return DEFAULT_INPUT_SHAPE


def _preprocess_from_model(model_config: dict[str, Any], *, input_shape: tuple[int, int, int]) -> dict[str, Any]:
    channels, height, width = input_shape
    input_cfg = model_config.get("input")
    if not isinstance(input_cfg, dict):
        input_cfg = {}
    normalization = input_cfg.get("normalization")
    if not isinstance(normalization, dict):
        normalization = {"type": "imagenet"}
    return {
        "resize_policy": "stretch",
        "resize": {"width": int(width), "height": int(height)},
        "normalization": normalization,
        "channels": int(channels),
    }


def _default_validation_error(message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "onnx_checker": {"status": "failed", "error": message},
        "onnxruntime": {"status": "failed", "error": message},
    }


def _supports_dynamic_batch_for_task(task: str | None) -> bool:
    normalized_task = str(task or "").strip().lower()
    if normalized_task == "detection":
        return False
    return True


def _build_dynamic_axes(
    input_names: list[str],
    output_names: list[str],
    *,
    enable_batch_dynamic: bool,
) -> dict[str, dict[int, str]]:
    if not enable_batch_dynamic:
        return {}
    return {name: {0: "batch_size"} for name in input_names + output_names}


def _validate_onnxruntime_batch_outputs(
    outputs: list[Any],
    *,
    batch_size: int,
    task: str | None,
) -> dict[str, Any]:
    output_shapes: list[list[int]] = []
    for output in outputs:
        out_shape = list(getattr(output, "shape", []))
        if not out_shape:
            raise ValueError("onnxruntime returned output without shape")
        output_shapes.append([int(v) for v in out_shape])
    if not output_shapes:
        raise ValueError("onnxruntime returned no outputs")

    normalized_task = str(task or "").strip().lower()
    if normalized_task != "detection" and int(output_shapes[0][0]) != batch_size:
        raise ValueError(
            f"dynamic batch validation failed: expected batch={batch_size}, got output_shape={output_shapes[0]}"
        )

    result: dict[str, Any] = {
        "status": "passed",
        "output_shapes": output_shapes,
        "output_shape": output_shapes[0],
    }
    if normalized_task == "detection":
        result["batch_semantics"] = "postprocessed_detection_outputs"
    return result


def _validate_exported_onnx(
    onnx_path: Path,
    *,
    input_shape: tuple[int, int, int],
    output_names: list[str] | None = None,
    task: str | None = None,
    batch_sizes: tuple[int, ...] = (1, 4),
) -> dict[str, Any]:
    try:
        import onnx
    except Exception as exc:
        return _default_validation_error(f"onnx import failed: {exc}")

    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
    except Exception as exc:
        return _default_validation_error(f"onnx checker failed: {exc}")

    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        return _default_validation_error(f"onnxruntime import failed: {exc}")

    validation: dict[str, Any] = {
        "status": "passed",
        "onnx_checker": {"status": "passed", "error": None},
        "onnxruntime": {"status": "passed", "error": None, "providers": [], "batch_results": {}},
    }

    try:
        providers = ort.get_available_providers()
        if providers:
            session = ort.InferenceSession(str(onnx_path), providers=providers)
        else:
            session = ort.InferenceSession(str(onnx_path))
        validation["onnxruntime"]["providers"] = list(session.get_providers())

        for batch_size in batch_sizes:
            dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)
            requested_outputs = output_names if isinstance(output_names, list) and output_names else None
            outputs = session.run(requested_outputs, {"input": dummy})
            validation["onnxruntime"]["batch_results"][str(batch_size)] = _validate_onnxruntime_batch_outputs(
                outputs,
                batch_size=batch_size,
                task=task,
            )
    except Exception as exc:
        validation["status"] = "failed"
        validation["onnxruntime"]["status"] = "failed"
        validation["onnxruntime"]["error"] = str(exc)
        return validation

    return validation


def _resolve_best_checkpoint(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
) -> tuple[str | None, Path | None]:
    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    for kind in ("best_metric", "best_loss", "latest"):
        row = next((item for item in rows if str(item.get("kind")) == kind), None)
        uri = str(row.get("uri") or "") if isinstance(row, dict) else ""
        status = str(row.get("status") or "") if isinstance(row, dict) else ""
        if not uri or (status and status != "ok"):
            continue
        try:
            checkpoint_path = storage.resolve(uri)
        except Exception:
            continue
        if checkpoint_path.exists() and checkpoint_path.is_file():
            return kind, checkpoint_path
    return None, None


def _as_relative_uri(storage: ExperimentStorage, path: Path) -> str:
    return str(path.relative_to(storage.root)).replace("\\", "/")


def export_best_classification_onnx(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    model_config: dict[str, Any],
    num_classes: int,
    class_names: list[str],
    class_order: list[str] | None = None,
) -> OnnxExportResult:
    configure_torchvision_cache(str(storage.root))
    onnx_dir = storage.run_dir(project_id, experiment_id, attempt) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model_path = onnx_dir / "model.onnx"
    metadata_path = onnx_dir / "onnx.metadata.json"

    checkpoint_kind, checkpoint_path = _resolve_best_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
    )
    if checkpoint_path is None:
        metadata_payload = {
            "schema_version": "1",
            "status": "failed",
            "attempt": int(attempt),
            "error": "No checkpoint found for ONNX export",
            "checkpoint_kind": None,
            "checkpoint_uri": None,
            "exported_at": utc_now_iso(),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
        return OnnxExportResult(
            status="failed",
            attempt=int(attempt),
            model_uri=None,
            metadata_uri=_as_relative_uri(storage, metadata_path),
            error="No checkpoint found for ONNX export",
            validation=None,
        )

    checkpoint_uri = _as_relative_uri(storage, checkpoint_path)
    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("model_state_dict"), dict):
            state_dict = checkpoint_payload["model_state_dict"]
        elif isinstance(checkpoint_payload, dict):
            state_dict = checkpoint_payload
        else:
            raise ValueError("checkpoint payload is invalid")

        model = build_classifier_model(model_config, num_classes_override=int(num_classes))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        input_shape = _input_shape_from_model(model_config)
        preprocess = _preprocess_from_model(model_config, input_shape=input_shape)
        example_input = torch.randn(1, *input_shape, dtype=torch.float32)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        export_backend = _export_with_fallback(
            model,
            example_input,
            model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        validation = _validate_exported_onnx(
            model_path,
            input_shape=input_shape,
            output_names=["output"],
            task="classification",
            batch_sizes=(1, 4),
        )
        status = str(validation.get("status") or "failed")
        error = None
        if status != "passed":
            error = str(validation.get("onnxruntime", {}).get("error") or "ONNX validation failed")
            try:
                model_path.unlink(missing_ok=True)
            except OSError:
                pass

        model_uri = _as_relative_uri(storage, model_path) if model_path.exists() else None
        metadata_payload = {
            "schema_version": "1",
            "status": "exported" if status == "passed" else "failed",
            "attempt": int(attempt),
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_uri": checkpoint_uri,
            "model_uri": model_uri,
            "input_shape": [int(v) for v in input_shape],
            "class_order": [str(value) for value in class_order] if isinstance(class_order, list) else [],
            "class_names": [str(name) for name in class_names],
            "class_ids": [str(value) for value in class_order] if isinstance(class_order, list) else [],
            "preprocess": preprocess,
            "onnx": {
                "opset_version": 17,
                "input_names": ["input"],
                "output_names": ["output"],
                "dynamic_axes": dynamic_axes,
                "export_backend": export_backend,
            },
            "validation": validation,
            "exported_at": utc_now_iso(),
            "error": error,
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
        return OnnxExportResult(
            status="exported" if status == "passed" else "failed",
            attempt=int(attempt),
            model_uri=model_uri,
            metadata_uri=_as_relative_uri(storage, metadata_path),
            error=error,
            validation=validation,
        )
    except Exception as exc:
        try:
            model_path.unlink(missing_ok=True)
        except OSError:
            pass
        metadata_payload = {
            "schema_version": "1",
            "status": "failed",
            "attempt": int(attempt),
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_uri": checkpoint_uri,
            "error": str(exc),
            "exported_at": utc_now_iso(),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
        return OnnxExportResult(
            status="failed",
            attempt=int(attempt),
            model_uri=None,
            metadata_uri=_as_relative_uri(storage, metadata_path),
            error=str(exc),
            validation=None,
        )


def export_model_to_onnx(
    model: "nn.Module",
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    checkpoint_kind: str | None,
    checkpoint_uri: str | None,
    input_shape: tuple[int, int, int],
    input_names: list[str],
    output_names: list[str],
    preprocess: dict[str, Any],
    class_order: list[str] | None,
    class_names: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> OnnxExportResult:
    """Generic ONNX export for any pre-built model.

    Callers are responsible for building and configuring the model before calling this.
    This function handles the ONNX serialization, validation, and metadata persistence.
    """
    onnx_dir = storage.run_dir(project_id, experiment_id, attempt) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model_path = onnx_dir / "model.onnx"
    metadata_path = onnx_dir / "onnx.metadata.json"

    try:
        model.eval()
        task = str(extra_metadata.get("task")) if isinstance(extra_metadata, dict) and isinstance(extra_metadata.get("task"), str) else None
        dynamic_batch_enabled = _supports_dynamic_batch_for_task(task)
        dynamic_axes = _build_dynamic_axes(
            input_names,
            output_names,
            enable_batch_dynamic=dynamic_batch_enabled,
        )
        example_input = torch.randn(1, *input_shape, dtype=torch.float32)

        export_backend = _export_with_fallback(
            model,
            example_input,
            model_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        validation = _validate_exported_onnx(
            model_path,
            input_shape=input_shape,
            output_names=output_names,
            task=task,
            batch_sizes=(1, 4) if dynamic_batch_enabled else (1,),
        )
        status = str(validation.get("status") or "failed")
        error = None
        if status != "passed":
            error = str(validation.get("onnxruntime", {}).get("error") or "ONNX validation failed")
            try:
                model_path.unlink(missing_ok=True)
            except OSError:
                pass

        model_uri = _as_relative_uri(storage, model_path) if model_path.exists() else None
        metadata_payload: dict[str, Any] = {
            "schema_version": "1",
            "status": "exported" if status == "passed" else "failed",
            "attempt": int(attempt),
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_uri": checkpoint_uri,
            "model_uri": model_uri,
            "input_shape": [int(v) for v in input_shape],
            "class_order": [str(v) for v in class_order] if isinstance(class_order, list) else [],
            "class_names": [str(n) for n in class_names],
            "class_ids": [str(v) for v in class_order] if isinstance(class_order, list) else [],
            "preprocess": preprocess,
            "onnx": {
                "opset_version": 17,
                "input_names": input_names,
                "output_names": output_names,
                "dynamic_axes": {k: dict(v) for k, v in dynamic_axes.items()},
                "export_backend": export_backend,
            },
            "validation": validation,
            "exported_at": utc_now_iso(),
            "error": error,
        }
        if task == "detection" and not dynamic_batch_enabled:
            metadata_payload["onnx"]["runtime_note"] = (
                "Dynamic batch is disabled for detection exports; torchvision detection ONNX graphs are validated for batch=1."
            )
        if extra_metadata:
            metadata_payload.update(extra_metadata)
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
        return OnnxExportResult(
            status="exported" if status == "passed" else "failed",
            attempt=int(attempt),
            model_uri=model_uri,
            metadata_uri=_as_relative_uri(storage, metadata_path),
            error=error,
            validation=validation,
        )
    except Exception as exc:
        try:
            model_path.unlink(missing_ok=True)
        except OSError:
            pass
        metadata_payload = {
            "schema_version": "1",
            "status": "failed",
            "attempt": int(attempt),
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_uri": checkpoint_uri,
            "error": str(exc),
            "exported_at": utc_now_iso(),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
        return OnnxExportResult(
            status="failed",
            attempt=int(attempt),
            model_uri=None,
            metadata_uri=_as_relative_uri(storage, metadata_path),
            error=str(exc),
            validation=None,
        )
