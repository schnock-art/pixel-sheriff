from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pixel_sheriff_ml.model_factory import build_resnet_classifier
from pixel_sheriff_trainer.io.checkpoints import read_checkpoints
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.time import utc_now_iso


DEFAULT_INPUT_SHAPE = (3, 224, 224)


@dataclass(frozen=True)
class OnnxExportResult:
    status: str
    attempt: int
    model_uri: str | None
    metadata_uri: str
    error: str | None
    validation: dict[str, Any] | None


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


def _validate_exported_onnx(onnx_path: Path, *, input_shape: tuple[int, int, int]) -> dict[str, Any]:
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

        for batch_size in (1, 4):
            dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)
            outputs = session.run(["output"], {"input": dummy})
            if not outputs:
                raise ValueError("onnxruntime returned no outputs")
            output = outputs[0]
            out_shape = list(getattr(output, "shape", []))
            if not out_shape:
                raise ValueError("onnxruntime returned output without shape")
            if int(out_shape[0]) != batch_size:
                raise ValueError(
                    f"dynamic batch validation failed: expected batch={batch_size}, got output_shape={out_shape}"
                )
            validation["onnxruntime"]["batch_results"][str(batch_size)] = {
                "status": "passed",
                "output_shape": [int(v) for v in out_shape],
            }
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
    class_order: list[int] | None = None,
) -> OnnxExportResult:
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

        model = build_resnet_classifier(model_config, num_classes_override=int(num_classes))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        input_shape = _input_shape_from_model(model_config)
        preprocess = _preprocess_from_model(model_config, input_shape=input_shape)
        example_input = torch.randn(1, *input_shape, dtype=torch.float32)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        with torch.no_grad():
            torch.onnx.export(
                model,
                (example_input,),
                model_path,
                export_params=True,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

        validation = _validate_exported_onnx(model_path, input_shape=input_shape)
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
            "class_order": [str(name) for name in class_names],
            "class_names": [str(name) for name in class_names],
            "class_ids": [int(value) for value in class_order] if isinstance(class_order, list) else [],
            "preprocess": preprocess,
            "onnx": {
                "opset_version": 17,
                "input_names": ["input"],
                "output_names": ["output"],
                "dynamic_axes": dynamic_axes,
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
