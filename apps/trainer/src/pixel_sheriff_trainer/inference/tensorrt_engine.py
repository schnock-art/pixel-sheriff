from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TensorRTUnavailableError(RuntimeError):
    pass


def resolve_target_device(device_preference: str) -> dict[str, Any]:
    normalized = str(device_preference or "auto").strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if normalized == "cpu" or not cuda_available:
        return {
            "device_selected": "cpu",
            "gpu_name": None,
            "cuda_version": str(torch.version.cuda or "") or None,
            "fingerprint": "cpu",
        }
    device_index = torch.cuda.current_device()
    gpu_name = str(torch.cuda.get_device_name(device_index))
    cuda_version = str(torch.version.cuda or "") or None
    return {
        "device_selected": "cuda",
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "fingerprint": _engine_fingerprint(gpu_name, cuda_version),
    }


def _engine_fingerprint(gpu_name: str, cuda_version: str | None) -> str:
    raw = f"{gpu_name}|{cuda_version or 'unknown'}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _import_tensorrt() -> Any:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised through graceful failure callers
        raise TensorRTUnavailableError(f"TensorRT is not available: {exc}") from exc
    return trt


def _logger(trt: Any) -> Any:
    return trt.Logger(trt.Logger.WARNING)


def engine_key_for(model_key: str, target_device_fingerprint: str) -> str:
    digest = hashlib.sha256(f"{model_key}:{target_device_fingerprint}".encode("utf-8")).hexdigest()
    return digest


def ensure_detection_engine(
    storage_root: Path,
    *,
    onnx_path: Path,
    model_key: str,
    target_device: dict[str, Any],
    precision: str = "int8",
) -> tuple[Path, Path]:
    if str(target_device.get("device_selected") or "") != "cuda":
        raise TensorRTUnavailableError("TensorRT detection requires CUDA")

    fingerprint = str(target_device.get("fingerprint") or "").strip()
    if not fingerprint:
        raise TensorRTUnavailableError("Target device fingerprint is missing")

    engine_key = engine_key_for(model_key, fingerprint)
    engine_dir = storage_root / "inference_engines" / "tensorrt" / model_key / fingerprint
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{precision}-{engine_key[:12]}.plan"
    metadata_path = engine_dir / f"{precision}-{engine_key[:12]}.json"
    if engine_path.exists() and metadata_path.exists():
        return engine_path, metadata_path

    trt = _import_tensorrt()
    logger = _logger(trt)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors: list[str] = []
        try:
            error_count = int(parser.num_errors)
        except Exception:
            error_count = 0
        for index in range(error_count):
            try:
                errors.append(str(parser.get_error(index)))
            except Exception:
                continue
        detail = "; ".join(errors) if errors else "unknown TensorRT parser failure"
        raise TensorRTUnavailableError(f"TensorRT ONNX parse failed: {detail}")

    config = builder.create_builder_config()
    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = 1 << 30

    if getattr(builder, "platform_has_fast_fp16", False):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8" and getattr(builder, "platform_has_fast_int8", False):
        config.set_flag(trt.BuilderFlag.INT8)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise TensorRTUnavailableError("TensorRT engine build failed")

    engine_path.write_bytes(bytes(serialized))
    metadata_path.write_text(
        json.dumps(
            {
                "runtime_backend": "tensorrt",
                "precision": precision,
                "model_key": model_key,
                "target_device": target_device,
                "source_onnx_relpath": str(onnx_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return engine_path, metadata_path


def load_engine_bundle(engine_path: Path) -> dict[str, Any]:
    trt = _import_tensorrt()
    logger = _logger(trt)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise TensorRTUnavailableError("TensorRT engine load failed")
    context = engine.create_execution_context()
    if context is None:
        raise TensorRTUnavailableError("TensorRT execution context creation failed")
    return {"logger": logger, "runtime": runtime, "engine": engine, "context": context}


def _torch_dtype_for_trt(dtype: Any, trt: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    resolved = mapping.get(dtype)
    if resolved is None:
        raise TensorRTUnavailableError(f"Unsupported TensorRT dtype: {dtype}")
    return resolved


def run_detection_bundle(bundle: dict[str, Any], tensor: np.ndarray) -> list[np.ndarray]:
    trt = _import_tensorrt()
    engine = bundle["engine"]
    context = bundle["context"]
    if not torch.cuda.is_available():
        raise TensorRTUnavailableError("CUDA is required for TensorRT inference")

    device_tensor = torch.as_tensor(tensor, device="cuda")
    output_tensors: list[torch.Tensor] = []
    output_names: list[str] = []
    input_name: str | None = None
    tensor_names = [engine.get_tensor_name(index) for index in range(int(engine.num_io_tensors))]
    for name in tensor_names:
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
            context.set_input_shape(name, tuple(int(value) for value in device_tensor.shape))
            context.set_tensor_address(name, int(device_tensor.data_ptr()))
            continue
        shape = tuple(int(value) for value in context.get_tensor_shape(name))
        dtype = _torch_dtype_for_trt(engine.get_tensor_dtype(name), trt)
        output = torch.empty(shape, device="cuda", dtype=dtype)
        context.set_tensor_address(name, int(output.data_ptr()))
        output_tensors.append(output)
        output_names.append(name)

    if not input_name:
        raise TensorRTUnavailableError("TensorRT engine is missing an input tensor")

    stream = torch.cuda.current_stream()
    success = context.execute_async_v3(stream.cuda_stream)
    if not success:
        raise TensorRTUnavailableError("TensorRT execution failed")
    stream.synchronize()

    return [output.detach().cpu().numpy() for output in output_tensors]
