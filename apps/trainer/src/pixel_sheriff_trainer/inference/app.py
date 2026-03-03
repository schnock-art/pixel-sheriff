from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .preprocess import load_metadata, preprocess_asset
from .schemas import (
    InferClassificationRequest,
    InferClassificationResponse,
    InferClassificationWarmupRequest,
    InferWarmupResponse,
    PredictionRow,
)
from .session_cache import CacheBusyError, SessionCache, sha256_file


def _top_k_predictions(logits: np.ndarray, top_k: int) -> tuple[list[PredictionRow], int]:
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("invalid logits shape")
    row = logits[0]
    output_dim = int(row.shape[0])
    max_logit = float(np.max(row))
    probs = np.exp(row - max_logit)
    probs = probs / np.sum(probs)
    indexed = [(idx, float(score)) for idx, score in enumerate(probs.tolist())]
    indexed.sort(key=lambda item: (-item[1], item[0]))
    return [PredictionRow(class_index=idx, score=score) for idx, score in indexed[:top_k]], output_dim


def _resolve_paths(storage_root: Path, *, onnx_relpath: str, metadata_relpath: str, asset_relpath: str | None = None) -> tuple[Path, Path, Path | None]:
    onnx_path = (storage_root / onnx_relpath).resolve()
    metadata_path = (storage_root / metadata_relpath).resolve()
    asset_path = (storage_root / asset_relpath).resolve() if isinstance(asset_relpath, str) else None
    if not onnx_path.is_relative_to(storage_root.resolve()) or not metadata_path.is_relative_to(storage_root.resolve()):
        raise ValueError("path escapes storage root")
    if asset_path is not None and not asset_path.is_relative_to(storage_root.resolve()):
        raise ValueError("path escapes storage root")
    return onnx_path, metadata_path, asset_path


def _run_onnx(session: object, tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    first = outputs[0]
    return np.asarray(first, dtype=np.float32)


def create_app() -> FastAPI:
    storage_root = Path(os.getenv("STORAGE_ROOT", "/app/data"))
    cache = SessionCache(
        max_models_gpu=int(os.getenv("INFERENCE_CACHE_MAX_MODELS_GPU", "1")),
        max_models_cpu=int(os.getenv("INFERENCE_CACHE_MAX_MODELS_CPU", "3")),
        ttl_seconds=int(os.getenv("INFERENCE_CACHE_TTL_SECONDS", "600")),
    )
    app = FastAPI(title="pixel-sheriff-trainer-inference", version="0.1.0")

    @app.post("/infer/classification", response_model=InferClassificationResponse)
    async def infer_classification(payload: InferClassificationRequest) -> InferClassificationResponse:
        try:
            onnx_path, metadata_path, asset_path = _resolve_paths(
                storage_root,
                onnx_relpath=payload.onnx_relpath,
                metadata_relpath=payload.metadata_relpath,
                asset_relpath=payload.asset_relpath,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "path_invalid", "message": str(exc)}) from exc

        if not onnx_path.exists() or not metadata_path.exists() or asset_path is None or not asset_path.exists():
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "Inference artifacts not found"})

        metadata = await asyncio.to_thread(load_metadata, metadata_path)
        model_key = payload.model_key or await asyncio.to_thread(sha256_file, onnx_path)
        # Safety check for aliasing/mismatch across paths.
        onnx_hash = await asyncio.to_thread(sha256_file, onnx_path)
        if onnx_hash != model_key:
            raise HTTPException(
                status_code=409,
                detail={"code": "model_key_mismatch", "message": "Provided model_key does not match ONNX content"},
            )

        try:
            session, device_selected = await cache.acquire_session(
                model_key=model_key,
                onnx_path=onnx_path,
                device_preference=payload.device_preference,
            )
        except CacheBusyError as exc:
            raise HTTPException(status_code=503, detail={"code": "cache_busy", "message": "Inference cache is busy"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "session_load_failed", "message": str(exc)}) from exc

        try:
            tensor = await asyncio.to_thread(preprocess_asset, asset_path, metadata)
            logits = await asyncio.to_thread(_run_onnx, session, tensor)
            rows, output_dim = _top_k_predictions(logits, payload.top_k)
            return InferClassificationResponse(
                device_selected=device_selected,
                predictions=rows,
                output_dim=output_dim,
            )
        finally:
            await cache.release(model_key, device_selected)

    @app.post("/infer/classification/warmup", response_model=InferWarmupResponse)
    async def warmup_classification(payload: InferClassificationWarmupRequest) -> InferWarmupResponse:
        try:
            onnx_path, metadata_path, _asset_path = _resolve_paths(
                storage_root,
                onnx_relpath=payload.onnx_relpath,
                metadata_relpath=payload.metadata_relpath,
                asset_relpath=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "path_invalid", "message": str(exc)}) from exc
        if not onnx_path.exists() or not metadata_path.exists():
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "Inference artifacts not found"})

        model_key = payload.model_key or await asyncio.to_thread(sha256_file, onnx_path)
        onnx_hash = await asyncio.to_thread(sha256_file, onnx_path)
        if onnx_hash != model_key:
            raise HTTPException(
                status_code=409,
                detail={"code": "model_key_mismatch", "message": "Provided model_key does not match ONNX content"},
            )

        try:
            _session, device_selected = await cache.acquire_session(
                model_key=model_key,
                onnx_path=onnx_path,
                device_preference=payload.device_preference,
            )
        except CacheBusyError as exc:
            raise HTTPException(status_code=503, detail={"code": "cache_busy", "message": "Inference cache is busy"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "session_load_failed", "message": str(exc)}) from exc
        await cache.release(model_key, device_selected)

        return InferWarmupResponse(device_selected=device_selected, warmed=True)

    return app
