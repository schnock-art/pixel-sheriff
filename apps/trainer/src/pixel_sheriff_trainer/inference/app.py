from __future__ import annotations

import asyncio
import gc
import math
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
import torch

from .preprocess import load_metadata, preprocess_asset
from .schemas import (
    DetectionBox,
    FlorenceDetectRequest,
    FlorenceDetectResponse,
    FlorenceDetectionBox,
    FlorenceWarmupRequest,
    InferClassificationRequest,
    InferClassificationResponse,
    InferClassificationWarmupRequest,
    InferDetectionRequest,
    InferDetectionResponse,
    InferDetectionWarmupRequest,
    InferSegmentationRequest,
    InferSegmentationResponse,
    InferWarmupResponse,
    PredictionRow,
    SegmentationObject,
)
from .session_cache import CacheBusyError, SessionCache, sha256_file


_FLORENCE_CACHE: dict[str, tuple[object, object, str]] = {}
_FLORENCE_CACHE_LOCK = threading.Lock()


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


def _resolve_asset_path(storage_root: Path, *, asset_relpath: str) -> Path:
    asset_path = (storage_root / asset_relpath).resolve()
    if not asset_path.is_relative_to(storage_root.resolve()):
        raise ValueError("path escapes storage root")
    return asset_path


def _run_onnx(session: object, tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    first = outputs[0]
    return np.asarray(first, dtype=np.float32)


async def _warmup_session(
    *,
    cache: SessionCache,
    onnx_path: Path,
    metadata_path: Path,
    device_preference: str,
    model_key: str | None,
) -> InferWarmupResponse:
    if not onnx_path.exists() or not metadata_path.exists():
        raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "Inference artifacts not found"})

    resolved_model_key = model_key or await asyncio.to_thread(sha256_file, onnx_path)
    onnx_hash = await asyncio.to_thread(sha256_file, onnx_path)
    if onnx_hash != resolved_model_key:
        raise HTTPException(
            status_code=409,
            detail={"code": "model_key_mismatch", "message": "Provided model_key does not match ONNX content"},
        )

    try:
        _session, device_selected = await cache.acquire_session(
            model_key=resolved_model_key,
            onnx_path=onnx_path,
            device_preference=device_preference,
        )
    except CacheBusyError as exc:
        raise HTTPException(status_code=503, detail={"code": "cache_busy", "message": "Inference cache is busy"}) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "session_load_failed", "message": str(exc)}) from exc
    await cache.release(resolved_model_key, device_selected)

    return InferWarmupResponse(device_selected=device_selected, warmed=True)


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
        return await _warmup_session(
            cache=cache,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            device_preference=payload.device_preference,
            model_key=payload.model_key,
        )

    @app.post("/infer/detection", response_model=InferDetectionResponse)
    async def infer_detection(payload: InferDetectionRequest) -> InferDetectionResponse:
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
        class_names: list[str] = metadata.get("class_names", [])
        model_key = payload.model_key or await asyncio.to_thread(sha256_file, onnx_path)
        onnx_hash = await asyncio.to_thread(sha256_file, onnx_path)
        if onnx_hash != model_key:
            raise HTTPException(status_code=409, detail={"code": "model_key_mismatch", "message": "Provided model_key does not match ONNX content"})

        try:
            session, device_selected = await cache.acquire_session(
                model_key=model_key, onnx_path=onnx_path, device_preference=payload.device_preference,
            )
        except CacheBusyError as exc:
            raise HTTPException(status_code=503, detail={"code": "cache_busy", "message": "Inference cache is busy"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "session_load_failed", "message": str(exc)}) from exc

        try:
            tensor = await asyncio.to_thread(preprocess_asset, asset_path, metadata)
            raw_outputs = await asyncio.to_thread(_run_onnx_detection, session, tensor)
            boxes = _parse_detection_output(
                raw_outputs, class_names=class_names, score_threshold=payload.score_threshold,
            )
            return InferDetectionResponse(device_selected=device_selected, boxes=boxes)
        finally:
            await cache.release(model_key, device_selected)

    @app.post("/infer/detection/warmup", response_model=InferWarmupResponse)
    async def warmup_detection(payload: InferDetectionWarmupRequest) -> InferWarmupResponse:
        try:
            onnx_path, metadata_path, _asset_path = _resolve_paths(
                storage_root,
                onnx_relpath=payload.onnx_relpath,
                metadata_relpath=payload.metadata_relpath,
                asset_relpath=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "path_invalid", "message": str(exc)}) from exc
        return await _warmup_session(
            cache=cache,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            device_preference=payload.device_preference,
            model_key=payload.model_key,
        )

    @app.post("/infer/segmentation", response_model=InferSegmentationResponse)
    async def infer_segmentation(payload: InferSegmentationRequest) -> InferSegmentationResponse:
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
        class_names = metadata.get("class_names", [])
        model_key = payload.model_key or await asyncio.to_thread(sha256_file, onnx_path)
        onnx_hash = await asyncio.to_thread(sha256_file, onnx_path)
        if onnx_hash != model_key:
            raise HTTPException(status_code=409, detail={"code": "model_key_mismatch", "message": "Provided model_key does not match ONNX content"})

        try:
            session, device_selected = await cache.acquire_session(
                model_key=model_key, onnx_path=onnx_path, device_preference=payload.device_preference,
            )
        except CacheBusyError as exc:
            raise HTTPException(status_code=503, detail={"code": "cache_busy", "message": "Inference cache is busy"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "session_load_failed", "message": str(exc)}) from exc

        try:
            tensor = await asyncio.to_thread(preprocess_asset, asset_path, metadata)
            logits = await asyncio.to_thread(_run_onnx, session, tensor)
            objects = _parse_segmentation_output(logits, class_names=class_names)
            return InferSegmentationResponse(device_selected=device_selected, objects=objects)
        finally:
            await cache.release(model_key, device_selected)

    @app.post("/infer/florence/warmup", response_model=InferWarmupResponse)
    async def warmup_florence(payload: FlorenceWarmupRequest) -> InferWarmupResponse:
        try:
            _model, _processor, device_selected = await asyncio.to_thread(_load_florence_runtime, payload.model_name)
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "florence_load_failed", "message": str(exc)}) from exc
        return InferWarmupResponse(device_selected=device_selected, warmed=True)

    @app.post("/infer/florence/detect", response_model=FlorenceDetectResponse)
    async def florence_detect(payload: FlorenceDetectRequest) -> FlorenceDetectResponse:
        try:
            asset_path = _resolve_asset_path(storage_root, asset_relpath=payload.asset_relpath)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "path_invalid", "message": str(exc)}) from exc
        if not asset_path.exists():
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "Asset not found"})
        try:
            model, processor, device_selected = await asyncio.to_thread(_load_florence_runtime, payload.model_name)
            boxes = await asyncio.to_thread(
                _run_florence_detection,
                model,
                processor,
                device_selected,
                asset_path,
                payload.prompts,
                payload.score_threshold,
                payload.max_detections,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "florence_inference_failed", "message": str(exc)}) from exc
        return FlorenceDetectResponse(device_selected=device_selected, boxes=boxes)

    return app


def _run_onnx_detection(session: object, tensor: np.ndarray) -> list[np.ndarray]:
    """Run detection model; returns list of output arrays."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    return [np.asarray(o, dtype=np.float32) for o in outputs]


def _parse_detection_output(
    raw_outputs: list[np.ndarray],
    *,
    class_names: list[str],
    score_threshold: float,
) -> list[DetectionBox]:
    """Parse ONNX detection output into DetectionBox list.

    Expected output format (ONNX export of RetinaNet):
    - Single output tensor of shape [N, 6]: [x_min, y_min, x_max, y_max, score, class_id]
    OR torchvision-style separate outputs for boxes/scores/labels.
    """
    if not raw_outputs:
        return []
    boxes: list[DetectionBox] = []
    first = raw_outputs[0]
    if first.ndim == 2 and first.shape[-1] >= 6:
        for row in first:
            x_min, y_min, x_max, y_max, score, cls = (
                float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5])
            )
            if score < score_threshold:
                continue
            name = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
            boxes.append(DetectionBox(
                class_index=cls, class_name=name, score=score,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
            ))
    return sorted(boxes, key=lambda b: -b.score)


def _parse_segmentation_output(
    logits: np.ndarray,
    *,
    class_names: list[str],
) -> list[SegmentationObject]:
    """Parse segmentation logits into polygon objects.

    Takes argmax over class dim → pixel class map.
    Extracts simple bounding-box polygon per unique class.
    For production, connected-components contour extraction would be more precise.
    """
    if logits.ndim < 3:
        return []

    # logits shape: (1, C, H, W) or (C, H, W)
    if logits.ndim == 4:
        class_map = logits[0].argmax(axis=0)  # (H, W)
    else:
        class_map = logits.argmax(axis=0)

    objects: list[SegmentationObject] = []
    unique_classes = np.unique(class_map)
    for cls_idx in unique_classes:
        if cls_idx == 0:
            continue  # skip background
        fg_idx = int(cls_idx) - 1  # 0-indexed foreground class
        name = class_names[fg_idx] if 0 <= fg_idx < len(class_names) else f"class_{fg_idx}"
        ys, xs = np.where(class_map == cls_idx)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        polygon = [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)],
        ]
        pixel_count = int(len(xs))
        total_pixels = int(class_map.size)
        score = float(pixel_count / max(total_pixels, 1))
        objects.append(SegmentationObject(
            class_index=fg_idx, class_name=name, score=score, polygon=polygon,
        ))
    return objects


def _is_meta_tensor_load_error(exc: Exception) -> bool:
    return "meta tensor" in str(exc).lower()


def _create_florence_runtime(*, model_name: str, low_cpu_mem_usage: bool | None) -> tuple[object, object, str]:
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        raise RuntimeError("transformers is not available for Florence-2 inference") from exc

    device_selected = "cuda" if torch.cuda.is_available() else "cpu"
    # Florence-2 remote code expects eager attention during init on current transformers releases.
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if low_cpu_mem_usage is not None:
        model_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = model.to(device_selected)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor, device_selected


def _load_florence_runtime(model_name: str) -> tuple[object, object, str]:
    normalized_name = str(model_name or "microsoft/Florence-2-base-ft").strip() or "microsoft/Florence-2-base-ft"
    cached = _FLORENCE_CACHE.get(normalized_name)
    if cached is not None:
        return cached

    with _FLORENCE_CACHE_LOCK:
        cached = _FLORENCE_CACHE.get(normalized_name)
        if cached is not None:
            return cached
        try:
            cached = _create_florence_runtime(model_name=normalized_name, low_cpu_mem_usage=None)
        except Exception as exc:
            if not _is_meta_tensor_load_error(exc):
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cached = _create_florence_runtime(model_name=normalized_name, low_cpu_mem_usage=False)
        _FLORENCE_CACHE[normalized_name] = cached
        return cached


def _normalize_florence_prompt_text(prompts: list[str]) -> tuple[str, str]:
    normalized_prompts = [" ".join(str(value or "").strip().split()) for value in prompts]
    normalized_prompts = [value for value in normalized_prompts if value]
    task = "<OPEN_VOCABULARY_DETECTION>"
    prompt_text = ", ".join(normalized_prompts) if normalized_prompts else "object"
    return task, f"{task} {prompt_text}"


def _parse_florence_generation(
    processor: object,
    *,
    generated_text: str,
    task: str,
    image_size: tuple[int, int],
    score_threshold: float,
    max_detections: int,
) -> list[FlorenceDetectionBox]:
    post_process = getattr(processor, "post_process_generation", None)
    if not callable(post_process):
        raise RuntimeError("Florence processor does not support post_process_generation")
    parsed = post_process(generated_text, task=task, image_size=image_size)
    if isinstance(parsed, dict) and len(parsed) == 1:
        only_value = next(iter(parsed.values()))
        if isinstance(only_value, dict):
            parsed = only_value
    if not isinstance(parsed, dict):
        return []

    raw_boxes = parsed.get("bboxes") or parsed.get("boxes") or []
    raw_labels = parsed.get("labels") or parsed.get("label_texts") or []
    raw_scores = parsed.get("scores") or []
    detections: list[FlorenceDetectionBox] = []
    for index, raw_box in enumerate(raw_boxes):
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue
        if not all(isinstance(value, (int, float)) for value in raw_box):
            continue
        x1, y1, x2, y2 = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
        label_text = str(raw_labels[index] if index < len(raw_labels) else "").strip()
        if not label_text:
            continue
        score = float(raw_scores[index]) if index < len(raw_scores) and isinstance(raw_scores[index], (int, float)) else 1.0
        if (
            not math.isfinite(score)
            or not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or (max(x1, x2) - min(x1, x2)) <= 0
            or (max(y1, y2) - min(y1, y2)) <= 0
            or score < score_threshold
        ):
            continue
        detections.append(
            FlorenceDetectionBox(
                label_text=label_text,
                score=score,
                bbox=[x1, y1, x2, y2],
            )
        )
        if len(detections) >= max_detections:
            break
    return detections


def _run_florence_detection(
    model: object,
    processor: object,
    device_selected: str,
    asset_path: Path,
    prompts: list[str],
    score_threshold: float,
    max_detections: int,
) -> list[FlorenceDetectionBox]:
    with Image.open(asset_path) as image:
        rgb_image = image.convert("RGB")
        task, prompt_text = _normalize_florence_prompt_text(prompts)
        processor_inputs = processor(text=prompt_text, images=rgb_image, return_tensors="pt")
        if hasattr(processor_inputs, "to"):
            processor_inputs = processor_inputs.to(device_selected)
        generated_ids = model.generate(
            input_ids=processor_inputs["input_ids"],
            pixel_values=processor_inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
            use_cache=False,
        )
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)
        generated_text = decoded[0] if isinstance(decoded, list) and decoded else ""
        return _parse_florence_generation(
            processor,
            generated_text=generated_text,
            task=task,
            image_size=(rgb_image.width, rgb_image.height),
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
