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

from .preprocess import (
    PreprocessContext,
    load_metadata,
    preprocess_asset,
    preprocess_asset_with_context,
    remap_bbox_xyxy_to_original_xywh,
)
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
            tensor, preprocess_context = await asyncio.to_thread(preprocess_asset_with_context, asset_path, metadata)
            raw_outputs = await asyncio.to_thread(_run_onnx_detection, session, tensor)
            boxes = _parse_detection_output(
                raw_outputs,
                class_names=class_names,
                score_threshold=payload.score_threshold,
                preprocess_context=preprocess_context,
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
    return [np.asarray(output) for output in outputs]


def _is_integral_array(values: np.ndarray) -> bool:
    if np.issubdtype(values.dtype, np.integer):
        return True
    if not np.issubdtype(values.dtype, np.floating):
        return False
    if values.size == 0:
        return True
    return bool(np.all(np.isfinite(values)) and np.all(np.equal(values, np.round(values))))


def _normalize_detection_labels(
    raw_labels: np.ndarray,
    *,
    num_classes: int,
    prefer_one_based: bool,
) -> list[int | None]:
    normalized_raw: list[int | None] = []
    for raw_label in raw_labels.tolist():
        if isinstance(raw_label, bool) or not isinstance(raw_label, (int, float)):
            normalized_raw.append(None)
            continue
        label = float(raw_label)
        if not math.isfinite(label) or not label.is_integer():
            normalized_raw.append(None)
            continue
        normalized_raw.append(int(label))

    if num_classes <= 0:
        return [value if value is not None and value >= 0 else None for value in normalized_raw]

    matches_zero_based = sum(1 for value in normalized_raw if value is not None and 0 <= value < num_classes)
    matches_one_based = sum(1 for value in normalized_raw if value is not None and 1 <= value <= num_classes)

    if matches_zero_based == 0 and matches_one_based == 0:
        return [None for _value in normalized_raw]
    if matches_zero_based == matches_one_based:
        label_offset = 1 if prefer_one_based else 0
    elif matches_one_based > matches_zero_based:
        label_offset = 1
    else:
        label_offset = 0

    normalized: list[int | None] = []
    for value in normalized_raw:
        if value is None:
            normalized.append(None)
            continue
        class_index = value - label_offset
        if class_index < 0 or class_index >= num_classes:
            normalized.append(None)
            continue
        normalized.append(class_index)
    return normalized


def _detection_box_from_xyxy(
    *,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    class_index: int | None,
    score: Any,
    class_names: list[str],
    score_threshold: float,
    preprocess_context: PreprocessContext | None,
) -> DetectionBox | None:
    if isinstance(score, np.generic):
        score = score.item()
    if not isinstance(score, (int, float)):
        return None
    score_value = float(score)
    if not math.isfinite(score_value) or score_value < score_threshold:
        return None

    if len(bbox_xyxy) != 4:
        return None
    try:
        x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        return None

    if class_index is None:
        return None

    if preprocess_context is not None:
        bbox = remap_bbox_xyxy_to_original_xywh([x_min, y_min, x_max, y_max], preprocess_context)
    else:
        if x_max <= x_min or y_max <= y_min:
            return None
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    if bbox is None:
        return None

    class_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_index}"
    return DetectionBox(class_index=class_index, class_name=class_name, score=score_value, bbox=bbox)


def _parse_combined_detection_output(
    raw_outputs: list[np.ndarray],
    *,
    class_names: list[str],
    score_threshold: float,
    preprocess_context: PreprocessContext | None,
) -> list[DetectionBox] | None:
    if not raw_outputs:
        return None
    first = np.asarray(raw_outputs[0])
    if first.ndim != 2 or first.shape[-1] < 6:
        return None

    class_indexes = _normalize_detection_labels(
        np.asarray(first[:, 5]),
        num_classes=len(class_names),
        prefer_one_based=False,
    )
    boxes: list[DetectionBox] = []
    for row, class_index in zip(first, class_indexes, strict=False):
        box = _detection_box_from_xyxy(
            bbox_xyxy=[row[0], row[1], row[2], row[3]],
            class_index=class_index,
            score=row[4],
            class_names=class_names,
            score_threshold=score_threshold,
            preprocess_context=preprocess_context,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def _match_separate_detection_outputs(raw_outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    arrays = [np.asarray(output) for output in raw_outputs]
    boxes = next((array for array in arrays if array.ndim == 2 and array.shape[-1] == 4), None)
    if boxes is None:
        return None

    candidates = [array for array in arrays if array.ndim == 1 and array.shape[0] == boxes.shape[0]]
    if len(candidates) < 2:
        return None

    labels_index = next((index for index, array in enumerate(candidates) if _is_integral_array(array)), None)
    if labels_index is None:
        return None
    labels = candidates[labels_index]
    scores = next((array for index, array in enumerate(candidates) if index != labels_index), None)
    if scores is None:
        return None

    return boxes, scores.astype(np.float32, copy=False), labels


def _parse_separate_detection_output(
    raw_outputs: list[np.ndarray],
    *,
    class_names: list[str],
    score_threshold: float,
    preprocess_context: PreprocessContext | None,
) -> list[DetectionBox] | None:
    matched = _match_separate_detection_outputs(raw_outputs)
    if matched is None:
        return None
    raw_boxes, raw_scores, raw_labels = matched

    class_indexes = _normalize_detection_labels(
        np.asarray(raw_labels),
        num_classes=len(class_names),
        prefer_one_based=True,
    )
    boxes: list[DetectionBox] = []
    for bbox_row, score, class_index in zip(raw_boxes, raw_scores, class_indexes, strict=False):
        box = _detection_box_from_xyxy(
            bbox_xyxy=[bbox_row[0], bbox_row[1], bbox_row[2], bbox_row[3]],
            class_index=class_index,
            score=score,
            class_names=class_names,
            score_threshold=score_threshold,
            preprocess_context=preprocess_context,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def _parse_detection_output(
    raw_outputs: list[np.ndarray],
    *,
    class_names: list[str],
    score_threshold: float,
    preprocess_context: PreprocessContext | None = None,
) -> list[DetectionBox]:
    """Parse ONNX detection output into DetectionBox list.

    Expected output format (ONNX export of RetinaNet):
    - Single output tensor of shape [N, 6]: [x_min, y_min, x_max, y_max, score, class_id]
    OR torchvision-style separate outputs for boxes/scores/labels.
    """
    if not raw_outputs:
        return []
    boxes = _parse_combined_detection_output(
        raw_outputs,
        class_names=class_names,
        score_threshold=score_threshold,
        preprocess_context=preprocess_context,
    )
    if boxes is None:
        boxes = _parse_separate_detection_output(
            raw_outputs,
            class_names=class_names,
            score_threshold=score_threshold,
            preprocess_context=preprocess_context,
        )
    if boxes is None:
        return []
    return sorted(boxes, key=lambda box: (-box.score, box.class_index))


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
