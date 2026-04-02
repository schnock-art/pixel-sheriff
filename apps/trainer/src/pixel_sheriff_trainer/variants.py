from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Iterable
import zipfile

import numpy as np
import torch

from pixel_sheriff_ml.model_factory import build_classifier_model
from pixel_sheriff_trainer.classification.dataset import _asset_label_samples
from pixel_sheriff_trainer.classification.eval import ClassMetricsRow, ClassifierEvaluation, PredictionRow
from pixel_sheriff_trainer.classification.train import run_training
from pixel_sheriff_trainer.detection.evaluation import DetectionGroundTruth, DetectionPrediction, evaluate_detection_set
from pixel_sheriff_trainer.detection.evaluation.boxes import bbox_xywh_to_xyxy
from pixel_sheriff_trainer.detection.train import (
    DetectionEpochMetrics,
    _build_detection_model,
    run_detection_training,
)
from pixel_sheriff_trainer.export_onnx import (
    _as_relative_uri,
    _input_shape_from_model,
    _preprocess_from_model,
    _resolve_best_checkpoint,
    export_model_to_onnx,
)
from pixel_sheriff_trainer.inference.app import _parse_detection_output, _run_onnx, _run_onnx_detection
from pixel_sheriff_trainer.inference.preprocess import load_metadata, preprocess_asset, preprocess_asset_with_context
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.pipeline import PIPELINE_REGISTRY
from pixel_sheriff_trainer.training_config import resolve_device
from pixel_sheriff_trainer.utils.time import utc_now_iso


VARIANT_FP32 = "fp32"
VARIANT_FP16 = "fp16"
VARIANT_PTQ_INT8 = "ptq_int8"
VARIANT_QAT_INT8 = "qat_int8"
VARIANT_PREFERRED_ORDER = (VARIANT_QAT_INT8, VARIANT_PTQ_INT8, VARIANT_FP16, VARIANT_FP32)
VARIANT_LABELS = {
    VARIANT_FP32: "FP32",
    VARIANT_FP16: "FP16",
    VARIANT_PTQ_INT8: "PTQ INT8",
    VARIANT_QAT_INT8: "QAT INT8",
}
VARIANT_KINDS = {
    VARIANT_FP32: "baseline",
    VARIANT_FP16: "fp16",
    VARIANT_PTQ_INT8: "ptq",
    VARIANT_QAT_INT8: "qat",
}
VARIANT_STATUSES = {"queued", "running", "ready", "failed", "unsupported"}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _variant_default(variant_key: str, *, attempt: int) -> dict[str, Any]:
    return {
        "variant_key": variant_key,
        "label": VARIANT_LABELS.get(variant_key, variant_key),
        "kind": VARIANT_KINDS.get(variant_key, "baseline"),
        "attempt": int(attempt),
        "status": "queued",
        "error": None,
        "preferred": False,
        "updated_at": utc_now_iso(),
    }


def _variant_index_default(*, attempt: int) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "attempt": int(attempt),
        "preferred_variant_key": VARIANT_FP32,
        "variants": {},
        "updated_at": utc_now_iso(),
    }


def _preferred_variant_key(variants: dict[str, Any]) -> str:
    for variant_key in VARIANT_PREFERRED_ORDER:
        row = variants.get(variant_key)
        if isinstance(row, dict) and str(row.get("status") or "") == "ready":
            return variant_key
    if VARIANT_FP32 in variants:
        return VARIANT_FP32
    return next(iter(variants.keys()), VARIANT_FP32)


def _update_variant_row(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    variant_key: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    index_path = storage.variants_index_path(project_id, experiment_id, attempt)
    index_payload = _read_json(index_path, _variant_index_default(attempt=attempt))
    if not isinstance(index_payload, dict):
        index_payload = _variant_index_default(attempt=attempt)
    variants = index_payload.get("variants")
    if not isinstance(variants, dict):
        variants = {}

    current = _read_json(storage.variant_status_path(project_id, experiment_id, attempt, variant_key), _variant_default(variant_key, attempt=attempt))
    if not isinstance(current, dict):
        current = _variant_default(variant_key, attempt=attempt)
    merged = _merge_dict(current, patch)
    status = str(merged.get("status") or "").strip().lower()
    if status not in VARIANT_STATUSES:
        merged["status"] = "failed"
    merged["attempt"] = int(attempt)
    merged["variant_key"] = variant_key
    merged["label"] = VARIANT_LABELS.get(variant_key, variant_key)
    merged["kind"] = VARIANT_KINDS.get(variant_key, merged.get("kind") or "baseline")
    merged["updated_at"] = utc_now_iso()

    variants[variant_key] = dict(merged)
    preferred_variant_key = _preferred_variant_key(variants)
    for key, row in list(variants.items()):
        if not isinstance(row, dict):
            continue
        row["preferred"] = key == preferred_variant_key
    index_payload["variants"] = variants
    index_payload["preferred_variant_key"] = preferred_variant_key
    index_payload["updated_at"] = utc_now_iso()

    _write_json(storage.variant_status_path(project_id, experiment_id, attempt, variant_key), merged)
    _write_json(index_path, index_payload)
    return variants[variant_key]


def _variant_event(
    variant_key: str,
    status: str,
    *,
    attempt: int,
    message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "variant_status",
        "variant_key": variant_key,
        "status": status,
        "attempt": int(attempt),
        "ts": utc_now_iso(),
    }
    if message:
        payload["message"] = message
    if error:
        payload["error"] = error
    return payload


def variant_task_support(task: str) -> dict[str, Any]:
    normalized = str(task or "").strip().lower()
    fp16_supported = normalized in {"classification", "detection"}
    ptq_supported = normalized in {"classification", "detection"}
    qat_supported = normalized in {"classification", "detection"}
    fp16_reason = None if fp16_supported else "FP16 is not supported for this task"
    qat_reason = None if qat_supported else "QAT is not supported for this task"
    return {
        "fp16_supported": fp16_supported,
        "fp16_reason": fp16_reason,
        "ptq_supported": ptq_supported,
        "qat_supported": qat_supported,
        "qat_reason": qat_reason,
    }


def selected_checkpoint_kind(experiment_record: dict[str, Any] | None) -> str:
    artifacts = experiment_record.get("artifacts_json") if isinstance(experiment_record, dict) else None
    if isinstance(artifacts, dict):
        kind = str(artifacts.get("selected_checkpoint_kind") or "").strip().lower()
        if kind in {"best_metric", "best_loss", "latest"}:
            return kind
    return "best_metric"


def queue_variant(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    variant_key: str,
    checkpoint_kind: str | None = None,
    quantization_strategy: str | None = None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "status": "queued",
        "error": None,
        "checkpoint_kind": checkpoint_kind,
    }
    if quantization_strategy:
        patch["quantization_strategy"] = quantization_strategy
    return _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=variant_key,
        patch=patch,
    )


def fail_variant(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    variant_key: str,
    error: str,
) -> dict[str, Any]:
    return _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=variant_key,
        patch={"status": "failed", "error": str(error)},
    )


def _variant_onnx_paths(storage: ExperimentStorage, project_id: str, experiment_id: str, attempt: int, variant_key: str) -> tuple[Path, Path]:
    return (
        storage.variant_onnx_model_path(project_id, experiment_id, attempt, variant_key),
        storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variant_key),
    )


def _extract_dataset_if_needed(storage: ExperimentStorage, *, workdir: Path, dataset_export: dict[str, Any]) -> Path:
    dataset_dir = workdir / "dataset"
    manifest_path = dataset_dir / "manifest.json"
    coco_path = dataset_dir / "coco_instances.json"
    if manifest_path.exists() or coco_path.exists():
        return dataset_dir
    zip_relpath = str(dataset_export.get("zip_relpath") or "")
    if not zip_relpath:
        raise ValueError("dataset_export_missing")
    zip_path = storage.resolve(zip_relpath)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_dir)
    return dataset_dir


def _split_asset_ids(manifest: dict[str, Any], split: str) -> list[str]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return []
    split_row = splits.get(split)
    if not isinstance(split_row, dict):
        return []
    asset_ids = split_row.get("asset_ids")
    if not isinstance(asset_ids, list):
        return []
    return [str(asset_id) for asset_id in asset_ids if isinstance(asset_id, str)]


def _calibration_paths_for_split(
    *,
    task: str,
    manifest: dict[str, Any],
    dataset_dir: Path,
    split: str,
    max_samples: int,
) -> list[Path]:
    split_asset_ids = _split_asset_ids(manifest, split)
    if not split_asset_ids:
        return []

    limit = max(1, int(max_samples))
    normalized_task = str(task or "").strip().lower()
    if normalized_task == "classification":
        samples, _class_order, _class_names, _skipped = _asset_label_samples(manifest, dataset_dir)
        sample_map = {sample.asset_id: sample for sample in samples}
        return [sample_map[asset_id].path for asset_id in split_asset_ids if asset_id in sample_map][:limit]

    asset_rows = manifest.get("assets")
    if not isinstance(asset_rows, list):
        asset_rows = []
    asset_map = {
        str(row.get("asset_id")): dataset_dir / str(row.get("path"))
        for row in asset_rows
        if isinstance(row, dict) and isinstance(row.get("asset_id"), str) and isinstance(row.get("path"), str)
    }
    return [asset_map[asset_id] for asset_id in split_asset_ids if asset_id in asset_map][:limit]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    return exp / max(float(denom), 1e-8)


def _classification_metrics_from_predictions(
    *,
    num_classes: int,
    predictions: list[PredictionRow],
) -> tuple[list[list[int]], list[ClassMetricsRow], float, float, float, float]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    total_correct = 0
    for row in predictions:
        confusion[int(row.true_class_index)][int(row.pred_class_index)] += 1
        total_correct += int(row.true_class_index == row.pred_class_index)

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    per_class: list[ClassMetricsRow] = []
    for class_index in range(num_classes):
        tp = float(confusion[class_index][class_index])
        fp = float(sum(confusion[row_index][class_index] for row_index in range(num_classes) if row_index != class_index))
        fn = float(sum(confusion[class_index][col_index] for col_index in range(num_classes) if col_index != class_index))
        support = int(sum(confusion[class_index]))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class.append(
            ClassMetricsRow(
                class_index=class_index,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        )
    total = max(len(predictions), 1)
    return (
        confusion,
        per_class,
        float(total_correct / total),
        float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        float(sum(precision_values) / len(precision_values)) if precision_values else 0.0,
        float(sum(recall_values) / len(recall_values)) if recall_values else 0.0,
    )


def _classification_sample_buckets(predictions: list[PredictionRow], *, limit: int = 100) -> dict[str, Any]:
    wrong = [row for row in predictions if int(row.true_class_index) != int(row.pred_class_index)]
    correct = [row for row in predictions if int(row.true_class_index) == int(row.pred_class_index)]
    wrong_by_confidence = sorted(wrong, key=lambda row: float(row.confidence), reverse=True)
    lowest_confidence_correct = sorted(correct, key=lambda row: float(row.confidence))

    def _payload(row: PredictionRow) -> dict[str, Any]:
        return {
            "asset_id": row.asset_id,
            "relative_path": row.relative_path,
            "true_class_index": int(row.true_class_index),
            "pred_class_index": int(row.pred_class_index),
            "confidence": float(row.confidence),
            "margin": float(row.margin),
        }

    return {
        "misclassified": [_payload(row) for row in wrong_by_confidence[:limit]],
        "lowest_confidence_correct": [_payload(row) for row in lowest_confidence_correct[:limit]],
        "highest_confidence_wrong": [_payload(row) for row in wrong_by_confidence[:limit]],
    }


def _available_onnx_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers() or ["CPUExecutionProvider"])


def _onnx_session(path: Path, *, providers: list[str] | None = None, cpu_only: bool = False) -> Any:
    import onnxruntime as ort

    if providers is None:
        providers = ["CPUExecutionProvider"] if cpu_only else (_available_onnx_providers() or ["CPUExecutionProvider"])
    return ort.InferenceSession(str(path), providers=providers)


def _session_candidates_for_variant(variant_key: str) -> list[list[str]]:
    available = _available_onnx_providers()
    candidates: list[list[str]] = []
    if variant_key == VARIANT_FP16 and "CUDAExecutionProvider" in available:
        candidates.append(["CUDAExecutionProvider"])
    if "CPUExecutionProvider" in available:
        candidates.append(["CPUExecutionProvider"])
    elif available:
        candidates.append(list(available))
    return candidates or [["CPUExecutionProvider"]]


def _open_variant_session(path: Path, *, variant_key: str, providers: list[str] | None = None) -> tuple[Any, str]:
    attempts = [providers] if providers is not None else _session_candidates_for_variant(variant_key)
    last_error: Exception | None = None
    for provider_list in attempts:
        if not provider_list:
            continue
        try:
            return _onnx_session(path, providers=list(provider_list)), str(provider_list[0])
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return _onnx_session(path, cpu_only=True), "CPUExecutionProvider"


def _evaluate_classification_split(
    *,
    dataset_dir: Path,
    split: str,
    metadata: dict[str, Any],
    onnx_path: Path,
    variant_key: str,
) -> dict[str, Any]:
    manifest = _read_json(dataset_dir / "manifest.json", {})
    if not isinstance(manifest, dict):
        raise ValueError("manifest_missing")
    split_ids = _split_asset_ids(manifest, split)
    if not split_ids:
        return {"schema_version": "1", "task": "classification", "split": split, "status": "unavailable", "message": "No assets in split"}

    samples, class_order, class_names, _skipped = _asset_label_samples(manifest, dataset_dir)
    sample_map = {sample.asset_id: sample for sample in samples}
    selected = [sample_map[asset_id] for asset_id in split_ids if asset_id in sample_map]
    if not selected:
        return {"schema_version": "1", "task": "classification", "split": split, "status": "unavailable", "message": "No labeled assets in split"}

    session, _provider = _open_variant_session(onnx_path, variant_key=variant_key)
    predictions: list[PredictionRow] = []
    total_loss = 0.0
    for sample in selected:
        tensor = preprocess_asset(sample.path, metadata)
        logits = _run_onnx(session, tensor)
        if logits.ndim != 2 or logits.shape[0] < 1:
            raise ValueError("classification_logits_invalid")
        row = np.asarray(logits[0], dtype=np.float32)
        probs = _softmax(row)
        pred_class_index = int(np.argmax(probs))
        sorted_probs = np.sort(probs)[::-1]
        confidence = float(probs[pred_class_index])
        second_best = float(sorted_probs[1]) if sorted_probs.shape[0] > 1 else 0.0
        margin = float(confidence - second_best)
        total_loss += float(-math.log(max(confidence, 1e-8)))
        predictions.append(
            PredictionRow(
                asset_id=sample.asset_id,
                relative_path=sample.relative_path,
                true_class_index=int(sample.label),
                pred_class_index=pred_class_index,
                confidence=confidence,
                margin=margin,
            )
        )

    confusion, per_class, accuracy, macro_f1, macro_precision, macro_recall = _classification_metrics_from_predictions(
        num_classes=len(class_order),
        predictions=predictions,
    )
    return {
        "schema_version": "1",
        "task": "classification",
        "split": split,
        "status": "ready",
        "computed_at": utc_now_iso(),
        "num_samples": len(predictions),
        "classes": {
            "class_order": [str(value) for value in class_order],
            "class_names": [str(value) for value in class_names],
            "id_to_index": {str(class_id): index for index, class_id in enumerate(class_order)},
        },
        "overall": {
            "avg_loss": float(total_loss / max(len(predictions), 1)),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
        },
        "per_class": [
            {
                "class_index": int(row.class_index),
                "class_id": str(class_order[int(row.class_index)]),
                "name": str(class_names[int(row.class_index)]),
                "precision": float(row.precision),
                "recall": float(row.recall),
                "f1": float(row.f1),
                "support": int(row.support),
            }
            for row in per_class
        ],
        "confusion_matrix": {
            "matrix": confusion,
            "normalized_by": "none",
            "labels": {"axis": "true_rows_pred_cols"},
        },
        "samples": _classification_sample_buckets(predictions),
    }


def _evaluate_detection_split(
    *,
    dataset_dir: Path,
    split: str,
    metadata: dict[str, Any],
    onnx_path: Path,
    variant_key: str,
) -> dict[str, Any]:
    manifest = _read_json(dataset_dir / "manifest.json", {})
    coco = _read_json(dataset_dir / "coco_instances.json", {})
    if not isinstance(manifest, dict) or not isinstance(coco, dict):
        raise ValueError("detection_manifest_missing")
    split_ids = set(_split_asset_ids(manifest, split))
    if not split_ids:
        return {"schema_version": "1", "task": "detection", "split": split, "status": "unavailable", "message": "No assets in split"}

    images = coco.get("images")
    categories = coco.get("categories")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(categories, list) or not isinstance(annotations, list):
        raise ValueError("coco_instances_invalid")

    image_rows = [row for row in images if isinstance(row, dict) and str(row.get("asset_id") or row.get("id") or "") in split_ids]
    if not image_rows:
        return {"schema_version": "1", "task": "detection", "split": split, "status": "unavailable", "message": "No images in split"}

    class_order = [str(value) for value in metadata.get("class_order") or []]
    class_names = [str(value) for value in metadata.get("class_names") or []]
    if not class_order:
        class_order = [str(cat.get("id")) for cat in categories if isinstance(cat, dict) and cat.get("id") is not None]
    if not class_names:
        class_names = [str(cat.get("name") or f"class_{cat.get('id')}") for cat in categories if isinstance(cat, dict)]
    category_index_by_id = {
        int(cat.get("id")): index
        for index, cat in enumerate(categories)
        if isinstance(cat, dict) and isinstance(cat.get("id"), int)
    }

    image_by_id = {str(row.get("id")): row for row in image_rows}
    annotations_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in annotations:
        if not isinstance(row, dict):
            continue
        image_id = str(row.get("image_id") or "")
        if image_id not in image_by_id:
            continue
        annotations_by_image.setdefault(image_id, []).append(row)

    session, _provider = _open_variant_session(onnx_path, variant_key=variant_key)
    predictions: list[DetectionPrediction] = []
    ground_truth: list[DetectionGroundTruth] = []
    for image_id, image_row in image_by_id.items():
        relative_path = str(image_row.get("file_name") or "")
        asset_id = str(image_row.get("asset_id") or image_id)
        image_path = dataset_dir / relative_path
        if not image_path.exists():
            continue
        tensor, preprocess_context = preprocess_asset_with_context(image_path, metadata)
        raw_outputs = _run_onnx_detection(session, tensor)
        boxes = _parse_detection_output(
            raw_outputs,
            class_names=class_names,
            score_threshold=0.0,
            preprocess_context=preprocess_context,
        )
        for index, box in enumerate(boxes):
            predictions.append(
                DetectionPrediction(
                    image_id=image_id,
                    class_index=int(box.class_index),
                    bbox=bbox_xywh_to_xyxy(box.bbox),
                    score=float(box.score),
                    prediction_id=f"{image_id}-pred-{index}",
                    asset_id=asset_id,
                    relative_path=relative_path,
                )
            )
        for index, annotation in enumerate(annotations_by_image.get(image_id, [])):
            bbox = annotation.get("bbox")
            category_id = annotation.get("category_id")
            if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(category_id, int):
                continue
            class_index = category_index_by_id.get(category_id)
            if class_index is None:
                continue
            ground_truth.append(
                DetectionGroundTruth(
                    image_id=image_id,
                    class_index=int(class_index),
                    bbox=bbox_xywh_to_xyxy(bbox),
                    annotation_id=str(annotation.get("id") or f"{image_id}-ann-{index}"),
                    asset_id=asset_id,
                    relative_path=relative_path,
                    area=float(annotation.get("area")) if isinstance(annotation.get("area"), (int, float)) else None,
                )
            )

    evaluation = evaluate_detection_set(
        predictions,
        ground_truth,
        class_order=class_order,
        class_names=class_names,
        box_format="xyxy",
    )
    overall = evaluation.overall
    return {
        "schema_version": "1",
        "task": "detection",
        "split": split,
        "status": "ready",
        "computed_at": utc_now_iso(),
        "num_images": int(overall.image_count if overall is not None else 0),
        "num_predictions": int(len(predictions)),
        "num_ground_truth": int(overall.ground_truth_count if overall is not None else 0),
        "classes": {
            "class_order": class_order,
            "class_names": class_names,
            "id_to_index": {str(class_id): index for index, class_id in enumerate(class_order)},
        },
        "overall": {
            "mAP50": float(evaluation.mAP50),
            "mAP50_95": float(evaluation.mAP50_95),
            "precision": float(overall.precision) if overall is not None else 0.0,
            "recall": float(overall.recall) if overall is not None else 0.0,
            "tp": int(overall.tp) if overall is not None else 0,
            "fp": int(overall.fp) if overall is not None else 0,
            "fn": int(overall.fn) if overall is not None else 0,
            "duplicate_fp": int(overall.duplicate_fp) if overall is not None else 0,
            "matched_mean_iou": float(overall.matched_mean_iou) if overall is not None and isinstance(overall.matched_mean_iou, (int, float)) else None,
        },
        "per_class": [
            {
                "class_index": int(row.class_index),
                "class_id": str(row.class_id),
                "name": str(row.name),
                "precision": float(row.precision),
                "recall": float(row.recall),
                "f1": float(row.f1),
                "support": int(row.support),
                "ap50": float(row.ap50) if isinstance(row.ap50, (int, float)) else None,
                "ap75": float(row.ap75) if isinstance(row.ap75, (int, float)) else None,
                "map_50_95": float(row.map_50_95) if isinstance(row.map_50_95, (int, float)) else None,
                "tp": int(row.tp),
                "fp": int(row.fp),
                "fn": int(row.fn),
                "duplicate_fp": int(row.duplicate_fp),
                "matched_mean_iou": float(row.matched_mean_iou) if isinstance(row.matched_mean_iou, (int, float)) else None,
            }
            for row in evaluation.per_class
        ],
    }


def _write_variant_evaluations(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    variant_key: str,
    task: str,
    dataset_export: dict[str, Any],
) -> dict[str, Any]:
    workdir = storage.variant_dir(project_id, experiment_id, attempt, variant_key) / "workdir"
    dataset_dir = _extract_dataset_if_needed(storage, workdir=workdir, dataset_export=dataset_export)
    _model_path, metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, variant_key)
    onnx_path, _metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, variant_key)
    metadata = load_metadata(metadata_path)
    evaluations: dict[str, Any] = {}
    for split in ("val", "test"):
        if task == "detection":
            payload = _evaluate_detection_split(
                dataset_dir=dataset_dir,
                split=split,
                metadata=metadata,
                onnx_path=onnx_path,
                variant_key=variant_key,
            )
        elif task == "classification":
            payload = _evaluate_classification_split(
                dataset_dir=dataset_dir,
                split=split,
                metadata=metadata,
                onnx_path=onnx_path,
                variant_key=variant_key,
            )
        else:
            payload = {
                "schema_version": "1",
                "task": task,
                "split": split,
                "status": "unavailable",
                "message": "Variant evaluation is not supported for this task",
            }
        _write_json(storage.variant_evaluation_path(project_id, experiment_id, attempt, variant_key, split), payload)
        evaluations[split] = {
            "status": payload.get("status"),
            "relpath": _as_relative_uri(storage, storage.variant_evaluation_path(project_id, experiment_id, attempt, variant_key, split)),
            "overall": payload.get("overall") if isinstance(payload, dict) else None,
        }
    return evaluations


def _iter_split_samples_for_benchmark(
    dataset_dir: Path,
    task: str,
    *,
    limit: int = 8,
) -> Iterable[Path]:
    manifest = _read_json(dataset_dir / "manifest.json", {})
    if not isinstance(manifest, dict):
        return []
    asset_rows = manifest.get("assets")
    if not isinstance(asset_rows, list):
        return []
    assets_by_id = {
        str(row.get("asset_id")): dataset_dir / str(row.get("path"))
        for row in asset_rows
        if isinstance(row, dict) and isinstance(row.get("asset_id"), str) and isinstance(row.get("path"), str)
    }
    selected: list[Path] = []
    for split in ("val", "test", "train"):
        for asset_id in _split_asset_ids(manifest, split):
            asset_path = assets_by_id.get(asset_id)
            if asset_path is None or not asset_path.exists():
                continue
            selected.append(asset_path)
            if len(selected) >= limit:
                return selected
    return selected


def _benchmark_variant_provider(
    *,
    model_path: Path,
    metadata: dict[str, Any],
    sample_paths: list[Path],
    task: str,
    provider_name: str,
) -> dict[str, Any]:
    session = _onnx_session(model_path, providers=[provider_name])
    tensors = [preprocess_asset(path, metadata) for path in sample_paths]
    for tensor in tensors[: min(3, len(tensors))]:
        if task == "detection":
            _run_onnx_detection(session, tensor)
        else:
            _run_onnx(session, tensor)

    latencies_ms: list[float] = []
    for index in range(12):
        tensor = tensors[index % len(tensors)]
        import time

        t0 = time.perf_counter()
        if task == "detection":
            _run_onnx_detection(session, tensor)
        else:
            _run_onnx(session, tensor)
        latencies_ms.append(float((time.perf_counter() - t0) * 1000.0))

    mean_ms = float(sum(latencies_ms) / max(len(latencies_ms), 1))
    p50_ms = float(statistics.median(latencies_ms))
    p95_index = max(0, min(len(latencies_ms) - 1, math.ceil(len(latencies_ms) * 0.95) - 1))
    p95_ms = float(sorted(latencies_ms)[p95_index])
    return {
        "status": "ready",
        "provider": provider_name,
        "batch_size": 1,
        "sample_count": len(latencies_ms),
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "throughput_items_per_second": float(1000.0 / max(mean_ms, 1e-8)),
        "computed_at": utc_now_iso(),
    }


def _write_variant_benchmark(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    variant_key: str,
    task: str,
    dataset_export: dict[str, Any],
) -> dict[str, Any]:
    workdir = storage.variant_dir(project_id, experiment_id, attempt, variant_key) / "workdir"
    dataset_dir = _extract_dataset_if_needed(storage, workdir=workdir, dataset_export=dataset_export)
    model_path, metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, variant_key)
    metadata = load_metadata(metadata_path)
    sample_paths = list(_iter_split_samples_for_benchmark(dataset_dir, task))
    if not sample_paths:
        payload = {"schema_version": "2", "status": "unavailable", "message": "No benchmark samples available", "devices": {}}
        _write_json(storage.variant_benchmark_path(project_id, experiment_id, attempt, variant_key), payload)
        return payload

    device_payloads: dict[str, Any] = {}
    available_providers = set(_available_onnx_providers())
    provider_by_device = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    for device_key, provider_name in provider_by_device.items():
        if provider_name not in available_providers:
            continue
        try:
            device_payloads[device_key] = _benchmark_variant_provider(
                model_path=model_path,
                metadata=metadata,
                sample_paths=sample_paths,
                task=task,
                provider_name=provider_name,
            )
        except Exception as exc:
            device_payloads[device_key] = {
                "status": "unavailable",
                "provider": provider_name,
                "message": str(exc),
            }

    default_benchmark = device_payloads.get("cpu")
    overall_status = "ready" if any(str(payload.get("status")) == "ready" for payload in device_payloads.values()) else "unavailable"
    payload = {
        "schema_version": "2",
        "status": overall_status,
        "benchmark": default_benchmark,
        "devices": device_payloads,
        "default_device": "cpu" if "cpu" in device_payloads else (next(iter(device_payloads.keys()), None)),
        "computed_at": utc_now_iso(),
    }
    _write_json(storage.variant_benchmark_path(project_id, experiment_id, attempt, variant_key), payload)
    return payload


class _CalibrationReader:
    def __init__(self, input_name: str, tensors: list[np.ndarray]) -> None:
        self._input_name = input_name
        self._tensors = list(tensors)
        self._cursor = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._cursor >= len(self._tensors):
            return None
        tensor = self._tensors[self._cursor]
        self._cursor += 1
        return {self._input_name: tensor}

    def rewind(self) -> None:
        self._cursor = 0


def create_fp32_baseline_variant(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    task: str,
    dataset_export: dict[str, Any],
    emit_event: callable | None = None,
) -> dict[str, Any]:
    base_model = storage.run_dir(project_id, experiment_id, attempt) / "onnx" / "model.onnx"
    base_metadata = storage.run_dir(project_id, experiment_id, attempt) / "onnx" / "onnx.metadata.json"
    if not base_metadata.exists():
        raise ValueError("onnx_metadata_missing")

    if emit_event is not None:
        emit_event(_variant_event(VARIANT_FP32, "running", attempt=attempt, message="Building FP32 baseline"))
    _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP32,
        patch={"status": "running", "error": None, "source_variant_key": None},
    )

    model_path, metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, VARIANT_FP32)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if base_model.exists():
        shutil.copy2(base_model, model_path)
    metadata_payload = _read_json(base_metadata, {})
    if not isinstance(metadata_payload, dict):
        metadata_payload = {}
    metadata_payload = dict(metadata_payload)
    metadata_payload["variant_key"] = VARIANT_FP32
    metadata_payload["variant_kind"] = "baseline"
    metadata_payload["quantized"] = False
    metadata_payload["variant_exported_at"] = utc_now_iso()
    _write_json(metadata_path, metadata_payload)

    evaluations = _write_variant_evaluations(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP32,
        task=task,
        dataset_export=dataset_export,
    )
    benchmark = _write_variant_benchmark(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP32,
        task=task,
        dataset_export=dataset_export,
    )

    row = _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP32,
        patch={
            "status": "ready",
            "onnx": {
                "model_relpath": _as_relative_uri(storage, model_path) if model_path.exists() else None,
                "metadata_relpath": _as_relative_uri(storage, metadata_path),
                "size_bytes": int(model_path.stat().st_size) if model_path.exists() else None,
            },
            "evaluation": evaluations,
            "benchmark": benchmark.get("benchmark"),
            "benchmarks": benchmark.get("devices", {}),
            "quantized": False,
        },
    )
    if emit_event is not None:
        emit_event(_variant_event(VARIANT_FP32, "ready", attempt=attempt, message="FP32 baseline ready"))
    return row


def run_fp16_variant(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    task: str,
    dataset_export: dict[str, Any],
    checkpoint_kind: str | None,
    emit_event: callable | None = None,
) -> dict[str, Any]:
    normalized_task = str(task or "").strip().lower()
    if normalized_task not in {"classification", "detection"}:
        row = _update_variant_row(
            storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            variant_key=VARIANT_FP16,
            patch={"status": "unsupported", "error": "FP16 is not supported for this task"},
        )
        if emit_event is not None:
            emit_event(_variant_event(VARIANT_FP16, "unsupported", attempt=attempt, error="FP16 is not supported for this task"))
        return row

    if emit_event is not None:
        emit_event(_variant_event(VARIANT_FP16, "running", attempt=attempt, message="Converting FP16 variant"))
    _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP16,
        patch={"status": "running", "error": None, "checkpoint_kind": checkpoint_kind or "best_metric"},
    )

    base_model, base_metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, VARIANT_FP32)
    if not base_model.exists() or not base_metadata_path.exists():
        create_fp32_baseline_variant(
            storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            task=normalized_task,
            dataset_export=dataset_export,
            emit_event=None,
        )
    if not base_model.exists() or not base_metadata_path.exists():
        raise ValueError("fp32_variant_missing")

    try:
        import onnx
        from onnxconverter_common.float16 import convert_float_to_float16
    except Exception as exc:
        raise ValueError(f"fp16_conversion_unavailable:{exc}") from exc

    model_path, metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, VARIANT_FP16)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_model = onnx.load_model(str(base_model))
    fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)
    onnx.save_model(fp16_model, str(model_path))

    base_metadata = load_metadata(base_metadata_path)
    metadata_payload = dict(base_metadata)
    metadata_payload["variant_key"] = VARIANT_FP16
    metadata_payload["variant_kind"] = "fp16"
    metadata_payload["quantized"] = False
    metadata_payload["numeric_precision"] = "fp16"
    metadata_payload["checkpoint_kind"] = checkpoint_kind or base_metadata.get("checkpoint_kind")
    metadata_payload["checkpoint_uri"] = base_metadata.get("checkpoint_uri")
    metadata_payload["variant_exported_at"] = utc_now_iso()
    _write_json(metadata_path, metadata_payload)

    evaluations = _write_variant_evaluations(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP16,
        task=normalized_task,
        dataset_export=dataset_export,
    )
    benchmark_payload = _write_variant_benchmark(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP16,
        task=normalized_task,
        dataset_export=dataset_export,
    )
    row = _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_FP16,
        patch={
            "status": "ready",
            "onnx": {
                "model_relpath": _as_relative_uri(storage, model_path),
                "metadata_relpath": _as_relative_uri(storage, metadata_path),
                "size_bytes": int(model_path.stat().st_size),
            },
            "evaluation": evaluations,
            "benchmark": benchmark_payload.get("benchmark"),
            "benchmarks": benchmark_payload.get("devices", {}),
            "quantized": False,
        },
    )
    if emit_event is not None:
        emit_event(_variant_event(VARIANT_FP16, "ready", attempt=attempt, message="FP16 variant ready"))
    return row


def _resolve_checkpoint_for_kind(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    checkpoint_kind: str | None,
) -> tuple[str | None, Path | None]:
    normalized = str(checkpoint_kind or "").strip().lower()
    if normalized not in {"best_metric", "best_loss", "latest"}:
        normalized = "best_metric"
    if normalized == "best_metric":
        return _resolve_best_checkpoint(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    checkpoints_path = storage.run_dir(project_id, experiment_id, attempt) / "checkpoints" / f"{normalized}.pt"
    if checkpoints_path.exists():
        return normalized, checkpoints_path
    return _resolve_best_checkpoint(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)


def run_ptq_variant(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    task: str,
    dataset_export: dict[str, Any],
    checkpoint_kind: str | None,
    calibration_max_samples: int = 256,
    emit_event: callable | None = None,
) -> dict[str, Any]:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    if emit_event is not None:
        emit_event(_variant_event(VARIANT_PTQ_INT8, "running", attempt=attempt, message="Quantizing PTQ INT8"))
    _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_PTQ_INT8,
        patch={"status": "running", "error": None, "checkpoint_kind": checkpoint_kind or "best_metric"},
    )

    base_model, base_metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, VARIANT_FP32)
    if not base_model.exists():
        raise ValueError("fp32_variant_missing")
    base_metadata = load_metadata(base_metadata_path)

    workdir = storage.variant_dir(project_id, experiment_id, attempt, VARIANT_PTQ_INT8) / "workdir"
    dataset_dir = _extract_dataset_if_needed(storage, workdir=workdir, dataset_export=dataset_export)
    manifest = _read_json(dataset_dir / "manifest.json", {})
    if not isinstance(manifest, dict):
        raise ValueError("manifest_missing")
    calibration_paths = _calibration_paths_for_split(
        task=task,
        manifest=manifest,
        dataset_dir=dataset_dir,
        split="train",
        max_samples=calibration_max_samples,
    )
    if not calibration_paths:
        raise ValueError("ptq_calibration_split_missing")

    session = _onnx_session(base_model, cpu_only=True)
    input_name = session.get_inputs()[0].name
    tensors = [preprocess_asset(path, base_metadata) for path in calibration_paths if path.exists()]
    if not tensors:
        raise ValueError("ptq_calibration_tensors_missing")

    model_path, metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, VARIANT_PTQ_INT8)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(base_model),
        str(model_path),
        _CalibrationReader(input_name, tensors),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    quantized_metadata = dict(base_metadata)
    quantized_metadata["variant_key"] = VARIANT_PTQ_INT8
    quantized_metadata["variant_kind"] = "ptq"
    quantized_metadata["quantized"] = True
    quantized_metadata["quantization"] = {
        "mode": "static_ptq",
        "activation_type": "qint8",
        "weight_type": "qint8",
        "calibration_max_samples": len(tensors),
    }
    quantized_metadata["checkpoint_kind"] = checkpoint_kind or base_metadata.get("checkpoint_kind")
    quantized_metadata["checkpoint_uri"] = base_metadata.get("checkpoint_uri")
    quantized_metadata["variant_exported_at"] = utc_now_iso()
    _write_json(metadata_path, quantized_metadata)

    evaluations = _write_variant_evaluations(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_PTQ_INT8,
        task=task,
        dataset_export=dataset_export,
    )
    benchmark = _write_variant_benchmark(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_PTQ_INT8,
        task=task,
        dataset_export=dataset_export,
    )
    row = _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_PTQ_INT8,
        patch={
            "status": "ready",
            "onnx": {
                "model_relpath": _as_relative_uri(storage, model_path),
                "metadata_relpath": _as_relative_uri(storage, metadata_path),
                "size_bytes": int(model_path.stat().st_size),
            },
            "evaluation": evaluations,
            "benchmark": benchmark.get("benchmark"),
            "benchmarks": benchmark.get("devices", {}),
            "quantized": True,
            "quantization_strategy": "static_ptq",
        },
    )
    if emit_event is not None:
        emit_event(_variant_event(VARIANT_PTQ_INT8, "ready", attempt=attempt, message="PTQ INT8 ready"))
    return row


def _variant_checkpoint_defaults() -> list[dict[str, Any]]:
    return [
        {"kind": "best_metric", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None, "status": "pending", "error": None},
        {"kind": "best_loss", "epoch": None, "metric_name": "val_loss", "value": None, "updated_at": None, "uri": None, "status": "pending", "error": None},
        {"kind": "latest", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None, "status": "pending", "error": None},
    ]


def _save_variant_checkpoint(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    variant_key: str,
    kind: str,
    epoch: int,
    metric_name: str | None,
    value: float | None,
    state_dict: dict[str, Any],
    keep_last: int = 1,
) -> dict[str, Any]:
    checkpoint_dir = storage.variant_checkpoints_dir(project_id, experiment_id, attempt, variant_key)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if kind == "latest":
        versioned_path = checkpoint_dir / f"latest_epoch_{int(epoch)}.pt"
        torch.save(state_dict, versioned_path)
        path = checkpoint_dir / "latest.pt"
        shutil.copy2(versioned_path, path)
        epoch_files = sorted(checkpoint_dir.glob("latest_epoch_*.pt"), reverse=True)
        for stale_path in epoch_files[max(1, int(keep_last)) :]:
            stale_path.unlink(missing_ok=True)
    else:
        path = checkpoint_dir / f"{kind}.pt"
        torch.save(state_dict, path)

    checkpoints_path = storage.variant_checkpoints_path(project_id, experiment_id, attempt, variant_key)
    rows = _read_json(checkpoints_path, _variant_checkpoint_defaults())
    if not isinstance(rows, list):
        rows = _variant_checkpoint_defaults()
    updated_row = {
        "kind": kind,
        "epoch": int(epoch),
        "metric_name": metric_name,
        "value": float(value) if isinstance(value, (int, float)) else None,
        "updated_at": utc_now_iso(),
        "uri": _as_relative_uri(storage, path),
        "status": "ok",
        "error": None,
    }
    replaced = False
    for index, row in enumerate(rows):
        if isinstance(row, dict) and str(row.get("kind") or "") == kind:
            rows[index] = updated_row
            replaced = True
            break
    if not replaced:
        rows.append(updated_row)
    _write_json(checkpoints_path, rows)
    return updated_row


def run_qat_variant(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    task: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    dataset_export: dict[str, Any],
    checkpoint_kind: str | None,
    epochs_override: int | None,
    learning_rate_override: float | None,
    calibration_max_samples: int,
    emit_event: callable | None = None,
) -> dict[str, Any]:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    normalized_task = str(task or "").strip().lower()
    if normalized_task not in {"classification", "detection"}:
        row = _update_variant_row(
            storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            variant_key=VARIANT_QAT_INT8,
            patch={"status": "unsupported", "error": "QAT is not supported for this task"},
        )
        if emit_event is not None:
            emit_event(_variant_event(VARIANT_QAT_INT8, "unsupported", attempt=attempt, error="QAT is not supported for this task"))
        return row

    checkpoint_kind_resolved, checkpoint_path = _resolve_checkpoint_for_kind(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        checkpoint_kind=checkpoint_kind,
    )
    if checkpoint_path is None:
        raise ValueError("qat_checkpoint_missing")

    variant_dir = storage.variant_dir(project_id, experiment_id, attempt, VARIANT_QAT_INT8)
    workdir = variant_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    variant_metrics_path = storage.variant_metrics_path(project_id, experiment_id, attempt, VARIANT_QAT_INT8)
    variant_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    variant_metrics_path.write_text("", encoding="utf-8")

    _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_QAT_INT8,
        patch={
            "status": "running",
            "error": None,
            "checkpoint_kind": checkpoint_kind_resolved,
            "quantization_strategy": "finetune_then_ptq",
        },
    )
    if emit_event is not None:
        emit_event(_variant_event(VARIANT_QAT_INT8, "running", attempt=attempt, message="Starting QAT fine-tune"))

    pipeline = PIPELINE_REGISTRY.get(normalized_task)
    if pipeline is None:
        raise ValueError(f"{normalized_task}_pipeline_missing")

    job_like = type(
        "VariantJob",
        (),
        {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "attempt": attempt,
            "model_id": "",
            "task": normalized_task,
            "task_id": None,
            "job_id": f"qat-{attempt}",
            "model_config": model_config,
            "training_config": training_config,
            "dataset_export": dataset_export,
        },
    )()

    base_epochs = int(training_config.get("epochs") or 1)
    qat_epochs = epochs_override if isinstance(epochs_override, int) and epochs_override >= 1 else max(3, min(10, math.ceil(base_epochs * 0.2)))
    optimizer_cfg = training_config.get("optimizer")
    optimizer_cfg = dict(optimizer_cfg) if isinstance(optimizer_cfg, dict) else {}
    base_lr = float(optimizer_cfg.get("lr") or 0.001)
    qat_lr = learning_rate_override if isinstance(learning_rate_override, (int, float)) and float(learning_rate_override) > 0 else base_lr * 0.1

    tuned_training_config = dict(training_config)
    tuned_training_config["epochs"] = qat_epochs
    tuned_training_config["optimizer"] = dict(optimizer_cfg)
    tuned_training_config["optimizer"]["lr"] = qat_lr
    tuned_training_config["resume"] = {"enabled": False, "checkpoint_kind": "latest"}
    tuned_training_config["runtime"] = dict(training_config.get("runtime") or {})
    job_like.training_config = tuned_training_config

    loaders = pipeline.build_loaders(job_like, workdir, storage)
    device = resolve_device(tuned_training_config)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint_payload.get("model_state_dict") if isinstance(checkpoint_payload, dict) else None
    resume_state = {"model_state_dict": model_state} if isinstance(model_state, dict) else None

    training_log_path = storage.variant_training_log_path(project_id, experiment_id, attempt, VARIANT_QAT_INT8)
    training_log_path.parent.mkdir(parents=True, exist_ok=True)
    training_log_path.write_text("", encoding="utf-8")
    summary: dict[str, Any] = {"best_metric": None, "best_epoch": None}

    def _append_log(message: str) -> None:
        with training_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
            handle.write("\n")

    def on_epoch(epoch_row: Any) -> None:
        if normalized_task == "classification":
            row = {
                "attempt": attempt,
                "epoch": int(epoch_row.epoch),
                "train_loss": float(epoch_row.train_loss),
                "train_accuracy": float(epoch_row.train_accuracy) if isinstance(epoch_row.train_accuracy, (int, float)) else None,
                "val_loss": float(epoch_row.val_loss) if isinstance(epoch_row.val_loss, (int, float)) else None,
                "val_accuracy": float(epoch_row.val_accuracy) if isinstance(epoch_row.val_accuracy, (int, float)) else None,
                "val_macro_f1": float(epoch_row.val_macro_f1) if isinstance(epoch_row.val_macro_f1, (int, float)) else None,
                "val_macro_precision": float(epoch_row.val_macro_precision) if isinstance(epoch_row.val_macro_precision, (int, float)) else None,
                "val_macro_recall": float(epoch_row.val_macro_recall) if isinstance(epoch_row.val_macro_recall, (int, float)) else None,
                "lr": float(epoch_row.lr),
                "epoch_seconds": float(epoch_row.epoch_seconds),
                "eta_seconds": float(epoch_row.eta_seconds) if isinstance(epoch_row.eta_seconds, (int, float)) else None,
            }
            log_message = f"epoch={row['epoch']} train_loss={row['train_loss']:.4f} val_accuracy={row['val_accuracy']}"
        else:
            detection_row: DetectionEpochMetrics = epoch_row
            row = {
                "attempt": attempt,
                "epoch": int(detection_row.epoch),
                "train_loss": float(detection_row.train_loss),
                "val_map": float(detection_row.mAP50) if isinstance(detection_row.mAP50, (int, float)) else None,
                "val_map_50_95": float(detection_row.mAP50_95) if isinstance(detection_row.mAP50_95, (int, float)) else None,
                "val_precision": float(detection_row.precision) if isinstance(detection_row.precision, (int, float)) else None,
                "val_recall": float(detection_row.recall) if isinstance(detection_row.recall, (int, float)) else None,
                "val_matched_mean_iou": (
                    float(detection_row.matched_mean_iou)
                    if isinstance(detection_row.matched_mean_iou, (int, float))
                    else None
                ),
                "val_tp": int(detection_row.tp) if isinstance(detection_row.tp, int) else None,
                "val_fp": int(detection_row.fp) if isinstance(detection_row.fp, int) else None,
                "val_fn": int(detection_row.fn) if isinstance(detection_row.fn, int) else None,
                "val_duplicate_fp": int(detection_row.duplicate_fp) if isinstance(detection_row.duplicate_fp, int) else None,
                "lr": float(detection_row.lr),
                "epoch_seconds": float(detection_row.epoch_seconds),
                "eta_seconds": float(detection_row.eta_seconds) if isinstance(detection_row.eta_seconds, (int, float)) else None,
                "evaluated": bool(detection_row.evaluated),
            }
            log_message = f"epoch={row['epoch']} train_loss={row['train_loss']:.4f} val_map_50_95={row['val_map_50_95']}"
        with variant_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
        _append_log(log_message)
        if emit_event is not None:
            emit_event(
                _variant_event(
                    VARIANT_QAT_INT8,
                    "running",
                    attempt=attempt,
                    message=f"QAT fine-tune epoch {row['epoch']}/{qat_epochs}",
                )
            )

    def on_checkpoint(kind: str, epoch: int, metric_name: str | None, value: float | None, state: dict[str, Any]) -> None:
        _save_variant_checkpoint(
            storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            variant_key=VARIANT_QAT_INT8,
            kind=kind,
            epoch=epoch,
            metric_name=metric_name,
            value=value,
            state_dict=state,
            keep_last=max(1, int((training_config.get("logging") or {}).get("keep_last") or 1)),
        )
        if kind == "best_metric":
            summary["best_metric"] = value
            summary["best_epoch"] = epoch

    if normalized_task == "classification":
        run_status, _final_eval = run_training(
            model_config=model_config,
            training_config=tuned_training_config,
            train_loader=loaders.train,
            val_loader=loaders.val,
            num_classes=loaders.num_classes,
            should_cancel=lambda: False,
            on_epoch=on_epoch,
            on_checkpoint=on_checkpoint,
            device=device,
            resume_state=resume_state,
        )
    else:
        run_status, _final_eval = run_detection_training(
            model_config=model_config,
            training_config=tuned_training_config,
            train_loader=loaders.train,
            val_loader=loaders.val,
            num_classes=loaders.num_classes,
            class_names=loaders.class_names,
            class_order=loaders.class_order,
            should_cancel=lambda: False,
            on_epoch=on_epoch,
            on_checkpoint=on_checkpoint,
            device=device,
            resume_state=resume_state,
        )
    if run_status not in {"completed", "done"}:
        raise ValueError(f"qat_training_{run_status}")

    best_checkpoint = storage.variant_checkpoints_dir(project_id, experiment_id, attempt, VARIANT_QAT_INT8) / "best_metric.pt"
    if not best_checkpoint.exists():
        best_checkpoint = storage.variant_checkpoints_dir(project_id, experiment_id, attempt, VARIANT_QAT_INT8) / "latest.pt"
    checkpoint_payload = torch.load(best_checkpoint, map_location="cpu")
    model_state = checkpoint_payload.get("model_state_dict") if isinstance(checkpoint_payload, dict) else checkpoint_payload
    if normalized_task == "classification":
        model = build_classifier_model(model_config, num_classes_override=loaders.num_classes)
    else:
        model = _build_detection_model(model_config, num_classes=loaders.num_classes)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    fp32_temp_dir = variant_dir / "fp32_qat"
    input_shape = _input_shape_from_model(model_config)
    preprocess = _preprocess_from_model(model_config, input_shape=input_shape)
    export_result = export_model_to_onnx(
        model,
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        checkpoint_kind=checkpoint_kind_resolved,
        checkpoint_uri=_as_relative_uri(storage, best_checkpoint),
        input_shape=input_shape,
        input_names=["input"],
        output_names=["output"],
        preprocess=preprocess,
        class_order=loaders.class_order,
        class_names=loaders.class_names,
        extra_metadata={"task": normalized_task, "variant_intermediate": "qat_fp32"},
        output_dir=fp32_temp_dir,
    )
    if export_result.status != "exported" or export_result.model_uri is None:
        raise ValueError(export_result.error or "qat_export_failed")

    intermediate_model = storage.resolve(export_result.model_uri)
    intermediate_metadata = storage.resolve(export_result.metadata_uri)
    base_metadata = load_metadata(intermediate_metadata)
    session = _onnx_session(intermediate_model, cpu_only=True)
    input_name = session.get_inputs()[0].name
    dataset_dir = _extract_dataset_if_needed(storage, workdir=workdir, dataset_export=dataset_export)
    sample_manifest = _read_json(dataset_dir / "manifest.json", {})
    calibration_paths = _calibration_paths_for_split(
        task=normalized_task,
        manifest=sample_manifest if isinstance(sample_manifest, dict) else {},
        dataset_dir=dataset_dir,
        split="train",
        max_samples=calibration_max_samples,
    )
    tensors = [preprocess_asset(path, base_metadata) for path in calibration_paths if path.exists()]
    if not tensors:
        raise ValueError("qat_calibration_samples_missing")

    model_path, metadata_path = _variant_onnx_paths(storage, project_id, experiment_id, attempt, VARIANT_QAT_INT8)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(intermediate_model),
        str(model_path),
        _CalibrationReader(input_name, tensors),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    metadata_payload = dict(base_metadata)
    metadata_payload["variant_key"] = VARIANT_QAT_INT8
    metadata_payload["variant_kind"] = "qat"
    metadata_payload["quantized"] = True
    metadata_payload["quantization"] = {
        "mode": "static_int8_after_finetune",
        "strategy": "finetune_then_ptq",
        "activation_type": "qint8",
        "weight_type": "qint8",
        "calibration_max_samples": len(tensors),
    }
    metadata_payload["qat"] = {
        "epochs": qat_epochs,
        "learning_rate": qat_lr,
        "checkpoint_kind": checkpoint_kind_resolved,
        "best_epoch": summary.get("best_epoch"),
        "best_metric": summary.get("best_metric"),
    }
    metadata_payload["variant_exported_at"] = utc_now_iso()
    _write_json(metadata_path, metadata_payload)

    evaluations = _write_variant_evaluations(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_QAT_INT8,
        task=normalized_task,
        dataset_export=dataset_export,
    )
    benchmark = _write_variant_benchmark(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_QAT_INT8,
        task=normalized_task,
        dataset_export=dataset_export,
    )
    row = _update_variant_row(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        variant_key=VARIANT_QAT_INT8,
        patch={
            "status": "ready",
            "onnx": {
                "model_relpath": _as_relative_uri(storage, model_path),
                "metadata_relpath": _as_relative_uri(storage, metadata_path),
                "size_bytes": int(model_path.stat().st_size),
            },
            "evaluation": evaluations,
            "benchmark": benchmark.get("benchmark"),
            "benchmarks": benchmark.get("devices", {}),
            "quantized": True,
            "qat": {
                "epochs": qat_epochs,
                "learning_rate": qat_lr,
                "strategy": "finetune_then_ptq",
            },
        },
    )
    if emit_event is not None:
        emit_event(_variant_event(VARIANT_QAT_INT8, "ready", attempt=attempt, message="QAT INT8 ready"))
    return row
