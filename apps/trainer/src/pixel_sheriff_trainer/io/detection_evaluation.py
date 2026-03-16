from __future__ import annotations

import json
from typing import Any

from pixel_sheriff_trainer.detection.evaluation import DetectionEvaluation
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.time import utc_now_iso


def _bbox_payload(values: tuple[float, float, float, float] | list[float]) -> list[float]:
    return [float(value) for value in values]


def _prediction_row_payload(row: Any) -> dict[str, Any]:
    return {
        "prediction_id": str(row.prediction_id),
        "image_id": str(row.image_id),
        "asset_id": str(row.asset_id or ""),
        "relative_path": str(row.relative_path or ""),
        "class_index": int(row.class_index),
        "class_id": str(row.class_id),
        "name": str(row.name),
        "bbox": _bbox_payload(row.bbox),
        "score": float(row.score),
        "status": str(row.status),
        "reason": str(row.reason),
        "rank": int(row.rank),
        "matched_ground_truth_id": str(row.matched_ground_truth_id) if row.matched_ground_truth_id is not None else None,
        "matched_iou": float(row.matched_iou) if isinstance(row.matched_iou, (int, float)) else None,
        "best_same_class_iou": float(row.best_same_class_iou) if isinstance(row.best_same_class_iou, (int, float)) else None,
        "best_same_class_ground_truth_id": (
            str(row.best_same_class_ground_truth_id)
            if row.best_same_class_ground_truth_id is not None
            else None
        ),
        "best_any_iou": float(row.best_any_iou) if isinstance(row.best_any_iou, (int, float)) else None,
        "best_any_ground_truth_id": str(row.best_any_ground_truth_id) if row.best_any_ground_truth_id is not None else None,
        "best_any_ground_truth_class_index": (
            int(row.best_any_ground_truth_class_index)
            if isinstance(row.best_any_ground_truth_class_index, int)
            else None
        ),
    }


def _ground_truth_payload(row: Any) -> dict[str, Any]:
    return {
        "annotation_id": str(row.annotation_id),
        "image_id": str(row.image_id),
        "asset_id": str(row.asset_id or ""),
        "relative_path": str(row.relative_path or ""),
        "class_index": int(row.class_index),
        "class_id": str(row.class_id),
        "name": str(row.name),
        "bbox": _bbox_payload(row.bbox),
        "area": float(row.area),
        "matched_prediction_id": str(row.matched_prediction_id) if row.matched_prediction_id is not None else None,
        "matched_iou": float(row.matched_iou) if isinstance(row.matched_iou, (int, float)) else None,
    }


def write_detection_evaluation(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    model_id: str | None,
    task_id: str | None,
    job_id: str | None,
    dataset_export: dict[str, Any] | None,
    class_order: list[str],
    class_names: list[str],
    evaluation: DetectionEvaluation,
) -> None:
    computed_at = utc_now_iso()
    class_id_to_index = {str(class_id): index for index, class_id in enumerate(class_order)}
    overall = evaluation.overall
    diagnostics = evaluation.diagnostics
    prediction_rows = list(diagnostics.prediction_rows if diagnostics is not None else [])
    unmatched_ground_truths = list(diagnostics.unmatched_ground_truths if diagnostics is not None else [])

    dataset_export = dataset_export if isinstance(dataset_export, dict) else {}
    provenance = {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "attempt": int(attempt),
        "model_id": str(model_id or ""),
        "task_id": str(task_id or ""),
        "job_id": str(job_id or ""),
        "dataset_version_id": str(dataset_export.get("dataset_version_id") or ""),
        "dataset_export_hash": str(dataset_export.get("content_hash") or ""),
        "dataset_export_relpath": str(dataset_export.get("zip_relpath") or ""),
    }

    size_bucket_payload = {
        str(bucket.name): {
            "ground_truth_count": int(bucket.ground_truth_count),
            "prediction_count": int(bucket.prediction_count),
            "ap50": float(bucket.ap50) if isinstance(bucket.ap50, (int, float)) else None,
            "map_50_95": float(bucket.map_50_95) if isinstance(bucket.map_50_95, (int, float)) else None,
            "precision": float(bucket.precision),
            "recall": float(bucket.recall),
        }
        for bucket in (overall.size_buckets if overall is not None else [])
    }

    evaluation_payload = {
        "schema_version": "1",
        "task": "detection",
        "computed_at": computed_at,
        "split": "val",
        "num_images": int(overall.image_count if overall is not None else 0),
        "num_predictions": int(len(prediction_rows)),
        "num_ground_truth": int(overall.ground_truth_count if overall is not None else 0),
        "provenance": provenance,
        "classes": {
            "class_order": [str(class_id) for class_id in class_order],
            "class_names": [str(name) for name in class_names],
            "id_to_index": class_id_to_index,
        },
        "thresholds": {
            "iou": [float(value) for value in evaluation.iou_thresholds],
            "diagnostics_iou_threshold": float(evaluation.diagnostics_iou_threshold),
            "score_threshold": float(evaluation.score_threshold) if isinstance(evaluation.score_threshold, (int, float)) else None,
            "max_detections_per_image": (
                int(evaluation.max_detections_per_image)
                if isinstance(evaluation.max_detections_per_image, int)
                else None
            ),
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
            "ap_small": float(overall.ap_small) if overall is not None and isinstance(overall.ap_small, (int, float)) else None,
            "ap_medium": float(overall.ap_medium) if overall is not None and isinstance(overall.ap_medium, (int, float)) else None,
            "ap_large": float(overall.ap_large) if overall is not None and isinstance(overall.ap_large, (int, float)) else None,
            "size_buckets": size_bucket_payload,
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
                "ap_by_iou": {
                    str(key): float(value) if isinstance(value, (int, float)) else None
                    for key, value in row.ap_by_iou.items()
                },
                "tp": int(row.tp),
                "fp": int(row.fp),
                "fn": int(row.fn),
                "duplicate_fp": int(row.duplicate_fp),
                "matched_mean_iou": float(row.matched_mean_iou) if isinstance(row.matched_mean_iou, (int, float)) else None,
            }
            for row in evaluation.per_class
        ],
        "pr_curves": [
            {
                "class_index": int(curve.class_index),
                "class_id": str(curve.class_id),
                "name": str(curve.name),
                "iou_threshold": float(curve.iou_threshold),
                "scores": [float(value) for value in curve.scores],
                "precision": [float(value) for value in curve.precision],
                "recall": [float(value) for value in curve.recall],
                "precision_envelope": [float(value) for value in curve.precision_envelope],
            }
            for curve in evaluation.pr_curves
        ],
        "diagnostics": {
            "per_image": [
                {
                    "image_id": str(row.image_id),
                    "asset_id": str(row.asset_id or ""),
                    "relative_path": str(row.relative_path or ""),
                    "prediction_count": int(row.prediction_count),
                    "ground_truth_count": int(row.ground_truth_count),
                    "predictions": [_prediction_row_payload(item) for item in row.predictions],
                    "matched_pairs": [
                        {
                            "image_id": str(item.image_id),
                            "prediction_id": str(item.prediction_id),
                            "ground_truth_id": str(item.ground_truth_id),
                            "class_index": int(item.class_index),
                            "class_id": str(item.class_id),
                            "name": str(item.name),
                            "score": float(item.score),
                            "iou": float(item.iou),
                        }
                        for item in row.matched_pairs
                    ],
                    "unmatched_ground_truths": [_ground_truth_payload(item) for item in row.unmatched_ground_truths],
                }
                for row in (diagnostics.per_image if diagnostics is not None else [])
            ],
            "unmatched_ground_truths": [_ground_truth_payload(row) for row in unmatched_ground_truths],
            "prediction_rows": [_prediction_row_payload(row) for row in prediction_rows],
            "confidence_traces": [
                {
                    "class_index": int(trace.class_index),
                    "class_id": str(trace.class_id),
                    "name": str(trace.name),
                    "iou_threshold": float(trace.iou_threshold),
                    "rows": [
                        {
                            "prediction_id": str(row.prediction_id),
                            "image_id": str(row.image_id),
                            "score": float(row.score),
                            "status": str(row.status),
                            "reason": str(row.reason),
                            "cumulative_tp": int(row.cumulative_tp),
                            "cumulative_fp": int(row.cumulative_fp),
                            "precision": float(row.precision),
                            "recall": float(row.recall),
                            "matched_ground_truth_id": (
                                str(row.matched_ground_truth_id)
                                if row.matched_ground_truth_id is not None
                                else None
                            ),
                            "iou": float(row.iou) if isinstance(row.iou, (int, float)) else None,
                        }
                        for row in trace.rows
                    ],
                }
                for trace in (diagnostics.confidence_traces if diagnostics is not None else [])
            ],
        },
        "samples": {
            "misclassified": [],
            "lowest_confidence_correct": [],
            "highest_confidence_wrong": [],
        },
    }

    predictions_meta_payload = {
        "schema_version": "1",
        "attempt": int(attempt),
        "task": "detection",
        "split": "val",
        "computed_at": computed_at,
        "num_predictions": len(prediction_rows),
        "num_ground_truth": len(unmatched_ground_truths) + int(overall.tp if overall is not None else 0),
        "provenance": provenance,
        "thresholds": evaluation_payload["thresholds"],
    }

    run_evaluation_path = storage.evaluation_path(project_id, experiment_id, attempt)
    run_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    run_evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")

    run_predictions_path = storage.predictions_path(project_id, experiment_id, attempt)
    run_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with run_predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(_prediction_row_payload(row), sort_keys=True))
            handle.write("\n")

    run_predictions_meta_path = storage.predictions_meta_path(project_id, experiment_id, attempt)
    run_predictions_meta_path.parent.mkdir(parents=True, exist_ok=True)
    run_predictions_meta_path.write_text(json.dumps(predictions_meta_payload, indent=2, sort_keys=True), encoding="utf-8")

    latest_evaluation_path = storage.evaluation_path(project_id, experiment_id, None)
    latest_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    latest_evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")

    latest_predictions_path = storage.predictions_path(project_id, experiment_id, None)
    latest_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with latest_predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(_prediction_row_payload(row), sort_keys=True))
            handle.write("\n")

    latest_predictions_meta_path = storage.predictions_meta_path(project_id, experiment_id, None)
    latest_predictions_meta_path.parent.mkdir(parents=True, exist_ok=True)
    latest_predictions_meta_path.write_text(json.dumps(predictions_meta_payload, indent=2, sort_keys=True), encoding="utf-8")
