from __future__ import annotations

import json
from typing import Any

from pixel_sheriff_trainer.classification.eval import ClassifierEvaluation, PredictionRow
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.utils.time import utc_now_iso

SAMPLE_BUCKET_LIMIT = 200


def _as_prediction_payload(row: PredictionRow) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "asset_id": row.asset_id,
        "relative_path": row.relative_path,
        "true_class_index": int(row.true_class_index),
        "pred_class_index": int(row.pred_class_index),
        "confidence": float(row.confidence),
    }
    if isinstance(row.margin, (float, int)):
        payload["margin"] = float(row.margin)
    return payload


def _sample_buckets(predictions: list[PredictionRow], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    wrong = [row for row in predictions if int(row.true_class_index) != int(row.pred_class_index)]
    correct = [row for row in predictions if int(row.true_class_index) == int(row.pred_class_index)]
    wrong_by_confidence = sorted(wrong, key=lambda row: float(row.confidence), reverse=True)
    lowest_confidence_correct = sorted(correct, key=lambda row: float(row.confidence))
    misclassified = wrong_by_confidence[:limit]
    highest_confidence_wrong = wrong_by_confidence[:limit]
    return {
        "misclassified": [_as_prediction_payload(row) for row in misclassified],
        "lowest_confidence_correct": [_as_prediction_payload(row) for row in lowest_confidence_correct[:limit]],
        "highest_confidence_wrong": [_as_prediction_payload(row) for row in highest_confidence_wrong],
    }


def write_classification_evaluation(
    storage: ExperimentStorage,
    *,
    project_id: str,
    experiment_id: str,
    attempt: int,
    class_order: list[int],
    class_names: list[str],
    evaluation: ClassifierEvaluation,
    sample_bucket_limit: int = SAMPLE_BUCKET_LIMIT,
) -> None:
    computed_at = utc_now_iso()
    class_id_to_index = {str(class_id): index for index, class_id in enumerate(class_order)}
    per_class_rows: list[dict[str, Any]] = []
    for row in evaluation.per_class:
        class_index = int(row.class_index)
        class_id = class_order[class_index] if 0 <= class_index < len(class_order) else class_index
        class_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_id}"
        per_class_rows.append(
            {
                "class_index": class_index,
                "class_id": int(class_id),
                "name": class_name,
                "precision": float(row.precision),
                "recall": float(row.recall),
                "f1": float(row.f1),
                "support": int(row.support),
            }
        )

    sample_limit = max(1, int(sample_bucket_limit))
    prediction_rows = [_as_prediction_payload(row) for row in evaluation.predictions]
    evaluation_payload = {
        "schema_version": "1",
        "task": "classification",
        "computed_at": computed_at,
        "split": "val",
        "num_samples": len(prediction_rows),
        "classes": {
            "class_order": [int(class_id) for class_id in class_order],
            "class_names": [str(name) for name in class_names],
            "id_to_index": class_id_to_index,
        },
        "overall": {
            "accuracy": float(evaluation.accuracy),
            "macro_f1": float(evaluation.macro_f1),
            "macro_precision": float(evaluation.macro_precision),
            "macro_recall": float(evaluation.macro_recall),
        },
        "per_class": per_class_rows,
        "confusion_matrix": {
            "matrix": evaluation.confusion_matrix,
            "normalized_by": "none",
            "labels": {"axis": "true_rows_pred_cols"},
        },
        "samples": _sample_buckets(evaluation.predictions, limit=sample_limit),
    }

    predictions_meta_payload = {
        "schema_version": "1",
        "attempt": int(attempt),
        "num_samples": len(prediction_rows),
        "task": "classification",
        "split": "val",
        "computed_at": computed_at,
    }

    # Persist run-attempt artifacts.
    run_evaluation_path = storage.evaluation_path(project_id, experiment_id, attempt)
    run_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    run_evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")

    run_predictions_path = storage.predictions_path(project_id, experiment_id, attempt)
    run_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with run_predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    run_predictions_meta_path = storage.predictions_meta_path(project_id, experiment_id, attempt)
    run_predictions_meta_path.parent.mkdir(parents=True, exist_ok=True)
    run_predictions_meta_path.write_text(json.dumps(predictions_meta_payload, indent=2, sort_keys=True), encoding="utf-8")

    # Persist latest mirrors at experiment root.
    latest_evaluation_path = storage.evaluation_path(project_id, experiment_id, None)
    latest_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    latest_evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")

    latest_predictions_path = storage.predictions_path(project_id, experiment_id, None)
    latest_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with latest_predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    latest_predictions_meta_path = storage.predictions_meta_path(project_id, experiment_id, None)
    latest_predictions_meta_path.parent.mkdir(parents=True, exist_ok=True)
    latest_predictions_meta_path.write_text(json.dumps(predictions_meta_payload, indent=2, sort_keys=True), encoding="utf-8")
