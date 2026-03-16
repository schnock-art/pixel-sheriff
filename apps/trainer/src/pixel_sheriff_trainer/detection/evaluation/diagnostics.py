from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from .boxes import compute_iou
from .types import (
    DetectionDiagnostics,
    DetectionGroundTruth,
    DetectionGroundTruthResult,
    DetectionImageResult,
    DetectionMatchDecision,
    DetectionMatchedPair,
    DetectionPrediction,
    DetectionPredictionResult,
    DetectionTrace,
    DetectionTraceRow,
)


def build_detection_diagnostics(
    *,
    all_predictions: Sequence[DetectionPrediction],
    active_predictions: Sequence[DetectionPrediction],
    filtered_statuses: dict[str, str],
    prediction_ranks: dict[str, int],
    ground_truth: Sequence[DetectionGroundTruth],
    class_order: Sequence[str],
    class_names: Sequence[str],
    match_decisions: dict[str, DetectionMatchDecision],
    diagnostics_iou_threshold: float,
    confidence_traces: Sequence[DetectionTrace],
) -> DetectionDiagnostics:
    """Build structured explainability artifacts for one diagnostics threshold."""

    ground_truth_by_image: dict[str, list[DetectionGroundTruth]] = defaultdict(list)
    for item in ground_truth:
        ground_truth_by_image[item.image_id].append(item)

    prediction_rows: list[DetectionPredictionResult] = []
    per_image: list[DetectionImageResult] = []
    unmatched_ground_truth_rows: list[DetectionGroundTruthResult] = []
    matched_ground_truth_ids = {
        decision.ground_truth_id
        for decision in match_decisions.values()
        if decision.is_true_positive and isinstance(decision.ground_truth_id, str)
    }

    for image_id in sorted({*ground_truth_by_image.keys(), *{item.image_id for item in active_predictions}, *{item.image_id for item in all_predictions}}):
        image_predictions = [item for item in all_predictions if item.image_id == image_id]
        image_predictions = sorted(
            image_predictions,
            key=lambda item: (-float(item.score), int(prediction_ranks.get(str(item.prediction_id or ""), 0))),
        )
        image_ground_truth = sorted(
            ground_truth_by_image.get(image_id, []),
            key=lambda item: (item.class_index, str(item.annotation_id or "")),
        )

        matched_pairs: list[DetectionMatchedPair] = []
        image_prediction_rows: list[DetectionPredictionResult] = []
        unmatched_ground_truths: list[DetectionGroundTruthResult] = []

        for prediction in image_predictions:
            prediction_id = str(prediction.prediction_id or "")
            rank = int(prediction_ranks.get(prediction_id, 0))
            filtered_status = filtered_statuses.get(prediction_id)
            class_index = int(prediction.class_index)
            class_id = class_order[class_index] if 0 <= class_index < len(class_order) else str(class_index)
            class_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_index}"

            best_any_iou = 0.0
            best_any_ground_truth_id: str | None = None
            best_any_ground_truth_class_index: int | None = None
            for target in image_ground_truth:
                target_id = str(target.annotation_id or "")
                if not target_id:
                    continue
                iou = compute_iou(prediction.bbox, target.bbox)
                if iou > best_any_iou:
                    best_any_iou = iou
                    best_any_ground_truth_id = target_id
                    best_any_ground_truth_class_index = int(target.class_index)

            decision = match_decisions.get(prediction_id)
            matched_ground_truth_id: str | None = None
            matched_iou: float | None = None
            best_same_iou: float | None = None
            best_same_ground_truth_id: str | None = None
            if decision is not None:
                matched_ground_truth_id = decision.ground_truth_id
                matched_iou = float(decision.iou) if decision.is_true_positive else None
                best_same_iou = float(decision.best_same_class_iou)
                best_same_ground_truth_id = decision.best_same_class_ground_truth_id

            if filtered_status == "low_score_filtered":
                status = "low_score_filtered"
                reason = "low_score_filtered"
            elif filtered_status == "max_detections_filtered":
                status = "max_detections_filtered"
                reason = "max_detections_filtered"
            elif decision is not None and decision.is_true_positive:
                status = "matched_tp"
                reason = "matched_tp"
                if matched_ground_truth_id is not None and matched_iou is not None:
                    matched_pairs.append(
                        DetectionMatchedPair(
                            image_id=image_id,
                            prediction_id=prediction_id,
                            ground_truth_id=matched_ground_truth_id,
                            class_index=class_index,
                            class_id=class_id,
                            name=class_name,
                            score=float(prediction.score),
                            iou=float(matched_iou),
                        )
                    )
            elif decision is not None and decision.is_duplicate:
                status = "duplicate_fp"
                reason = "duplicate_fp"
            elif (
                best_any_ground_truth_id is not None
                and best_any_ground_truth_class_index is not None
                and best_any_ground_truth_class_index != class_index
                and best_any_iou >= diagnostics_iou_threshold
            ):
                status = "class_fp"
                reason = "class_fp"
            elif isinstance(best_same_iou, float) and best_same_iou > 0.0:
                status = "localization_fp"
                reason = "localization_fp"
            else:
                status = "background_fp"
                reason = "background_fp"

            row = DetectionPredictionResult(
                prediction_id=prediction_id,
                image_id=image_id,
                asset_id=str(prediction.asset_id or ""),
                relative_path=str(prediction.relative_path or ""),
                class_index=class_index,
                class_id=class_id,
                name=class_name,
                bbox=tuple(float(value) for value in prediction.bbox),
                score=float(prediction.score),
                status=status,
                reason=reason,
                rank=rank,
                matched_ground_truth_id=matched_ground_truth_id,
                matched_iou=matched_iou,
                best_same_class_iou=best_same_iou,
                best_same_class_ground_truth_id=best_same_ground_truth_id,
                best_any_iou=float(best_any_iou) if best_any_iou > 0.0 else None,
                best_any_ground_truth_id=best_any_ground_truth_id,
                best_any_ground_truth_class_index=best_any_ground_truth_class_index,
            )
            prediction_rows.append(row)
            image_prediction_rows.append(row)

        for target in image_ground_truth:
            annotation_id = str(target.annotation_id or "")
            if not annotation_id or annotation_id in matched_ground_truth_ids:
                continue
            class_index = int(target.class_index)
            class_id = class_order[class_index] if 0 <= class_index < len(class_order) else str(class_index)
            class_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_index}"
            row = DetectionGroundTruthResult(
                annotation_id=annotation_id,
                image_id=image_id,
                asset_id=str(target.asset_id or ""),
                relative_path=str(target.relative_path or ""),
                class_index=class_index,
                class_id=class_id,
                name=class_name,
                bbox=tuple(float(value) for value in target.bbox),
                area=float(target.area or 0.0),
            )
            unmatched_ground_truths.append(row)
            unmatched_ground_truth_rows.append(row)

        asset_id = next((str(item.asset_id or "") for item in image_predictions if item.asset_id), "")
        if not asset_id:
            asset_id = next((str(item.asset_id or "") for item in image_ground_truth if item.asset_id), "")
        relative_path = next((str(item.relative_path or "") for item in image_predictions if item.relative_path), "")
        if not relative_path:
            relative_path = next((str(item.relative_path or "") for item in image_ground_truth if item.relative_path), "")

        per_image.append(
            DetectionImageResult(
                image_id=image_id,
                asset_id=asset_id,
                relative_path=relative_path,
                prediction_count=len(image_predictions),
                ground_truth_count=len(image_ground_truth),
                predictions=image_prediction_rows,
                matched_pairs=matched_pairs,
                unmatched_ground_truths=unmatched_ground_truths,
            )
        )

    prediction_rows_by_id = {row.prediction_id: row for row in prediction_rows}
    resolved_traces: list[DetectionTrace] = []
    for trace in confidence_traces:
        resolved_rows: list[DetectionTraceRow] = []
        for row in trace.rows:
            prediction_row = prediction_rows_by_id.get(row.prediction_id)
            if prediction_row is None:
                resolved_rows.append(row)
                continue
            resolved_rows.append(
                replace(
                    row,
                    status=prediction_row.status,
                    reason=prediction_row.reason,
                    matched_ground_truth_id=prediction_row.matched_ground_truth_id,
                    iou=prediction_row.matched_iou,
                )
            )
        resolved_traces.append(
            DetectionTrace(
                class_index=trace.class_index,
                class_id=trace.class_id,
                name=trace.name,
                iou_threshold=trace.iou_threshold,
                rows=resolved_rows,
            )
        )

    return DetectionDiagnostics(
        per_image=per_image,
        prediction_rows=prediction_rows,
        unmatched_ground_truths=unmatched_ground_truth_rows,
        confidence_traces=resolved_traces,
    )
