from __future__ import annotations

from typing import Sequence

from .boxes import compute_iou
from .types import DetectionGroundTruth, DetectionMatchDecision, DetectionPrediction


def match_image_class(
    predictions: Sequence[DetectionPrediction],
    ground_truth: Sequence[DetectionGroundTruth],
    *,
    iou_threshold: float,
) -> list[DetectionMatchDecision]:
    """Match one image / one class greedily in descending confidence order."""

    ground_truth_by_id = {
        str(item.annotation_id): item
        for item in ground_truth
        if isinstance(item.annotation_id, str) and item.annotation_id
    }
    matched_ground_truth_ids: set[str] = set()
    decisions: list[DetectionMatchDecision] = []

    for prediction in predictions:
        prediction_id = str(prediction.prediction_id or "")
        best_iou = 0.0
        best_ground_truth_id: str | None = None

        for target in ground_truth:
            target_id = str(target.annotation_id or "")
            if not target_id:
                continue
            iou = compute_iou(prediction.bbox, target.bbox)
            if iou > best_iou:
                best_iou = iou
                best_ground_truth_id = target_id

        is_true_positive = False
        is_duplicate = False
        matched_ground_truth_id: str | None = None
        if best_ground_truth_id is not None and best_iou >= iou_threshold:
            if best_ground_truth_id not in ground_truth_by_id:
                matched_ground_truth_id = None
            elif best_ground_truth_id not in matched_ground_truth_ids:
                matched_ground_truth_ids.add(best_ground_truth_id)
                matched_ground_truth_id = best_ground_truth_id
                is_true_positive = True
            else:
                is_duplicate = True

        decisions.append(
            DetectionMatchDecision(
                prediction_id=prediction_id,
                ground_truth_id=matched_ground_truth_id,
                iou=best_iou if matched_ground_truth_id is not None else 0.0,
                is_true_positive=is_true_positive,
                is_duplicate=is_duplicate,
                best_same_class_iou=best_iou,
                best_same_class_ground_truth_id=best_ground_truth_id,
            )
        )
    return decisions
