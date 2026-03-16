from __future__ import annotations

from typing import Any, Sequence

from pixel_sheriff_trainer.detection.evaluation import (
    DetectionEvaluation,
    DetectionGroundTruth,
    DetectionPrediction,
    evaluate_detection_set,
)
from pixel_sheriff_trainer.detection.evaluation.boxes import bbox_xywh_to_xyxy


def _scaled_xyxy_to_original(
    bbox_xyxy: Sequence[float],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return (
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    )


def evaluate_detection(
    model: Any,
    val_loader: Any,
    device: Any,
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
    class_order: Sequence[str] | None = None,
    iou_thresholds: Sequence[float] | None = None,
    diagnostics_iou_threshold: float = 0.50,
    score_threshold: float | None = None,
    max_detections_per_image: int | None = None,
    include_size_buckets: bool = True,
    label_offset: int = 0,
) -> DetectionEvaluation:
    """Evaluate a torchvision detection model against the validation loader.

    The loader is expected to be the project detection dataset. Predictions are
    normalized back to zero-based dataset class indices and original-image pixel
    coordinates before the pure evaluator runs.
    """

    import torch

    if iou_thresholds is None:
        iou_thresholds = [0.50 + (0.05 * index) for index in range(10)]

    dataset = getattr(val_loader, "dataset", None)
    sample_records = list(getattr(dataset, "samples", []) or [])
    annotations_by_image = getattr(dataset, "annotations", {})
    target_width = int(getattr(dataset, "target_width", 0) or 0)
    target_height = int(getattr(dataset, "target_height", 0) or 0)

    if not sample_records or not isinstance(annotations_by_image, dict):
        raise ValueError("detection_eval_sample_metadata_missing")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("detection_eval_target_size_missing")

    predictions: list[DetectionPrediction] = []
    ground_truth: list[DetectionGroundTruth] = []
    sample_cursor = 0
    non_blocking = getattr(device, "type", "") == "cuda"

    model.eval()
    with torch.no_grad():
        for images, _targets in val_loader:
            images = [image.to(device, non_blocking=non_blocking) for image in images]
            batch_predictions = model(images)
            batch_size = len(batch_predictions)
            batch_samples = sample_records[sample_cursor : sample_cursor + batch_size]
            if len(batch_samples) != batch_size:
                raise ValueError("detection_eval_sample_alignment_failed")

            for batch_index, (prediction_output, sample) in enumerate(zip(batch_predictions, batch_samples, strict=False)):
                scale_x = float(sample.width) / max(float(target_width), 1.0)
                scale_y = float(sample.height) / max(float(target_height), 1.0)
                prediction_offset = sample_cursor + batch_index

                raw_boxes = prediction_output.get("boxes")
                raw_scores = prediction_output.get("scores")
                raw_labels = prediction_output.get("labels")
                boxes = raw_boxes.detach().cpu() if raw_boxes is not None else torch.zeros((0, 4), dtype=torch.float32)
                scores = raw_scores.detach().cpu() if raw_scores is not None else torch.zeros((0,), dtype=torch.float32)
                labels = raw_labels.detach().cpu() if raw_labels is not None else torch.zeros((0,), dtype=torch.int64)
                detection_count = min(len(boxes), len(scores), len(labels))

                for detection_index in range(detection_count):
                    class_index = int(labels[detection_index].item()) - int(label_offset)
                    if class_index < 0 or class_index >= int(num_classes):
                        continue
                    predictions.append(
                        DetectionPrediction(
                            image_id=str(sample.image_id),
                            class_index=class_index,
                            bbox=_scaled_xyxy_to_original(
                                boxes[detection_index].tolist(),
                                scale_x=scale_x,
                                scale_y=scale_y,
                            ),
                            score=float(scores[detection_index].item()),
                            prediction_id=f"{sample.image_id}-pred-{prediction_offset}-{detection_index}",
                            asset_id=str(sample.asset_id),
                            relative_path=str(sample.relative_path),
                        )
                    )

                for annotation_index, annotation in enumerate(list(annotations_by_image.get(sample.image_id, []) or [])):
                    bbox = annotation.get("bbox")
                    category_id = annotation.get("category_id")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    if not isinstance(category_id, int):
                        continue
                    class_index = int(category_id) - int(label_offset)
                    if class_index < 0 or class_index >= int(num_classes):
                        continue
                    area_value = annotation.get("area")
                    ground_truth.append(
                        DetectionGroundTruth(
                            image_id=str(sample.image_id),
                            class_index=class_index,
                            bbox=bbox_xywh_to_xyxy(bbox),
                            annotation_id=str(annotation.get("id") or f"{sample.image_id}-ann-{annotation_index}"),
                            asset_id=str(sample.asset_id),
                            relative_path=str(sample.relative_path),
                            area=float(area_value) if isinstance(area_value, (int, float)) else None,
                        )
                    )

            sample_cursor += batch_size

    resolved_class_order = list(class_order or [str(index) for index in range(int(num_classes))])
    resolved_class_names = list(class_names or [f"class_{class_id}" for class_id in resolved_class_order])
    return evaluate_detection_set(
        predictions,
        ground_truth,
        class_order=resolved_class_order,
        class_names=resolved_class_names,
        box_format="xyxy",
        iou_thresholds=iou_thresholds,
        diagnostics_iou_threshold=diagnostics_iou_threshold,
        score_threshold=score_threshold,
        max_detections_per_image=max_detections_per_image,
        include_size_buckets=include_size_buckets,
    )
