from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from .boxes import bbox_area, normalize_bbox, size_bucket_for_area
from .diagnostics import build_detection_diagnostics
from .matching import match_image_class
from .types import (
    DetectionEvaluation,
    DetectionGroundTruth,
    DetectionMatchDecision,
    DetectionOverallMetrics,
    DetectionPerClassMetrics,
    DetectionPRCurve,
    DetectionPrediction,
    DetectionSizeBucketSummary,
    DetectionTrace,
    DetectionTraceRow,
)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _normalize_predictions(
    predictions: Iterable[DetectionPrediction],
    *,
    box_format: str,
) -> list[DetectionPrediction]:
    normalized: list[DetectionPrediction] = []
    for index, item in enumerate(predictions):
        prediction_id = str(item.prediction_id or f"pred-{item.image_id}-{index}")
        normalized.append(
            replace(
                item,
                prediction_id=prediction_id,
                bbox=normalize_bbox(item.bbox, box_format=box_format),
            )
        )
    return normalized


def _normalize_ground_truth(
    ground_truth: Iterable[DetectionGroundTruth],
    *,
    box_format: str,
) -> list[DetectionGroundTruth]:
    normalized: list[DetectionGroundTruth] = []
    for index, item in enumerate(ground_truth):
        annotation_id = str(item.annotation_id or f"gt-{item.image_id}-{index}")
        bbox = normalize_bbox(item.bbox, box_format=box_format)
        area = float(item.area) if isinstance(item.area, (int, float)) else bbox_area(bbox)
        normalized.append(
            replace(
                item,
                annotation_id=annotation_id,
                bbox=bbox,
                area=area,
            )
        )
    return normalized


def _sorted_predictions_for_image(predictions: Sequence[DetectionPrediction], order_lookup: dict[str, int]) -> list[DetectionPrediction]:
    return sorted(
        predictions,
        key=lambda item: (-float(item.score), int(order_lookup.get(str(item.prediction_id or ""), 0))),
    )


def _apply_prediction_filters(
    predictions: Sequence[DetectionPrediction],
    *,
    score_threshold: float | None,
    max_detections_per_image: int | None,
    order_lookup: dict[str, int],
) -> tuple[list[DetectionPrediction], dict[str, str], dict[str, int]]:
    filtered_statuses: dict[str, str] = {}
    prediction_ranks: dict[str, int] = {}
    active_predictions: list[DetectionPrediction] = []

    predictions_by_image: dict[str, list[DetectionPrediction]] = defaultdict(list)
    for item in predictions:
        predictions_by_image[item.image_id].append(item)

    for image_id in sorted(predictions_by_image):
        sorted_predictions = _sorted_predictions_for_image(predictions_by_image[image_id], order_lookup)
        kept_for_image = 0
        for rank, prediction in enumerate(sorted_predictions, start=1):
            prediction_id = str(prediction.prediction_id or "")
            prediction_ranks[prediction_id] = rank
            if score_threshold is not None and float(prediction.score) < float(score_threshold):
                filtered_statuses[prediction_id] = "low_score_filtered"
                continue
            if isinstance(max_detections_per_image, int) and max_detections_per_image >= 0 and kept_for_image >= max_detections_per_image:
                filtered_statuses[prediction_id] = "max_detections_filtered"
                continue
            kept_for_image += 1
            active_predictions.append(prediction)

    return active_predictions, filtered_statuses, prediction_ranks


def _precision_recall_arrays(tp_flags: Sequence[int], fp_flags: Sequence[int], *, support: int) -> tuple[list[float], list[float]]:
    precision: list[float] = []
    recall: list[float] = []
    cumulative_tp = 0
    cumulative_fp = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags, strict=False):
        cumulative_tp += int(tp_flag)
        cumulative_fp += int(fp_flag)
        precision.append(_safe_div(cumulative_tp, cumulative_tp + cumulative_fp))
        recall.append(_safe_div(cumulative_tp, support))
    return precision, recall


def _precision_envelope(precision: Sequence[float]) -> list[float]:
    envelope = list(float(value) for value in precision)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    return envelope


def _average_precision(precision_envelope: Sequence[float], recall: Sequence[float]) -> float:
    if not precision_envelope or not recall:
        return 0.0
    previous_recall = 0.0
    ap = 0.0
    for precision_value, recall_value in zip(precision_envelope, recall, strict=False):
        delta_recall = max(0.0, float(recall_value) - previous_recall)
        ap += float(precision_value) * delta_recall
        previous_recall = float(recall_value)
    return float(ap)


def _class_ap_for_threshold(
    *,
    class_index: int,
    predictions: Sequence[DetectionPrediction],
    ground_truth: Sequence[DetectionGroundTruth],
    iou_threshold: float,
    class_order: Sequence[str],
    class_names: Sequence[str],
    order_lookup: dict[str, int],
) -> tuple[float | None, DetectionPRCurve, dict[str, DetectionMatchDecision], DetectionTrace]:
    class_id = class_order[class_index] if 0 <= class_index < len(class_order) else str(class_index)
    class_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_index}"

    class_predictions = [item for item in predictions if int(item.class_index) == class_index]
    class_ground_truth = [item for item in ground_truth if int(item.class_index) == class_index]
    support = len(class_ground_truth)

    predictions_by_image: dict[str, list[DetectionPrediction]] = defaultdict(list)
    ground_truth_by_image: dict[str, list[DetectionGroundTruth]] = defaultdict(list)
    for item in class_predictions:
        predictions_by_image[item.image_id].append(item)
    for item in class_ground_truth:
        ground_truth_by_image[item.image_id].append(item)

    match_decisions: dict[str, DetectionMatchDecision] = {}
    for image_id in sorted({*predictions_by_image.keys(), *ground_truth_by_image.keys()}):
        image_predictions = _sorted_predictions_for_image(predictions_by_image.get(image_id, []), order_lookup)
        image_ground_truth = sorted(
            ground_truth_by_image.get(image_id, []),
            key=lambda item: str(item.annotation_id or ""),
        )
        for decision in match_image_class(image_predictions, image_ground_truth, iou_threshold=iou_threshold):
            match_decisions[decision.prediction_id] = decision

    sorted_predictions = _sorted_predictions_for_image(class_predictions, order_lookup)
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    scores: list[float] = []
    matched_ground_truth_ids: list[str | None] = []
    matched_ious: list[float | None] = []

    for prediction in sorted_predictions:
        prediction_id = str(prediction.prediction_id or "")
        decision = match_decisions.get(prediction_id)
        is_true_positive = bool(decision.is_true_positive) if decision is not None else False
        tp_flags.append(1 if is_true_positive else 0)
        fp_flags.append(0 if is_true_positive else 1)
        scores.append(float(prediction.score))
        matched_ground_truth_ids.append(decision.ground_truth_id if decision is not None else None)
        matched_ious.append(float(decision.iou) if decision is not None and decision.is_true_positive else None)

    precision, recall = _precision_recall_arrays(tp_flags, fp_flags, support=support)
    precision_envelope = _precision_envelope(precision)
    ap = _average_precision(precision_envelope, recall) if support > 0 else None

    curve = DetectionPRCurve(
        class_index=class_index,
        class_id=class_id,
        name=class_name,
        iou_threshold=float(iou_threshold),
        scores=scores,
        precision=precision,
        recall=recall,
        precision_envelope=precision_envelope,
    )

    trace_rows: list[DetectionTraceRow] = []
    cumulative_tp = 0
    cumulative_fp = 0
    for prediction, tp_flag, fp_flag, precision_value, recall_value, matched_ground_truth_id, matched_iou in zip(
        sorted_predictions,
        tp_flags,
        fp_flags,
        precision,
        recall,
        matched_ground_truth_ids,
        matched_ious,
        strict=False,
    ):
        cumulative_tp += int(tp_flag)
        cumulative_fp += int(fp_flag)
        status = "matched_tp" if tp_flag == 1 else "false_positive"
        trace_rows.append(
            DetectionTraceRow(
                prediction_id=str(prediction.prediction_id or ""),
                image_id=prediction.image_id,
                score=float(prediction.score),
                status=status,
                reason=status,
                cumulative_tp=cumulative_tp,
                cumulative_fp=cumulative_fp,
                precision=float(precision_value),
                recall=float(recall_value),
                matched_ground_truth_id=matched_ground_truth_id,
                iou=matched_iou,
            )
        )

    trace = DetectionTrace(
        class_index=class_index,
        class_id=class_id,
        name=class_name,
        iou_threshold=float(iou_threshold),
        rows=trace_rows,
    )
    return ap, curve, match_decisions, trace


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _build_bucket_summaries(
    *,
    active_predictions: Sequence[DetectionPrediction],
    ground_truth: Sequence[DetectionGroundTruth],
    class_order: Sequence[str],
    class_names: Sequence[str],
    iou_thresholds: Sequence[float],
    diagnostics_iou_threshold: float,
    order_lookup: dict[str, int],
) -> tuple[list[DetectionSizeBucketSummary], dict[str, float | None]]:
    bucket_summaries: list[DetectionSizeBucketSummary] = []
    bucket_map: dict[str, float | None] = {"small": None, "medium": None, "large": None}
    for bucket_name in ("small", "medium", "large"):
        bucket_predictions = [
            item for item in active_predictions if size_bucket_for_area(bbox_area(item.bbox)) == bucket_name
        ]
        bucket_ground_truth = [
            item for item in ground_truth if size_bucket_for_area(float(item.area or 0.0)) == bucket_name
        ]
        if not bucket_ground_truth:
            bucket_summaries.append(
                DetectionSizeBucketSummary(
                    name=bucket_name,
                    ground_truth_count=0,
                    prediction_count=len(bucket_predictions),
                    ap50=None,
                    map_50_95=None,
                    precision=0.0,
                    recall=0.0,
                )
            )
            continue

        bucket_evaluation, _bucket_match_decisions, _bucket_traces = _evaluate_active_predictions(
            active_predictions=bucket_predictions,
            ground_truth=bucket_ground_truth,
            class_order=class_order,
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            diagnostics_iou_threshold=diagnostics_iou_threshold,
            order_lookup=order_lookup,
            include_size_buckets=False,
        )
        bucket_summaries.append(
            DetectionSizeBucketSummary(
                name=bucket_name,
                ground_truth_count=len(bucket_ground_truth),
                prediction_count=len(bucket_predictions),
                ap50=float(bucket_evaluation.mAP50),
                map_50_95=float(bucket_evaluation.mAP50_95),
                precision=float(bucket_evaluation.overall.precision if bucket_evaluation.overall is not None else 0.0),
                recall=float(bucket_evaluation.overall.recall if bucket_evaluation.overall is not None else 0.0),
            )
        )
        bucket_map[bucket_name] = float(bucket_evaluation.mAP50_95)
    return bucket_summaries, bucket_map


def _evaluate_active_predictions(
    *,
    active_predictions: Sequence[DetectionPrediction],
    ground_truth: Sequence[DetectionGroundTruth],
    class_order: Sequence[str],
    class_names: Sequence[str],
    iou_thresholds: Sequence[float],
    diagnostics_iou_threshold: float,
    order_lookup: dict[str, int],
    include_size_buckets: bool,
) -> tuple[DetectionEvaluation, dict[str, DetectionMatchDecision], list[DetectionTrace]]:
    ap_by_class_and_threshold: dict[int, dict[str, float | None]] = {}
    pr_curves: list[DetectionPRCurve] = []
    traces_by_threshold: dict[str, list[DetectionTrace]] = defaultdict(list)

    supported_class_aps_at_50: list[float] = []
    supported_class_maps: list[float] = []

    diagnostics_match_decisions: dict[str, DetectionMatchDecision] = {}

    for class_index in range(len(class_order)):
        threshold_aps: dict[str, float | None] = {}
        class_ap_values: list[float] = []
        for iou_threshold in iou_thresholds:
            ap, curve, match_decisions, trace = _class_ap_for_threshold(
                class_index=class_index,
                predictions=active_predictions,
                ground_truth=ground_truth,
                iou_threshold=iou_threshold,
                class_order=class_order,
                class_names=class_names,
                order_lookup=order_lookup,
            )
            threshold_key = _threshold_key(iou_threshold)
            threshold_aps[threshold_key] = ap
            pr_curves.append(curve)
            traces_by_threshold[threshold_key].append(trace)
            if ap is not None:
                class_ap_values.append(float(ap))
                if abs(float(iou_threshold) - 0.50) < 1e-6:
                    supported_class_aps_at_50.append(float(ap))
            if abs(float(iou_threshold) - float(diagnostics_iou_threshold)) < 1e-6:
                diagnostics_match_decisions.update(match_decisions)

        ap_by_class_and_threshold[class_index] = threshold_aps
        if class_ap_values:
            supported_class_maps.append(sum(class_ap_values) / len(class_ap_values))

    diagnostics_threshold_key = _threshold_key(diagnostics_iou_threshold)
    diagnostic_traces = list(traces_by_threshold.get(diagnostics_threshold_key, []))

    diagnostics_tp_ids = {
        prediction_id
        for prediction_id, decision in diagnostics_match_decisions.items()
        if decision.is_true_positive
    }
    duplicate_prediction_ids = {
        prediction_id
        for prediction_id, decision in diagnostics_match_decisions.items()
        if decision.is_duplicate
    }
    matched_ious = [
        float(decision.iou)
        for decision in diagnostics_match_decisions.values()
        if decision.is_true_positive
    ]

    per_class_rows: list[DetectionPerClassMetrics] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_duplicate_fp = 0

    for class_index in range(len(class_order)):
        class_id = class_order[class_index]
        class_name = class_names[class_index] if class_index < len(class_names) else f"class_{class_id}"
        class_predictions = [item for item in active_predictions if int(item.class_index) == class_index]
        class_ground_truth = [item for item in ground_truth if int(item.class_index) == class_index]
        support = len(class_ground_truth)

        class_tp = sum(
            1
            for item in class_predictions
            if str(item.prediction_id or "") in diagnostics_tp_ids
        )
        class_duplicate_fp = sum(
            1
            for item in class_predictions
            if str(item.prediction_id or "") in duplicate_prediction_ids
        )
        class_fp = max(0, len(class_predictions) - class_tp)
        class_fn = max(0, support - class_tp)
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn
        total_duplicate_fp += class_duplicate_fp

        precision = _safe_div(class_tp, class_tp + class_fp)
        recall = _safe_div(class_tp, class_tp + class_fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        class_matched_ious = [
            float(decision.iou)
            for prediction_id, decision in diagnostics_match_decisions.items()
            if decision.is_true_positive
            and prediction_id
            in {
                str(item.prediction_id or "")
                for item in class_predictions
            }
        ]
        ap_by_iou = ap_by_class_and_threshold.get(class_index, {})
        map_50_95 = _mean_or_none([value for value in ap_by_iou.values() if isinstance(value, float)])
        per_class_rows.append(
            DetectionPerClassMetrics(
                class_index=class_index,
                class_id=str(class_id),
                name=class_name,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                support=support,
                ap50=ap_by_iou.get("0.50"),
                ap75=ap_by_iou.get("0.75"),
                map_50_95=map_50_95,
                ap_by_iou=dict(ap_by_iou),
                tp=class_tp,
                fp=class_fp,
                fn=class_fn,
                duplicate_fp=class_duplicate_fp,
                matched_mean_iou=_mean_or_none(class_matched_ious),
            )
        )

    size_buckets: list[DetectionSizeBucketSummary] = []
    bucket_ap_map = {"small": None, "medium": None, "large": None}
    if include_size_buckets:
        size_buckets, bucket_ap_map = _build_bucket_summaries(
            active_predictions=active_predictions,
            ground_truth=ground_truth,
            class_order=class_order,
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            diagnostics_iou_threshold=diagnostics_iou_threshold,
            order_lookup=order_lookup,
        )

    mAP50 = float(sum(supported_class_aps_at_50) / len(supported_class_aps_at_50)) if supported_class_aps_at_50 else 0.0
    mAP50_95 = float(sum(supported_class_maps) / len(supported_class_maps)) if supported_class_maps else 0.0
    overall = DetectionOverallMetrics(
        mAP50=mAP50,
        mAP50_95=mAP50_95,
        precision=_safe_div(total_tp, total_tp + total_fp),
        recall=_safe_div(total_tp, total_tp + total_fn),
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        duplicate_fp=total_duplicate_fp,
        matched_mean_iou=_mean_or_none(matched_ious),
        image_count=len({item.image_id for item in ground_truth} | {item.image_id for item in active_predictions}),
        prediction_count=len(active_predictions),
        ground_truth_count=len(ground_truth),
        ap_small=bucket_ap_map["small"],
        ap_medium=bucket_ap_map["medium"],
        ap_large=bucket_ap_map["large"],
        size_buckets=size_buckets,
    )
    return DetectionEvaluation(
        mAP50=mAP50,
        mAP50_95=mAP50_95,
        overall=overall,
        per_class=per_class_rows,
        pr_curves=pr_curves,
        diagnostics=None,
        iou_thresholds=[float(value) for value in iou_thresholds],
        diagnostics_iou_threshold=float(diagnostics_iou_threshold),
    ), diagnostics_match_decisions, diagnostic_traces


def evaluate_detection_set(
    predictions: Sequence[DetectionPrediction],
    ground_truth: Sequence[DetectionGroundTruth],
    *,
    class_order: Sequence[str],
    class_names: Sequence[str] | None = None,
    box_format: str = "xyxy",
    iou_thresholds: Sequence[float] = tuple(0.50 + (0.05 * index) for index in range(10)),
    diagnostics_iou_threshold: float = 0.50,
    score_threshold: float | None = None,
    max_detections_per_image: int | None = None,
    include_size_buckets: bool = True,
) -> DetectionEvaluation:
    """Evaluate a dataset split of axis-aligned detections.

    The evaluator uses greedy same-class matching and a recall-step AP
    integration. It intentionally does not implement COCO ignore/crowd rules.
    """

    resolved_class_names = list(class_names or [])
    if len(resolved_class_names) < len(class_order):
        resolved_class_names.extend(
            f"class_{class_order[index]}"
            for index in range(len(resolved_class_names), len(class_order))
        )

    normalized_predictions = _normalize_predictions(predictions, box_format=box_format)
    normalized_ground_truth = _normalize_ground_truth(ground_truth, box_format=box_format)
    order_lookup = {
        str(item.prediction_id or ""): index
        for index, item in enumerate(normalized_predictions)
    }
    active_predictions, filtered_statuses, prediction_ranks = _apply_prediction_filters(
        normalized_predictions,
        score_threshold=score_threshold,
        max_detections_per_image=max_detections_per_image,
        order_lookup=order_lookup,
    )

    metrics_result, diagnostics_match_decisions, diagnostic_traces = _evaluate_active_predictions(
        active_predictions=active_predictions,
        ground_truth=normalized_ground_truth,
        class_order=class_order,
        class_names=resolved_class_names,
        iou_thresholds=iou_thresholds,
        diagnostics_iou_threshold=diagnostics_iou_threshold,
        order_lookup=order_lookup,
        include_size_buckets=include_size_buckets,
    )

    diagnostics = build_detection_diagnostics(
        all_predictions=normalized_predictions,
        active_predictions=active_predictions,
        filtered_statuses=filtered_statuses,
        prediction_ranks=prediction_ranks,
        ground_truth=normalized_ground_truth,
        class_order=class_order,
        class_names=resolved_class_names,
        match_decisions=diagnostics_match_decisions,
        diagnostics_iou_threshold=diagnostics_iou_threshold,
        confidence_traces=diagnostic_traces,
    )
    metrics_result.diagnostics = diagnostics
    metrics_result.score_threshold = score_threshold
    metrics_result.max_detections_per_image = max_detections_per_image
    return metrics_result
