from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionEvaluation:
    mAP50: float
    mAP50_95: float
    per_class: list[dict[str, Any]] = field(default_factory=list)


def evaluate_detection(
    model: Any,
    val_loader: Any,
    device: Any,
    *,
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> DetectionEvaluation:
    """Compute mAP@50 and mAP@50:95 for a detection model on the validation set.

    Uses torchvision box_iou for IoU computation and a custom AP accumulator.
    Falls back to torchmetrics.detection.MeanAveragePrecision if available.
    """
    import torch

    if iou_thresholds is None:
        iou_thresholds = [0.50 + 0.05 * i for i in range(10)]  # 0.50:0.95

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                preds = model(images)
                metric.update(preds, [{k: v.to("cpu") for k, v in t.items()} for t in targets])
        result = metric.compute()
        return DetectionEvaluation(
            mAP50=float(result.get("map_50", 0.0)),
            mAP50_95=float(result.get("map", 0.0)),
        )
    except ImportError:
        pass

    # Fallback: manual AP computation
    from torchvision.ops import box_iou

    all_preds: list[dict[str, Any]] = []
    all_targets: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            all_preds.extend([{k: v.detach().cpu() for k, v in p.items()} for p in preds])
            all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    def compute_ap(iou_threshold: float) -> float:
        tp_list: list[float] = []
        fp_list: list[float] = []
        scores_list: list[float] = []
        n_gt = sum(len(t["boxes"]) for t in all_targets)
        if n_gt == 0:
            return 0.0
        matched: list[set[int]] = [set() for _ in all_targets]

        for img_idx, (pred, tgt) in enumerate(zip(all_preds, all_targets)):
            pred_boxes = pred.get("boxes", torch.zeros((0, 4)))
            pred_scores = pred.get("scores", torch.zeros(0))
            gt_boxes = tgt.get("boxes", torch.zeros((0, 4)))
            order = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            for i in range(len(pred_boxes)):
                score = float(pred_scores[i])
                scores_list.append(score)
                if len(gt_boxes) == 0:
                    fp_list.append(1.0)
                    tp_list.append(0.0)
                    continue
                iou = box_iou(pred_boxes[i:i+1], gt_boxes)
                best_iou, best_gt = iou.max(dim=1)
                best_iou_val = float(best_iou[0])
                best_gt_idx = int(best_gt[0])
                if best_iou_val >= iou_threshold and best_gt_idx not in matched[img_idx]:
                    tp_list.append(1.0)
                    fp_list.append(0.0)
                    matched[img_idx].add(best_gt_idx)
                else:
                    tp_list.append(0.0)
                    fp_list.append(1.0)

        if not scores_list:
            return 0.0

        order = sorted(range(len(scores_list)), key=lambda i: -scores_list[i])
        tp_sorted = [tp_list[i] for i in order]
        fp_sorted = [fp_list[i] for i in order]

        cum_tp = [sum(tp_sorted[:i+1]) for i in range(len(tp_sorted))]
        cum_fp = [sum(fp_sorted[:i+1]) for i in range(len(fp_sorted))]

        precisions = [tp / (tp + fp) if (tp + fp) > 0 else 0.0 for tp, fp in zip(cum_tp, cum_fp)]
        recalls = [tp / n_gt for tp in cum_tp]

        # 11-point interpolated AP
        ap = 0.0
        for thr in [i / 10 for i in range(11)]:
            p_at_r = max((p for p, r in zip(precisions, recalls) if r >= thr), default=0.0)
            ap += p_at_r / 11
        return ap

    map50 = compute_ap(0.50)
    map50_95 = sum(compute_ap(t) for t in iou_thresholds) / max(len(iou_thresholds), 1)
    return DetectionEvaluation(mAP50=map50, mAP50_95=map50_95)
