from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SegmentationEvaluation:
    mIoU: float
    per_class_iou: list[float] = field(default_factory=list)
    per_class: list[dict[str, Any]] = field(default_factory=list)


@torch.no_grad()
def evaluate_segmentation(
    model: Any,
    val_loader: Any,
    device: Any,
    *,
    num_classes: int,
    class_names: list[str] | None = None,
) -> SegmentationEvaluation:
    """Compute per-class IoU and mIoU over the validation set.

    Uses a confusion matrix approach: confusion[true][pred] += pixel count.
    IoU per class c = TP_c / (TP_c + FP_c + FN_c).
    """
    model.eval()
    num_classes_with_bg = num_classes + 1  # +1 for background (class 0)
    confusion = torch.zeros(num_classes_with_bg, num_classes_with_bg, dtype=torch.long)
    non_blocking = device.type == "cuda"

    for images, masks in val_loader:
        images = images.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        output = model(images)
        if isinstance(output, dict):
            output = output.get("out", list(output.values())[0])
        preds = output.argmax(dim=1)  # (B, H, W)

        preds_flat = preds.view(-1).cpu()
        masks_flat = masks.view(-1).cpu()
        valid = (masks_flat >= 0) & (masks_flat < num_classes_with_bg)
        preds_flat = preds_flat[valid]
        masks_flat = masks_flat[valid]

        indices = masks_flat * num_classes_with_bg + preds_flat
        confusion_flat = torch.bincount(indices, minlength=num_classes_with_bg ** 2)
        confusion += confusion_flat.view(num_classes_with_bg, num_classes_with_bg)

    per_class_iou: list[float] = []
    per_class: list[dict[str, Any]] = []
    for c in range(num_classes_with_bg):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - confusion[c, c])
        fn = float(confusion[c, :].sum() - confusion[c, c])
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class_iou.append(iou)
        name = "background" if c == 0 else (
            class_names[c - 1] if class_names and c - 1 < len(class_names) else f"class_{c}"
        )
        per_class.append({"class_index": c, "name": name, "iou": iou if not torch.isnan(torch.tensor(iou)) else None})

    valid_ious = [v for v in per_class_iou if not (v != v)]  # filter NaN
    mIoU = sum(valid_ious) / max(len(valid_ious), 1)

    return SegmentationEvaluation(
        mIoU=float(mIoU),
        per_class_iou=per_class_iou,
        per_class=per_class,
    )
