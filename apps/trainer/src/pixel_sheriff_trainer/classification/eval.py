from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class PredictionRow:
    asset_id: str
    relative_path: str
    true_class_index: int
    pred_class_index: int
    confidence: float
    margin: float


@dataclass(frozen=True)
class ClassMetricsRow:
    class_index: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassifierEvaluation:
    avg_loss: float
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetricsRow]
    predictions: list[PredictionRow]


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _zeros_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    include_predictions: bool = False,
    sample_records: list[Any] | None = None,
) -> ClassifierEvaluation:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    resolved_num_classes = max(1, int(num_classes))
    confusion_matrix = _zeros_matrix(resolved_num_classes)
    predictions: list[PredictionRow] = []
    sample_cursor = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        topk = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
        batch_size = int(labels.size(0))
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        labels_cpu = labels.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()

        for batch_index, (true_idx, pred_idx) in enumerate(zip(labels_cpu, preds_cpu)):
            true_class_index = int(true_idx)
            pred_class_index = int(pred_idx)
            if 0 <= true_class_index < resolved_num_classes and 0 <= pred_class_index < resolved_num_classes:
                confusion_matrix[true_class_index][pred_class_index] += 1

            if include_predictions:
                confidence = float(probs[batch_index, pred_class_index].item())
                second_best = float(topk[batch_index, 1].item()) if topk.shape[1] > 1 else 0.0
                margin = float(confidence - second_best)
                row = sample_records[sample_cursor + batch_index] if sample_records and (sample_cursor + batch_index) < len(sample_records) else None
                asset_id = str(getattr(row, "asset_id", "")) if row is not None else ""
                relative_path = str(getattr(row, "relative_path", "")) if row is not None else ""
                predictions.append(
                    PredictionRow(
                        asset_id=asset_id,
                        relative_path=relative_path,
                        true_class_index=true_class_index,
                        pred_class_index=pred_class_index,
                        confidence=confidence,
                        margin=margin,
                    )
                )
        sample_cursor += batch_size

    per_class: list[ClassMetricsRow] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_index in range(resolved_num_classes):
        tp = float(confusion_matrix[class_index][class_index])
        fp = float(sum(confusion_matrix[row_idx][class_index] for row_idx in range(resolved_num_classes) if row_idx != class_index))
        fn = float(sum(confusion_matrix[class_index][col_idx] for col_idx in range(resolved_num_classes) if col_idx != class_index))
        support = int(sum(confusion_matrix[class_index]))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
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

    if total_examples < 1:
        return ClassifierEvaluation(
            avg_loss=0.0,
            accuracy=0.0,
            macro_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            confusion_matrix=confusion_matrix,
            per_class=per_class,
            predictions=predictions,
        )

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    macro_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
    macro_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return ClassifierEvaluation(
        avg_loss=float(avg_loss),
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        confusion_matrix=confusion_matrix,
        per_class=per_class,
        predictions=predictions,
    )
