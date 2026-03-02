from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = int(labels.size(0))
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())

    if total_examples < 1:
        return 0.0, 0.0
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return float(avg_loss), float(accuracy)

