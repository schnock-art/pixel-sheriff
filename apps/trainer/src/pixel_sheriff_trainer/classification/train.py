from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pixel_sheriff_ml.model_factory import build_resnet_classifier
from pixel_sheriff_trainer.classification.eval import ClassifierEvaluation, evaluate_classifier


def _build_classifier(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    return build_resnet_classifier(model_config, num_classes_override=num_classes)


def _resolve_device(training_config: dict[str, Any]) -> torch.device:
    runtime = training_config.get("runtime")
    requested = "auto"
    if isinstance(runtime, dict) and isinstance(runtime.get("device"), str):
        requested = str(runtime["device"]).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_macro_precision: float
    val_macro_recall: float


def run_training(
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    num_classes: int,
    should_cancel: Callable[[], bool],
    on_epoch: Callable[[EpochMetrics], None],
    on_checkpoint: Callable[[str, int, str | None, float | None, dict[str, Any]], None],
) -> tuple[str, ClassifierEvaluation | None]:
    model = _build_classifier(model_config, num_classes)
    device = _resolve_device(training_config)
    model.to(device)

    optimizer_cfg = training_config.get("optimizer")
    if not isinstance(optimizer_cfg, dict):
        optimizer_cfg = {}
    lr = float(optimizer_cfg.get("lr") or 0.001)
    weight_decay = float(optimizer_cfg.get("weight_decay") or 0.0)
    optimizer_type = str(optimizer_cfg.get("type") or "adam").lower()
    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_cfg = training_config.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        scheduler_cfg = {}
    scheduler_type = str(scheduler_cfg.get("type") or "none").lower()
    if scheduler_type == "step":
        params = scheduler_cfg.get("params")
        if not isinstance(params, dict):
            params = {}
        step_size = int(params.get("step_size") or max(1, int(training_config.get("epochs") or 1) // 3))
        gamma = float(params.get("gamma") or 0.1)
        scheduler: Any = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(training_config.get("epochs") or 1)))
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    epochs = max(1, int(training_config.get("epochs") or 1))
    use_amp = str(training_config.get("precision") or "fp32") == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = None
    advanced = training_config.get("advanced")
    if isinstance(advanced, dict) and advanced.get("grad_clip_norm") is not None:
        try:
            grad_clip_norm = float(advanced.get("grad_clip_norm"))
        except (TypeError, ValueError):
            grad_clip_norm = None

    effective_batch_size = int(training_config.get("batch_size") or 1)

    def _freeze_batch_norm_layers() -> None:
        # Small-batch local runs can hit BatchNorm shape assertions (N=1).
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    best_loss = None
    best_metric = None
    final_evaluation: ClassifierEvaluation | None = None
    val_sample_records = getattr(getattr(val_loader, "dataset", None), "samples", None)
    if not isinstance(val_sample_records, list):
        val_sample_records = None
    for epoch in range(1, epochs + 1):
        if should_cancel():
            return "canceled", None

        model.train()
        if effective_batch_size < 2:
            _freeze_batch_norm_layers()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(labels.size(0))
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size

        if scheduler is not None:
            scheduler.step()
        train_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        evaluation = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
            include_predictions=epoch == epochs,
            sample_records=val_sample_records,
        )
        val_loss = float(evaluation.avg_loss)
        val_accuracy = float(evaluation.accuracy)
        if epoch == epochs:
            final_evaluation = evaluation

        on_epoch(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_macro_f1=float(evaluation.macro_f1),
                val_macro_precision=float(evaluation.macro_precision),
                val_macro_recall=float(evaluation.macro_recall),
            )
        )
        model_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": {"train_loss": float(train_loss), "val_loss": float(val_loss), "val_accuracy": float(val_accuracy)},
        }
        on_checkpoint("latest", epoch, "val_accuracy", float(val_accuracy), model_state)

        if best_loss is None or val_loss < best_loss:
            best_loss = float(val_loss)
            on_checkpoint("best_loss", epoch, "val_loss", best_loss, model_state)

        if best_metric is None or val_accuracy > best_metric:
            best_metric = float(val_accuracy)
            on_checkpoint("best_metric", epoch, "val_accuracy", best_metric, model_state)

    return "completed", final_evaluation
