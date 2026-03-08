from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.optim as optim

from pixel_sheriff_trainer.detection.eval import DetectionEvaluation, evaluate_detection


@dataclass
class DetectionEpochMetrics:
    epoch: int
    train_loss: float
    mAP50: float | None
    mAP50_95: float | None
    lr: float
    epoch_seconds: float
    eta_seconds: float | None
    evaluated: bool


def _build_retinanet(num_classes: int) -> torch.nn.Module:
    import torchvision.models.detection as tv_det
    # num_classes here is foreground classes (background added internally by RetinaNet)
    model = tv_det.retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


def run_detection_training(
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    num_classes: int,
    should_cancel: Callable[[], bool],
    on_epoch: Callable[[DetectionEpochMetrics], None],
    on_checkpoint: Callable[[str, int, str | None, float | None, dict[str, Any]], None],
    device: torch.device | None = None,
    resume_state: dict[str, Any] | None = None,
) -> tuple[str, DetectionEvaluation | None]:
    resolved_device = device or torch.device("cpu")
    model = _build_retinanet(num_classes)
    model.to(resolved_device)

    if resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    optimizer_cfg = training_config.get("optimizer") or {}
    lr = float(optimizer_cfg.get("lr", 0.0001))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0001))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = max(1, int(training_config.get("epochs", 1)))
    scheduler_cfg = training_config.get("scheduler") or {}
    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
    if scheduler_type == "cosine":
        scheduler: Any = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    else:
        scheduler = None

    logging_cfg = training_config.get("logging") or {}
    save_every = max(1, int(logging_cfg.get("save_every_epochs", 1)))
    keep_best = bool(logging_cfg.get("keep_best", True))

    evaluation_cfg = training_config.get("evaluation") or {}
    eval_interval = max(1, int(evaluation_cfg.get("eval_interval_epochs", 1)))

    start_epoch = 1
    if isinstance(resume_state, dict):
        model_state = resume_state.get("model_state_dict")
        if isinstance(model_state, dict):
            model.load_state_dict(model_state)
        optimizer_state = resume_state.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = resume_state.get("scheduler_state_dict")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        resumed_epoch = int(resume_state.get("epoch", 0))
        if resumed_epoch >= 1:
            start_epoch = resumed_epoch + 1

    best_map50: float | None = None
    final_evaluation: DetectionEvaluation | None = None
    total_epoch_seconds = 0.0

    for epoch in range(start_epoch, epochs + 1):
        if should_cancel():
            return "canceled", None

        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        num_batches = 0
        non_blocking = resolved_device.type == "cuda"

        for images, targets in train_loader:
            if should_cancel():
                return "canceled", None
            images = [img.to(resolved_device, non_blocking=non_blocking) for img in images]
            targets = [{k: v.to(resolved_device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(losses.item())
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(num_batches, 1)
        do_eval = (eval_interval <= 1) or epoch == 1 or epoch == epochs or (epoch % eval_interval == 0)
        evaluation: DetectionEvaluation | None = None
        if do_eval:
            evaluation = evaluate_detection(model, val_loader, resolved_device, num_classes=num_classes)
            if epoch == epochs:
                final_evaluation = evaluation

        epoch_seconds = float(time.perf_counter() - epoch_started)
        total_epoch_seconds += epoch_seconds
        avg_epoch_seconds = total_epoch_seconds / max(1, epoch - start_epoch + 1)
        remaining = max(0, epochs - epoch)
        eta_seconds = float(avg_epoch_seconds * remaining) if remaining > 0 else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])

        on_epoch(DetectionEpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            mAP50=float(evaluation.mAP50) if evaluation is not None else None,
            mAP50_95=float(evaluation.mAP50_95) if evaluation is not None else None,
            lr=current_lr,
            epoch_seconds=epoch_seconds,
            eta_seconds=eta_seconds,
            evaluated=evaluation is not None,
        ))

        metrics_payload: dict[str, Any] = {
            "train_loss": train_loss,
            "mAP50": float(evaluation.mAP50) if evaluation is not None else None,
            "mAP50_95": float(evaluation.mAP50_95) if evaluation is not None else None,
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        }

        should_save_latest = epoch == epochs or (epoch % save_every == 0)
        if should_save_latest:
            on_checkpoint("latest", epoch, "mAP50" if evaluation else None,
                          float(evaluation.mAP50) if evaluation else None, {
                              "epoch": epoch,
                              "model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                              "metrics": metrics_payload,
                          })

        if keep_best and evaluation is not None:
            if best_map50 is None or evaluation.mAP50 > best_map50:
                best_map50 = evaluation.mAP50
                on_checkpoint("best_metric", epoch, "mAP50", best_map50, {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics_payload,
                })

    return "completed", final_evaluation
