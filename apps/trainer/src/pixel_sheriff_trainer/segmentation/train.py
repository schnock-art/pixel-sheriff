from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim

from pixel_sheriff_trainer.segmentation.eval import SegmentationEvaluation, evaluate_segmentation


@dataclass
class SegmentationEpochMetrics:
    epoch: int
    train_loss: float
    mIoU: float | None
    lr: float
    epoch_seconds: float
    eta_seconds: float | None
    evaluated: bool


def _backbone_name(model_config: dict[str, Any]) -> str:
    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        return "resnet50"
    backbone = architecture.get("backbone")
    if not isinstance(backbone, dict):
        return "resnet50"
    name = backbone.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    return "resnet50"


def _backbone_pretrained(model_config: dict[str, Any]) -> bool:
    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        return False
    backbone = architecture.get("backbone")
    if not isinstance(backbone, dict):
        return False
    return bool(backbone.get("pretrained"))


def _build_deeplabv3(model_config: dict[str, Any], *, num_classes: int) -> torch.nn.Module:
    import torchvision.models.segmentation as tv_seg
    import torchvision.models as tv_models
    backbone_name = _backbone_name(model_config)
    pretrained = _backbone_pretrained(model_config)
    builders = {
        "resnet50": (tv_seg.deeplabv3_resnet50, tv_models.ResNet50_Weights.DEFAULT),
        "resnet101": (tv_seg.deeplabv3_resnet101, tv_models.ResNet101_Weights.DEFAULT),
    }
    selected = builders.get(backbone_name)
    if selected is None:
        raise ValueError(f"unsupported_backbone:{backbone_name}")
    builder, weights_backbone = selected
    # num_classes includes background (0) + foreground classes
    try:
        model = builder(
            weights=None,
            weights_backbone=weights_backbone if pretrained else None,
            num_classes=num_classes + 1,
        )
    except Exception as exc:
        if pretrained:
            raise ValueError(f"pretrained_weights_unavailable:deeplabv3/{backbone_name}:{exc}") from exc
        raise
    return model


def run_segmentation_training(
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    num_classes: int,
    should_cancel: Callable[[], bool],
    on_epoch: Callable[[SegmentationEpochMetrics], None],
    on_checkpoint: Callable[[str, int, str | None, float | None, dict[str, Any]], None],
    device: torch.device | None = None,
    resume_state: dict[str, Any] | None = None,
    class_names: list[str] | None = None,
) -> tuple[str, SegmentationEvaluation | None]:
    resolved_device = device or torch.device("cpu")
    model = _build_deeplabv3(model_config, num_classes=num_classes)
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

    criterion = nn.CrossEntropyLoss(ignore_index=255)

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

    best_miou: float | None = None
    final_evaluation: SegmentationEvaluation | None = None
    total_epoch_seconds = 0.0

    for epoch in range(start_epoch, epochs + 1):
        if should_cancel():
            return "canceled", None

        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        num_batches = 0
        non_blocking = resolved_device.type == "cuda"

        for images, masks in train_loader:
            if should_cancel():
                return "canceled", None
            images = images.to(resolved_device, non_blocking=non_blocking)
            masks = masks.to(resolved_device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            if isinstance(output, dict):
                output = output.get("out", list(output.values())[0])
            loss = criterion(output, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(num_batches, 1)
        do_eval = (eval_interval <= 1) or epoch == 1 or epoch == epochs or (epoch % eval_interval == 0)
        evaluation: SegmentationEvaluation | None = None
        if do_eval:
            evaluation = evaluate_segmentation(
                model, val_loader, resolved_device,
                num_classes=num_classes, class_names=class_names,
            )
            if epoch == epochs:
                final_evaluation = evaluation

        epoch_seconds = float(time.perf_counter() - epoch_started)
        total_epoch_seconds += epoch_seconds
        avg_epoch_seconds = total_epoch_seconds / max(1, epoch - start_epoch + 1)
        remaining = max(0, epochs - epoch)
        eta_seconds = float(avg_epoch_seconds * remaining) if remaining > 0 else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])

        on_epoch(SegmentationEpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            mIoU=float(evaluation.mIoU) if evaluation is not None else None,
            lr=current_lr,
            epoch_seconds=epoch_seconds,
            eta_seconds=eta_seconds,
            evaluated=evaluation is not None,
        ))

        metrics_payload: dict[str, Any] = {
            "train_loss": train_loss,
            "mIoU": float(evaluation.mIoU) if evaluation is not None else None,
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        }

        should_save_latest = epoch == epochs or (epoch % save_every == 0)
        if should_save_latest:
            on_checkpoint("latest", epoch, "mIoU" if evaluation else None,
                          float(evaluation.mIoU) if evaluation else None, {
                              "epoch": epoch,
                              "model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                              "metrics": metrics_payload,
                          })

        if keep_best and evaluation is not None:
            if best_miou is None or evaluation.mIoU > best_miou:
                best_miou = evaluation.mIoU
                on_checkpoint("best_metric", epoch, "mIoU", best_miou, {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics_payload,
                })

    return "completed", final_evaluation
