from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pixel_sheriff_ml.model_factory import build_classifier_model
from pixel_sheriff_trainer.classification.eval import ClassifierEvaluation, evaluate_classifier
from pixel_sheriff_trainer.training_config import (
    amp_enabled as config_amp_enabled,
    as_bool as _as_bool,
    as_float as _as_float,
    as_int as _as_int,
    build_optimizer,
    build_scheduler,
    grad_clip_norm_from_config,
    resolve_device as shared_resolve_device,
    resolve_runtime_info as shared_resolve_runtime_info,
    resolve_runtime_loader_settings as shared_resolve_runtime_loader_settings,
    RuntimeInfo,
)


def _build_classifier(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    return build_classifier_model(model_config, num_classes_override=num_classes)


def resolve_device(training_config: dict[str, Any]) -> torch.device:
    return shared_resolve_device(training_config)


def resolve_runtime_loader_settings(
    training_config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[int, bool, bool, int, bool, int]:
    settings = shared_resolve_runtime_loader_settings(training_config, device=device)
    return (
        settings.num_workers,
        settings.pin_memory,
        settings.persistent_workers,
        settings.prefetch_factor,
        settings.cache_resized_images,
        settings.max_cached_images,
    )


def resolve_runtime_info(training_config: dict[str, Any], *, device: torch.device) -> RuntimeInfo:
    return shared_resolve_runtime_info(training_config, device=device)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float | None
    val_loss: float | None
    val_accuracy: float | None
    val_macro_f1: float | None
    val_macro_precision: float | None
    val_macro_recall: float | None
    lr: float
    epoch_seconds: float
    eta_seconds: float | None
    evaluated: bool


def _loader_may_emit_small_batch(loader: DataLoader[Any]) -> bool:
    batch_size_raw = getattr(loader, "batch_size", None)
    if not isinstance(batch_size_raw, int):
        return False
    if batch_size_raw < 2:
        return True
    drop_last = bool(getattr(loader, "drop_last", False))
    if drop_last:
        return False
    dataset = getattr(loader, "dataset", None)
    try:
        dataset_len = len(dataset) if dataset is not None else None
    except TypeError:
        dataset_len = None
    if not isinstance(dataset_len, int) or dataset_len < 1:
        return False
    return dataset_len % batch_size_raw == 1


def _has_batchnorm_layers(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


def _should_evaluate(epoch: int, *, total_epochs: int, eval_interval: int) -> bool:
    if eval_interval <= 1:
        return True
    return epoch == 1 or epoch == total_epochs or (epoch % eval_interval == 0)


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
    device: torch.device | None = None,
    resume_state: dict[str, Any] | None = None,
) -> tuple[str, ClassifierEvaluation | None]:
    model = _build_classifier(model_config, num_classes)
    resolved_device = device or resolve_device(training_config)
    model.to(resolved_device)

    if resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    epochs = max(1, _as_int(training_config.get("epochs", 1), 1))
    optimizer = build_optimizer(
        model.parameters(),
        training_config,
        default_type="adam",
        default_lr=0.001,
        default_weight_decay=0.0,
    )
    scheduler = build_scheduler(optimizer, training_config, epochs=epochs, default_type="none")

    criterion = nn.CrossEntropyLoss()
    use_amp = config_amp_enabled(training_config, device=resolved_device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = grad_clip_norm_from_config(training_config)

    if _has_batchnorm_layers(model) and _loader_may_emit_small_batch(train_loader):
        raise ValueError("batchnorm_small_batch_unsupported")

    evaluation_cfg = training_config.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        evaluation_cfg = {}
    eval_interval = max(1, _as_int(evaluation_cfg.get("eval_interval_epochs", 1), 1))

    logging_cfg = training_config.get("logging")
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}
    save_every = max(1, _as_int(logging_cfg.get("save_every_epochs", 1), 1))
    keep_best = _as_bool(logging_cfg.get("keep_best", True), True)

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
        resumed_epoch = _as_int(resume_state.get("epoch", 0), 0)
        if resumed_epoch >= 1:
            start_epoch = resumed_epoch + 1

    best_loss: float | None = None
    best_metric: float | None = None
    final_evaluation: ClassifierEvaluation | None = None
    total_epoch_seconds = 0.0

    val_sample_records = getattr(getattr(val_loader, "dataset", None), "samples", None)
    if not isinstance(val_sample_records, list):
        val_sample_records = None

    for epoch in range(start_epoch, epochs + 1):
        if should_cancel():
            return "canceled", None

        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        non_blocking = resolved_device.type == "cuda"
        for images, labels in train_loader:
            if should_cancel():
                return "canceled", None
            images = images.to(resolved_device, non_blocking=non_blocking)
            labels = labels.to(resolved_device, non_blocking=non_blocking)
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
            preds = logits.argmax(dim=1)
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == labels).sum().item())

        if scheduler is not None:
            scheduler.step()
        train_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        train_accuracy = (total_correct / total_samples) if total_samples > 0 else None

        do_eval = _should_evaluate(epoch, total_epochs=epochs, eval_interval=eval_interval)
        evaluation: ClassifierEvaluation | None = None
        if do_eval:
            evaluation = evaluate_classifier(
                model,
                val_loader,
                criterion,
                resolved_device,
                num_classes=num_classes,
                include_predictions=epoch == epochs,
                sample_records=val_sample_records,
            )
            if epoch == epochs:
                final_evaluation = evaluation

        val_loss = float(evaluation.avg_loss) if evaluation is not None else None
        val_accuracy = float(evaluation.accuracy) if evaluation is not None else None
        val_macro_f1 = float(evaluation.macro_f1) if evaluation is not None else None
        val_macro_precision = float(evaluation.macro_precision) if evaluation is not None else None
        val_macro_recall = float(evaluation.macro_recall) if evaluation is not None else None

        current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
        epoch_seconds = float(time.perf_counter() - epoch_started)
        total_epoch_seconds += epoch_seconds
        average_epoch_seconds = total_epoch_seconds / max(1, (epoch - start_epoch + 1))
        remaining_epochs = max(0, epochs - epoch)
        eta_seconds = float(average_epoch_seconds * remaining_epochs) if remaining_epochs > 0 else 0.0
        on_epoch(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                train_accuracy=float(train_accuracy) if isinstance(train_accuracy, (int, float)) else None,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_macro_f1=val_macro_f1,
                val_macro_precision=val_macro_precision,
                val_macro_recall=val_macro_recall,
                lr=current_lr,
                epoch_seconds=epoch_seconds,
                eta_seconds=eta_seconds,
                evaluated=bool(evaluation is not None),
            )
        )

        metrics_payload = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy) if isinstance(train_accuracy, (int, float)) else None,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_macro_precision": val_macro_precision,
            "val_macro_recall": val_macro_recall,
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
            "eta_seconds": eta_seconds,
        }

        should_save_latest = epoch == epochs or (epoch % save_every == 0)
        if should_save_latest:
            latest_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "metrics": metrics_payload,
            }
            on_checkpoint(
                "latest",
                epoch,
                "val_accuracy" if isinstance(val_accuracy, float) else None,
                float(val_accuracy) if isinstance(val_accuracy, float) else None,
                latest_state,
            )

        if keep_best and evaluation is not None:
            if best_loss is None or float(evaluation.avg_loss) < best_loss:
                best_loss = float(evaluation.avg_loss)
                best_loss_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics_payload,
                }
                on_checkpoint("best_loss", epoch, "val_loss", best_loss, best_loss_state)

            if best_metric is None or float(evaluation.accuracy) > best_metric:
                best_metric = float(evaluation.accuracy)
                best_metric_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics_payload,
                }
                on_checkpoint("best_metric", epoch, "val_accuracy", best_metric, best_metric_state)

    return "completed", final_evaluation
