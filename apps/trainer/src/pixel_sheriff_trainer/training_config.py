from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.optim as optim


def as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(parsed)


def as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed)


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def resolve_device(training_config: dict[str, Any]) -> torch.device:
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


def amp_enabled(training_config: dict[str, Any], *, device: torch.device) -> bool:
    precision = str(training_config.get("precision", "fp32")).strip().lower()
    return precision == "amp" and device.type == "cuda"


@dataclass(frozen=True)
class LoaderRuntimeSettings:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    cache_resized_images: bool
    max_cached_images: int
    drop_last: bool


def resolve_runtime_loader_settings(
    training_config: dict[str, Any],
    *,
    device: torch.device,
) -> LoaderRuntimeSettings:
    runtime = training_config.get("runtime")
    advanced = training_config.get("advanced")

    num_workers = 0
    if isinstance(runtime, dict) and runtime.get("num_workers") is not None:
        num_workers = max(0, as_int(runtime.get("num_workers"), 0))
    elif isinstance(advanced, dict) and advanced.get("num_workers") is not None:
        num_workers = max(0, as_int(advanced.get("num_workers"), 0))

    pin_memory = device.type == "cuda"
    if isinstance(runtime, dict) and runtime.get("pin_memory") is not None:
        pin_memory = as_bool(runtime.get("pin_memory"), pin_memory)

    persistent_workers = num_workers > 0
    if isinstance(runtime, dict) and runtime.get("persistent_workers") is not None:
        persistent_workers = as_bool(runtime.get("persistent_workers"), persistent_workers)
    if num_workers < 1:
        persistent_workers = False

    prefetch_factor = 2
    if isinstance(runtime, dict) and runtime.get("prefetch_factor") is not None:
        prefetch_factor = max(1, as_int(runtime.get("prefetch_factor"), prefetch_factor))

    cache_resized_images = num_workers == 0
    if isinstance(runtime, dict) and runtime.get("cache_resized_images") is not None:
        cache_resized_images = as_bool(runtime.get("cache_resized_images"), cache_resized_images)

    max_cached_images = 1024
    if isinstance(runtime, dict) and runtime.get("max_cached_images") is not None:
        max_cached_images = max(0, as_int(runtime.get("max_cached_images"), max_cached_images))

    training_block = training_config.get("training")
    drop_last = True
    if isinstance(training_block, dict) and training_block.get("drop_last") is not None:
        drop_last = as_bool(training_block.get("drop_last"), drop_last)

    return LoaderRuntimeSettings(
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        prefetch_factor=int(prefetch_factor),
        cache_resized_images=bool(cache_resized_images),
        max_cached_images=int(max_cached_images),
        drop_last=bool(drop_last),
    )


@dataclass(frozen=True)
class RuntimeInfo:
    device_selected: str
    cuda_available: bool
    mps_available: bool
    amp_enabled: bool
    torch_version: str
    torchvision_version: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    cache_resized_images: bool
    max_cached_images: int


def resolve_runtime_info(training_config: dict[str, Any], *, device: torch.device) -> RuntimeInfo:
    loader_settings = resolve_runtime_loader_settings(training_config, device=device)
    torchvision_version = "unknown"
    try:
        import torchvision

        torchvision_version = str(getattr(torchvision, "__version__", "unknown"))
    except Exception:
        torchvision_version = "unknown"
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    return RuntimeInfo(
        device_selected=device.type,
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=mps_available,
        amp_enabled=amp_enabled(training_config, device=device),
        torch_version=str(getattr(torch, "__version__", "unknown")),
        torchvision_version=torchvision_version,
        num_workers=loader_settings.num_workers,
        pin_memory=loader_settings.pin_memory,
        persistent_workers=loader_settings.persistent_workers,
        prefetch_factor=loader_settings.prefetch_factor,
        cache_resized_images=loader_settings.cache_resized_images,
        max_cached_images=loader_settings.max_cached_images,
    )


def grad_clip_norm_from_config(training_config: dict[str, Any]) -> float | None:
    advanced = training_config.get("advanced")
    if isinstance(advanced, dict) and advanced.get("grad_clip_norm") is not None:
        try:
            return float(advanced.get("grad_clip_norm"))
        except (TypeError, ValueError):
            return None
    return None


def build_optimizer(
    parameters: Any,
    training_config: dict[str, Any],
    *,
    default_type: str,
    default_lr: float,
    default_weight_decay: float,
    default_momentum: float = 0.9,
) -> optim.Optimizer:
    optimizer_cfg = training_config.get("optimizer")
    if not isinstance(optimizer_cfg, dict):
        optimizer_cfg = {}
    lr = as_float(optimizer_cfg.get("lr", default_lr), default_lr)
    weight_decay = as_float(optimizer_cfg.get("weight_decay", default_weight_decay), default_weight_decay)
    optimizer_type = str(optimizer_cfg.get("type", default_type)).strip().lower()
    if optimizer_type == "sgd":
        momentum = as_float(optimizer_cfg.get("momentum", default_momentum), default_momentum)
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: optim.Optimizer,
    training_config: dict[str, Any],
    *,
    epochs: int,
    default_type: str,
) -> Any:
    scheduler_cfg = training_config.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        scheduler_cfg = {}
    scheduler_type = str(scheduler_cfg.get("type", default_type)).strip().lower()
    if scheduler_type == "step":
        params = scheduler_cfg.get("params")
        if not isinstance(params, dict):
            params = {}
        default_step = max(1, epochs // 3)
        step_size = max(1, as_int(params.get("step_size", default_step), default_step))
        gamma = as_float(params.get("gamma", 0.1), 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return None
