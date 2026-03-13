from __future__ import annotations

import os
from pathlib import Path


def resolve_torchvision_cache_root(storage_root: str | None = None) -> Path:
    override = os.getenv("TORCHVISION_CACHE_ROOT")
    if isinstance(override, str) and override.strip():
        root = Path(override.strip()).expanduser()
    else:
        base = storage_root or os.getenv("STORAGE_ROOT") or "./data"
        root = Path(base).expanduser() / "model_weights" / "torchvision"
    return root.resolve()


def configure_torchvision_cache(storage_root: str | None = None) -> Path:
    import torch

    root = resolve_torchvision_cache_root(storage_root)
    hub_dir = root / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(root)
    torch.hub.set_dir(str(hub_dir))
    return root
