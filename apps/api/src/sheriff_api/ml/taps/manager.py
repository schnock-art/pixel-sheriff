from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class TapSpec:
    name: str
    module_path: str | None = None


class TapManager:
    def __init__(self, model: "nn.Module") -> None:
        self._model = model
        self._captured: dict[str, "torch.Tensor"] = {}
        self._extractors: dict[str, Callable[["torch.Tensor | None"], "torch.Tensor"]] = {}
        self._handles: list["RemovableHandle"] = []

    def register_hook(self, tap_name: str, module_path: str) -> None:
        modules = dict(self._model.named_modules())
        module = modules.get(module_path)
        if module is None:
            raise ValueError(f"Cannot register tap hook. Module path '{module_path}' not found")

        def _capture(_module, _inputs, output) -> None:  # type: ignore[no-untyped-def]
            self._captured[tap_name] = output

        self._handles.append(module.register_forward_hook(_capture))

    def register_extractor(self, tap_name: str, extractor: Callable[["torch.Tensor | None"], "torch.Tensor"]) -> None:
        self._extractors[tap_name] = extractor

    def clear(self) -> None:
        self._captured.clear()

    def get(self, tap_name: str, x: "torch.Tensor | None" = None) -> "torch.Tensor | None":
        cached = self._captured.get(tap_name)
        if cached is not None:
            return cached
        extractor = self._extractors.get(tap_name)
        if extractor is None:
            return None
        return extractor(x)

    def require(self, tap_name: str, x: "torch.Tensor | None" = None) -> "torch.Tensor":
        tensor = self.get(tap_name, x)
        if tensor is None:
            raise ValueError(f"Tap '{tap_name}' is unavailable")
        return tensor

    def close(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            handle.remove()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
