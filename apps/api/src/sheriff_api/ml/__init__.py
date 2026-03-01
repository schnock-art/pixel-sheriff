from __future__ import annotations

from typing import Any

__all__ = ["BuiltModel", "build_model"]


def __getattr__(name: str) -> Any:
    if name in {"BuiltModel", "build_model"}:
        from sheriff_api.ml.model_factory import BuiltModel, build_model

        return {"BuiltModel": BuiltModel, "build_model": build_model}[name]
    raise AttributeError(name)
