from __future__ import annotations

from fastapi import APIRouter

from . import analytics, crud, evaluation, onnx, runs
from .shared import (
    experiment_store,
    model_store,
    settings,
    shared_architecture_family,
    storage,
    train_queue,
)

router = APIRouter(tags=["experiments"])
router.include_router(analytics.router, tags=["experiments"])
router.include_router(evaluation.router, tags=["experiments"])
router.include_router(onnx.router, tags=["experiments"])
router.include_router(runs.router, tags=["experiments"])
router.include_router(crud.router, tags=["experiments"])

__all__ = [
    "router",
    "settings",
    "model_store",
    "experiment_store",
    "storage",
    "train_queue",
    "shared_architecture_family",
]
