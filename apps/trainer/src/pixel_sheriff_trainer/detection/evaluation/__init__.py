from __future__ import annotations

"""Deterministic bounding-box detection evaluation helpers.

This package implements a small, pure evaluator for axis-aligned detection
tasks. It favors readable matching and diagnostics over pycocotools parity.

Current intentional simplifications:
- axis-aligned boxes only
- no crowd / ignore support
- no segmentation or keypoints
- size buckets use a COCO-like area approximation without ignore semantics
"""

from .metrics import evaluate_detection_set
from .types import (
    DetectionDiagnostics,
    DetectionEvaluation,
    DetectionGroundTruth,
    DetectionGroundTruthResult,
    DetectionImageResult,
    DetectionMatchDecision,
    DetectionMatchedPair,
    DetectionOverallMetrics,
    DetectionPerClassMetrics,
    DetectionPRCurve,
    DetectionPrediction,
    DetectionPredictionResult,
    DetectionSizeBucketSummary,
    DetectionTrace,
    DetectionTraceRow,
)

__all__ = [
    "DetectionDiagnostics",
    "DetectionEvaluation",
    "DetectionGroundTruth",
    "DetectionGroundTruthResult",
    "DetectionImageResult",
    "DetectionMatchDecision",
    "DetectionMatchedPair",
    "DetectionOverallMetrics",
    "DetectionPerClassMetrics",
    "DetectionPRCurve",
    "DetectionPrediction",
    "DetectionPredictionResult",
    "DetectionSizeBucketSummary",
    "DetectionTrace",
    "DetectionTraceRow",
    "evaluate_detection_set",
]
