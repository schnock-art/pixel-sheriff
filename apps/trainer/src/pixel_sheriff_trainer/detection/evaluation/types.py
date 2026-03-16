from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DetectionPredictionStatus = Literal[
    "matched_tp",
    "duplicate_fp",
    "localization_fp",
    "class_fp",
    "background_fp",
    "low_score_filtered",
    "max_detections_filtered",
]


@dataclass(frozen=True)
class DetectionPrediction:
    image_id: str
    class_index: int
    bbox: tuple[float, float, float, float]
    score: float
    prediction_id: str | None = None
    asset_id: str = ""
    relative_path: str = ""


@dataclass(frozen=True)
class DetectionGroundTruth:
    image_id: str
    class_index: int
    bbox: tuple[float, float, float, float]
    annotation_id: str | None = None
    asset_id: str = ""
    relative_path: str = ""
    area: float | None = None


@dataclass(frozen=True)
class DetectionMatchDecision:
    prediction_id: str
    ground_truth_id: str | None
    iou: float
    is_true_positive: bool
    is_duplicate: bool
    best_same_class_iou: float
    best_same_class_ground_truth_id: str | None


@dataclass(frozen=True)
class DetectionTraceRow:
    prediction_id: str
    image_id: str
    score: float
    status: str
    reason: str
    cumulative_tp: int
    cumulative_fp: int
    precision: float
    recall: float
    matched_ground_truth_id: str | None = None
    iou: float | None = None


@dataclass(frozen=True)
class DetectionTrace:
    class_index: int
    class_id: str
    name: str
    iou_threshold: float
    rows: list[DetectionTraceRow] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionPRCurve:
    class_index: int
    class_id: str
    name: str
    iou_threshold: float
    scores: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    precision_envelope: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionPredictionResult:
    prediction_id: str
    image_id: str
    asset_id: str
    relative_path: str
    class_index: int
    class_id: str
    name: str
    bbox: tuple[float, float, float, float]
    score: float
    status: DetectionPredictionStatus
    reason: str
    rank: int
    matched_ground_truth_id: str | None = None
    matched_iou: float | None = None
    best_same_class_iou: float | None = None
    best_same_class_ground_truth_id: str | None = None
    best_any_iou: float | None = None
    best_any_ground_truth_id: str | None = None
    best_any_ground_truth_class_index: int | None = None


@dataclass(frozen=True)
class DetectionGroundTruthResult:
    annotation_id: str
    image_id: str
    asset_id: str
    relative_path: str
    class_index: int
    class_id: str
    name: str
    bbox: tuple[float, float, float, float]
    area: float
    matched_prediction_id: str | None = None
    matched_iou: float | None = None


@dataclass(frozen=True)
class DetectionMatchedPair:
    image_id: str
    prediction_id: str
    ground_truth_id: str
    class_index: int
    class_id: str
    name: str
    score: float
    iou: float


@dataclass(frozen=True)
class DetectionImageResult:
    image_id: str
    asset_id: str
    relative_path: str
    prediction_count: int
    ground_truth_count: int
    predictions: list[DetectionPredictionResult] = field(default_factory=list)
    matched_pairs: list[DetectionMatchedPair] = field(default_factory=list)
    unmatched_ground_truths: list[DetectionGroundTruthResult] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionSizeBucketSummary:
    name: str
    ground_truth_count: int
    prediction_count: int
    ap50: float | None = None
    map_50_95: float | None = None
    precision: float = 0.0
    recall: float = 0.0


@dataclass(frozen=True)
class DetectionPerClassMetrics:
    class_index: int
    class_id: str
    name: str
    precision: float
    recall: float
    f1: float
    support: int
    ap50: float | None = None
    ap75: float | None = None
    map_50_95: float | None = None
    ap_by_iou: dict[str, float | None] = field(default_factory=dict)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicate_fp: int = 0
    matched_mean_iou: float | None = None


@dataclass(frozen=True)
class DetectionOverallMetrics:
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    duplicate_fp: int
    matched_mean_iou: float | None = None
    image_count: int = 0
    prediction_count: int = 0
    ground_truth_count: int = 0
    ap_small: float | None = None
    ap_medium: float | None = None
    ap_large: float | None = None
    size_buckets: list[DetectionSizeBucketSummary] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionDiagnostics:
    per_image: list[DetectionImageResult] = field(default_factory=list)
    prediction_rows: list[DetectionPredictionResult] = field(default_factory=list)
    unmatched_ground_truths: list[DetectionGroundTruthResult] = field(default_factory=list)
    confidence_traces: list[DetectionTrace] = field(default_factory=list)


@dataclass
class DetectionEvaluation:
    mAP50: float
    mAP50_95: float
    overall: DetectionOverallMetrics | None = None
    per_class: list[DetectionPerClassMetrics] = field(default_factory=list)
    pr_curves: list[DetectionPRCurve] = field(default_factory=list)
    diagnostics: DetectionDiagnostics | None = None
    iou_thresholds: list[float] = field(default_factory=list)
    diagnostics_iou_threshold: float = 0.5
    score_threshold: float | None = None
    max_detections_per_image: int | None = None
