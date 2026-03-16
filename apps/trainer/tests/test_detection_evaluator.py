from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixel_sheriff_trainer.detection.evaluation import (
    DetectionGroundTruth,
    DetectionPrediction,
    evaluate_detection_set,
)
from pixel_sheriff_trainer.io.detection_evaluation import write_detection_evaluation
from pixel_sheriff_trainer.io.storage import ExperimentStorage


CLASS_ORDER = ["boat", "truck"]
CLASS_NAMES = ["Boat", "Truck"]


def _gt(
    annotation_id: str,
    image_id: str,
    class_index: int,
    bbox: tuple[float, float, float, float],
) -> DetectionGroundTruth:
    return DetectionGroundTruth(
        image_id=image_id,
        class_index=class_index,
        bbox=bbox,
        annotation_id=annotation_id,
        asset_id=image_id,
        relative_path=f"assets/{image_id}.jpg",
    )


def _pred(
    prediction_id: str,
    image_id: str,
    class_index: int,
    bbox: tuple[float, float, float, float],
    score: float,
) -> DetectionPrediction:
    return DetectionPrediction(
        image_id=image_id,
        class_index=class_index,
        bbox=bbox,
        score=score,
        prediction_id=prediction_id,
        asset_id=image_id,
        relative_path=f"assets/{image_id}.jpg",
    )


def _evaluate(
    predictions: list[DetectionPrediction],
    ground_truth: list[DetectionGroundTruth],
):
    return evaluate_detection_set(
        predictions,
        ground_truth,
        class_order=CLASS_ORDER,
        class_names=CLASS_NAMES,
    )


def test_detection_evaluator_perfect_prediction_has_unit_map() -> None:
    evaluation = _evaluate(
        [_pred("pred-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.95)],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.mAP50 == pytest.approx(1.0)
    assert evaluation.mAP50_95 == pytest.approx(1.0)
    assert evaluation.overall is not None
    assert evaluation.overall.precision == pytest.approx(1.0)
    assert evaluation.overall.recall == pytest.approx(1.0)
    assert evaluation.per_class[0].ap50 == pytest.approx(1.0)
    assert evaluation.per_class[0].ap75 == pytest.approx(1.0)
    assert evaluation.diagnostics is not None
    assert evaluation.diagnostics.prediction_rows[0].status == "matched_tp"


def test_detection_evaluator_handles_no_predictions() -> None:
    evaluation = _evaluate(
        [],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.mAP50 == pytest.approx(0.0)
    assert evaluation.mAP50_95 == pytest.approx(0.0)
    assert evaluation.overall is not None
    assert evaluation.overall.tp == 0
    assert evaluation.overall.fn == 1
    assert evaluation.overall.recall == pytest.approx(0.0)
    assert evaluation.per_class[0].ap50 == pytest.approx(0.0)
    assert len(evaluation.diagnostics.unmatched_ground_truths) == 1  # type: ignore[union-attr]


def test_detection_evaluator_handles_no_ground_truth() -> None:
    evaluation = _evaluate(
        [_pred("pred-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.90)],
        [],
    )

    assert evaluation.mAP50 == pytest.approx(0.0)
    assert evaluation.mAP50_95 == pytest.approx(0.0)
    assert evaluation.overall is not None
    assert evaluation.overall.fp == 1
    assert evaluation.overall.recall == pytest.approx(0.0)
    assert evaluation.per_class[0].ap50 is None
    assert evaluation.diagnostics.prediction_rows[0].status == "background_fp"  # type: ignore[union-attr]


def test_detection_evaluator_marks_duplicate_predictions() -> None:
    evaluation = _evaluate(
        [
            _pred("pred-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.95),
            _pred("pred-2", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.85),
        ],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.overall is not None
    assert evaluation.overall.tp == 1
    assert evaluation.overall.fp == 1
    assert evaluation.overall.duplicate_fp == 1
    statuses = {row.prediction_id: row.status for row in evaluation.diagnostics.prediction_rows}  # type: ignore[union-attr]
    assert statuses == {"pred-1": "matched_tp", "pred-2": "duplicate_fp"}


def test_detection_evaluator_wrong_class_high_iou_is_class_fp() -> None:
    evaluation = _evaluate(
        [_pred("pred-1", "img-1", 1, (0.0, 0.0, 10.0, 10.0), 0.95)],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.overall is not None
    assert evaluation.overall.tp == 0
    assert evaluation.overall.fp == 1
    assert evaluation.overall.fn == 1
    assert evaluation.diagnostics.prediction_rows[0].status == "class_fp"  # type: ignore[union-attr]


def test_detection_evaluator_low_iou_prediction_is_localization_fp() -> None:
    evaluation = _evaluate(
        [_pred("pred-1", "img-1", 0, (0.0, 0.0, 7.0, 7.0), 0.95)],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.per_class[0].ap50 == pytest.approx(0.0)
    assert evaluation.per_class[0].ap75 == pytest.approx(0.0)
    assert evaluation.diagnostics.prediction_rows[0].status == "localization_fp"  # type: ignore[union-attr]


def test_detection_evaluator_confidence_order_changes_ap() -> None:
    ground_truth = [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))]
    low_ap = _evaluate(
        [
            _pred("pred-fp", "img-1", 0, (20.0, 20.0, 30.0, 30.0), 0.95),
            _pred("pred-tp", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.80),
        ],
        ground_truth,
    )
    high_ap = _evaluate(
        [
            _pred("pred-tp", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.95),
            _pred("pred-fp", "img-1", 0, (20.0, 20.0, 30.0, 30.0), 0.80),
        ],
        ground_truth,
    )

    assert low_ap.per_class[0].ap50 == pytest.approx(0.5)
    assert high_ap.per_class[0].ap50 == pytest.approx(1.0)


def test_detection_evaluator_aggregates_multiple_images() -> None:
    evaluation = _evaluate(
        [
            _pred("pred-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.95),
            _pred("pred-2", "img-2", 0, (0.0, 0.0, 10.0, 10.0), 0.90),
        ],
        [
            _gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0)),
            _gt("gt-2", "img-2", 0, (0.0, 0.0, 10.0, 10.0)),
        ],
    )

    assert evaluation.overall is not None
    assert evaluation.overall.image_count == 2
    assert evaluation.overall.tp == 2
    assert evaluation.overall.recall == pytest.approx(1.0)


def test_detection_evaluator_aggregates_multiple_classes() -> None:
    evaluation = _evaluate(
        [
            _pred("pred-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.95),
            _pred("pred-2", "img-2", 1, (20.0, 20.0, 40.0, 40.0), 0.90),
        ],
        [
            _gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0)),
            _gt("gt-2", "img-2", 1, (20.0, 20.0, 40.0, 40.0)),
        ],
    )

    assert evaluation.mAP50 == pytest.approx(1.0)
    assert [row.class_id for row in evaluation.per_class] == CLASS_ORDER
    assert evaluation.per_class[0].ap50 == pytest.approx(1.0)
    assert evaluation.per_class[1].ap50 == pytest.approx(1.0)


def test_detection_evaluator_ap50_and_ap75_can_differ() -> None:
    evaluation = _evaluate(
        [_pred("pred-1", "img-1", 0, (0.0, 0.0, 8.0, 8.0), 0.95)],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.per_class[0].ap50 == pytest.approx(1.0)
    assert evaluation.per_class[0].ap75 == pytest.approx(0.0)


def test_detection_evaluator_map_50_95_averages_thresholds() -> None:
    evaluation = _evaluate(
        [_pred("pred-1", "img-1", 0, (0.0, 0.0, 8.0, 8.0), 0.95)],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    assert evaluation.mAP50 == pytest.approx(1.0)
    assert evaluation.mAP50_95 == pytest.approx(0.3)


def test_detection_evaluator_reports_size_buckets() -> None:
    evaluation = _evaluate(
        [
            _pred("pred-small", "img-1", 0, (0.0, 0.0, 20.0, 20.0), 0.95),
            _pred("pred-medium", "img-2", 0, (0.0, 0.0, 60.0, 60.0), 0.95),
            _pred("pred-large", "img-3", 0, (0.0, 0.0, 120.0, 120.0), 0.95),
        ],
        [
            _gt("gt-small", "img-1", 0, (0.0, 0.0, 20.0, 20.0)),
            _gt("gt-medium", "img-2", 0, (0.0, 0.0, 60.0, 60.0)),
            _gt("gt-large", "img-3", 0, (0.0, 0.0, 120.0, 120.0)),
        ],
    )

    assert evaluation.overall is not None
    assert evaluation.overall.ap_small == pytest.approx(1.0)
    assert evaluation.overall.ap_medium == pytest.approx(1.0)
    assert evaluation.overall.ap_large == pytest.approx(1.0)


def test_detection_evaluator_records_filtered_predictions_and_unmatched_ground_truths() -> None:
    evaluation = evaluate_detection_set(
        [
            _pred("pred-low", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.20),
            _pred("pred-high", "img-1", 0, (20.0, 20.0, 30.0, 30.0), 0.95),
        ],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
        class_order=CLASS_ORDER,
        class_names=CLASS_NAMES,
        score_threshold=0.3,
        max_detections_per_image=1,
    )

    statuses = {row.prediction_id: row.status for row in evaluation.diagnostics.prediction_rows}  # type: ignore[union-attr]
    assert statuses["pred-low"] == "low_score_filtered"
    assert statuses["pred-high"] == "background_fp"
    assert len(evaluation.diagnostics.unmatched_ground_truths) == 1  # type: ignore[union-attr]


def test_detection_writer_persists_enriched_artifacts(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    evaluation = _evaluate(
        [
            _pred("pred-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.95),
            _pred("pred-2", "img-1", 0, (0.0, 0.0, 10.0, 10.0), 0.85),
        ],
        [_gt("gt-1", "img-1", 0, (0.0, 0.0, 10.0, 10.0))],
    )

    write_detection_evaluation(
        storage,
        project_id="project-1",
        experiment_id="experiment-1",
        attempt=1,
        model_id="model-1",
        task_id="task-1",
        job_id="job-1",
        dataset_export={
            "dataset_version_id": "dv-1",
            "content_hash": "hash-1",
            "zip_relpath": "exports/project-1/hash-1.zip",
        },
        class_order=CLASS_ORDER,
        class_names=CLASS_NAMES,
        evaluation=evaluation,
    )

    run_dir = tmp_path / "experiments" / "project-1" / "experiment-1" / "runs" / "1"
    evaluation_payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    predictions_lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    predictions_meta = json.loads((run_dir / "predictions.meta.json").read_text(encoding="utf-8"))

    assert evaluation_payload["task"] == "detection"
    assert evaluation_payload["overall"]["mAP50"] == pytest.approx(1.0)
    assert evaluation_payload["overall"]["duplicate_fp"] == 1
    assert evaluation_payload["per_class"][0]["precision"] == pytest.approx(0.5)
    assert evaluation_payload["per_class"][0]["ap50"] == pytest.approx(1.0)
    assert evaluation_payload["pr_curves"]
    assert evaluation_payload["diagnostics"]["per_image"][0]["predictions"][1]["status"] == "duplicate_fp"
    assert evaluation_payload["samples"]["misclassified"] == []
    assert len(predictions_lines) == 2
    assert predictions_meta["task"] == "detection"
    assert predictions_meta["attempt"] == 1
