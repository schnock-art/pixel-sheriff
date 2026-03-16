from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
import zipfile

import pytest

try:
    import torch  # noqa: F401
    from pixel_sheriff_ml.model_factory import build_classifier_model, build_resnet_classifier
    from pixel_sheriff_trainer.augmentation import (
        apply_detection_augmentation,
        apply_segmentation_augmentation,
        resolve_training_augmentation,
    )
    from pixel_sheriff_trainer.classification.dataset import build_classification_loaders
    from pixel_sheriff_trainer.detection.dataset import build_detection_loaders
    from pixel_sheriff_trainer.detection.eval import DetectionEvaluation
    from pixel_sheriff_trainer.detection.train import DetectionEpochMetrics, run_detection_training
    from pixel_sheriff_trainer.export_onnx import (
        _validate_onnxruntime_batch_outputs,
        export_best_classification_onnx,
        export_model_to_onnx,
    )
    from pixel_sheriff_trainer.io.checkpoints import compact_completed_checkpoints, read_checkpoints, save_checkpoint
    from pixel_sheriff_trainer.io.storage import ExperimentStorage
    from pixel_sheriff_trainer.jobs import TrainJob, parse_train_job
    from pixel_sheriff_trainer.runner import TrainRunner
    from pixel_sheriff_trainer.segmentation.dataset import build_segmentation_loaders
    from pixel_sheriff_trainer.utils.torchvision_cache import configure_torchvision_cache, resolve_torchvision_cache_root

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    build_classifier_model = None  # type: ignore[assignment]
    build_resnet_classifier = None  # type: ignore[assignment]
    apply_detection_augmentation = None  # type: ignore[assignment]
    apply_segmentation_augmentation = None  # type: ignore[assignment]
    resolve_training_augmentation = None  # type: ignore[assignment]
    build_classification_loaders = None  # type: ignore[assignment]
    build_detection_loaders = None  # type: ignore[assignment]
    DetectionEvaluation = None  # type: ignore[assignment]
    DetectionEpochMetrics = None  # type: ignore[assignment]
    run_detection_training = None  # type: ignore[assignment]
    export_best_classification_onnx = None  # type: ignore[assignment]
    export_model_to_onnx = None  # type: ignore[assignment]
    _validate_onnxruntime_batch_outputs = None  # type: ignore[assignment]
    compact_completed_checkpoints = None  # type: ignore[assignment]
    read_checkpoints = None  # type: ignore[assignment]
    save_checkpoint = None  # type: ignore[assignment]
    ExperimentStorage = None  # type: ignore[assignment]
    TrainJob = None  # type: ignore[assignment]
    parse_train_job = None  # type: ignore[assignment]
    TrainRunner = None  # type: ignore[assignment]
    build_segmentation_loaders = None  # type: ignore[assignment]
    configure_torchvision_cache = None  # type: ignore[assignment]
    resolve_torchvision_cache_root = None  # type: ignore[assignment]

try:
    import numpy as np  # noqa: F401
    import onnxruntime as ort  # noqa: F401

    HAS_ONNX_RUNTIME = True
except Exception:
    HAS_ONNX_RUNTIME = False

def _write_tiny_export_zip(root: Path, project_id: str) -> tuple[str, Path]:
    from PIL import Image

    assets_dir = root / "tmp_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        image_path = assets_dir / f"img{index}.png"
        Image.new("RGB", (8, 8), color=color).save(image_path)
        image_paths.append(image_path)

    manifest = {
        "schema_version": "1.2",
        "label_schema": {
            "classes": [{"id": 1, "name": "cat"}],
            "class_order": [1],
        },
        "splits": {
            "train": {"asset_ids": ["asset-0", "asset-1"]},
            "val": {"asset_ids": ["asset-2"]},
            "test": {"asset_ids": []},
        },
        "assets": [
            {
                "asset_id": "asset-0",
                "path": "assets/img0.png",
                "media_type": "image",
                "width": 8,
                "height": 8,
                "coco": {"image_id": "asset-0"},
            },
            {
                "asset_id": "asset-1",
                "path": "assets/img1.png",
                "media_type": "image",
                "width": 8,
                "height": 8,
                "coco": {"image_id": "asset-1"},
            },
            {
                "asset_id": "asset-2",
                "path": "assets/img2.png",
                "media_type": "image",
                "width": 8,
                "height": 8,
                "coco": {"image_id": "asset-2"},
            },
        ],
        "annotations": [
            {
                "annotation_id": "ann-0",
                "asset_id": "asset-0",
                "status": "approved",
                "labels": {
                    "image": {
                        "mode": "single",
                        "primary_class_id": 1,
                        "class_ids": [1],
                        "confidence": None,
                    },
                    "objects": [],
                },
                "exports": {"coco": {"image_id": "asset-0", "annotation_ids": []}},
            },
            {
                "annotation_id": "ann-1",
                "asset_id": "asset-1",
                "status": "approved",
                "labels": {
                    "image": {
                        "mode": "single",
                        "primary_class_id": 1,
                        "class_ids": [1],
                        "confidence": None,
                    },
                    "objects": [],
                },
                "exports": {"coco": {"image_id": "asset-1", "annotation_ids": []}},
            },
            {
                "annotation_id": "ann-2",
                "asset_id": "asset-2",
                "status": "approved",
                "labels": {
                    "image": {
                        "mode": "single",
                        "primary_class_id": 1,
                        "class_ids": [1],
                        "confidence": None,
                    },
                    "objects": [],
                },
                "exports": {"coco": {"image_id": "asset-2", "annotation_ids": []}},
            },
        ],
    }
    content_hash = "tinyhash123"
    zip_relpath = Path("exports") / project_id / f"{content_hash}.zip"
    zip_path = root / zip_relpath
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for index, image_path in enumerate(image_paths):
            bundle.write(image_path, arcname=f"assets/img{index}.png")
    return content_hash, zip_path


def _seed_experiment_layout(root: Path, project_id: str, experiment_id: str, job_id: str) -> None:
    exp_dir = root / "experiments" / project_id / experiment_id
    run_dir = exp_dir / "runs" / "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    records = [
        {
            "id": experiment_id,
            "project_id": project_id,
            "model_id": "model-1",
            "name": "run-1",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "status": "queued",
            "summary_json": {
                "best_metric_name": None,
                "best_metric_value": None,
                "best_epoch": None,
                "last_epoch": None,
            },
            "artifacts_json": {},
            "config_json": {},
        }
    ]
    records_path = root / "experiments" / project_id / "records.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    status = {
        "status": "queued",
        "cancel_requested": False,
        "current_run_attempt": 1,
        "last_completed_attempt": None,
        "active_job_id": job_id,
        "error": None,
        "updated_at": "2025-01-01T00:00:00Z",
    }
    (exp_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({"attempt": 1, "job_id": job_id, "dataset_export": {}, "started_at": None, "ended_at": None}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "events.meta.json").write_text(json.dumps({"line_count": 0, "updated_at": None}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    (run_dir / "checkpoints.json").write_text(
        json.dumps(
            [
                {"kind": "best_metric", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None},
                {"kind": "best_loss", "epoch": None, "metric_name": "val_loss", "value": None, "updated_at": None, "uri": None},
                {"kind": "latest", "epoch": None, "metric_name": None, "value": None, "updated_at": None, "uri": None},
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_tiny_coco_export_zip(root: Path, project_id: str, *, include_segmentation: bool) -> Path:
    from PIL import Image

    asset_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    assets_dir = root / "tmp_coco_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for index, color in enumerate([(255, 255, 255), (220, 220, 220)]):
        image_path = assets_dir / f"coco_{index}.png"
        Image.new("RGB", (16, 16), color=color).save(image_path)
        image_paths.append(image_path)

    manifest = {
        "schema_version": "1.2",
        "label_schema": {
            "classes": [{"id": 1, "name": "flower"}],
            "class_order": [1],
        },
        "splits": {
            "train": {"asset_ids": [asset_ids[0]]},
            "val": {"asset_ids": [asset_ids[1]]},
            "test": {"asset_ids": []},
        },
        "assets": [
            {
                "asset_id": asset_ids[0],
                "path": "assets/coco_0.png",
                "media_type": "image",
                "width": 16,
                "height": 16,
                "coco": {"image_id": asset_ids[0]},
            },
            {
                "asset_id": asset_ids[1],
                "path": "assets/coco_1.png",
                "media_type": "image",
                "width": 16,
                "height": 16,
                "coco": {"image_id": asset_ids[1]},
            },
        ],
    }

    coco = {
        "images": [
            {
                "id": asset_ids[0],
                "asset_id": asset_ids[0],
                "file_name": "assets/coco_0.png",
                "width": 16,
                "height": 16,
            },
            {
                "id": asset_ids[1],
                "asset_id": asset_ids[1],
                "file_name": "assets/coco_1.png",
                "width": 16,
                "height": 16,
            },
        ],
        "categories": [
            {
                "id": 1,
                "name": "flower",
                "stable_id": "flower",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": asset_ids[0],
                "category_id": 1,
                "bbox": [1, 1, 8, 8],
                "area": 64,
                "iscrowd": 0,
                **(
                    {"segmentation": [[1, 1, 9, 1, 9, 9, 1, 9]]}
                    if include_segmentation else
                    {}
                ),
            },
            {
                "id": 2,
                "image_id": asset_ids[1],
                "category_id": 1,
                "bbox": [2, 2, 6, 6],
                "area": 36,
                "iscrowd": 0,
                **(
                    {"segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]]}
                    if include_segmentation else
                    {}
                ),
            },
        ],
    }

    zip_path = root / "exports" / project_id / "tiny_coco.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        bundle.writestr("coco_instances.json", json.dumps(coco, indent=2, sort_keys=True))
        for index, image_path in enumerate(image_paths):
            bundle.write(image_path, arcname=f"assets/coco_{index}.png")
    return zip_path
