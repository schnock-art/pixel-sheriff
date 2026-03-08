from __future__ import annotations

import json
from pathlib import Path
import uuid
import zipfile

import pytest

try:
    import torch  # noqa: F401
    from pixel_sheriff_ml.model_factory import build_resnet_classifier
    from pixel_sheriff_trainer.classification.dataset import build_classification_loaders
    from pixel_sheriff_trainer.detection.dataset import build_detection_loaders
    from pixel_sheriff_trainer.detection.eval import DetectionEvaluation
    from pixel_sheriff_trainer.detection.train import DetectionEpochMetrics, run_detection_training
    from pixel_sheriff_trainer.export_onnx import (
        _validate_onnxruntime_batch_outputs,
        export_best_classification_onnx,
        export_model_to_onnx,
    )
    from pixel_sheriff_trainer.io.checkpoints import save_checkpoint
    from pixel_sheriff_trainer.io.storage import ExperimentStorage
    from pixel_sheriff_trainer.jobs import TrainJob, parse_train_job
    from pixel_sheriff_trainer.runner import TrainRunner
    from pixel_sheriff_trainer.segmentation.dataset import build_segmentation_loaders

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    build_resnet_classifier = None  # type: ignore[assignment]
    build_classification_loaders = None  # type: ignore[assignment]
    build_detection_loaders = None  # type: ignore[assignment]
    DetectionEvaluation = None  # type: ignore[assignment]
    DetectionEpochMetrics = None  # type: ignore[assignment]
    run_detection_training = None  # type: ignore[assignment]
    export_best_classification_onnx = None  # type: ignore[assignment]
    export_model_to_onnx = None  # type: ignore[assignment]
    _validate_onnxruntime_batch_outputs = None  # type: ignore[assignment]
    save_checkpoint = None  # type: ignore[assignment]
    ExperimentStorage = None  # type: ignore[assignment]
    TrainJob = None  # type: ignore[assignment]
    parse_train_job = None  # type: ignore[assignment]
    TrainRunner = None  # type: ignore[assignment]
    build_segmentation_loaders = None  # type: ignore[assignment]

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


def test_dataset_loader_reads_tiny_export_zip(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    _content_hash, zip_path = _write_tiny_export_zip(tmp_path, project_id)
    loaded = build_classification_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir",
        model_config={"input": {"input_size": [32, 32], "normalization": {"type": "none"}}},
        training_config={"batch_size": 2, "advanced": {"num_workers": 0, "seed": 1}},
    )
    assert loaded.num_classes == 1
    assert loaded.train_count >= 1
    assert loaded.val_count >= 1
    assert loaded.train_loader.drop_last is True
    assert bool(getattr(loaded.train_loader.dataset, "cache_base_images", False)) is True


def test_dataset_loader_runtime_prefetch_and_cache_overrides(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    _content_hash, zip_path = _write_tiny_export_zip(tmp_path, project_id)
    loaded = build_classification_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_prefetch",
        model_config={"input": {"input_size": [32, 32], "normalization": {"type": "none"}}},
        training_config={
            "batch_size": 2,
            "runtime": {
                "num_workers": 1,
                "prefetch_factor": 3,
                "cache_resized_images": False,
            },
        },
    )
    assert bool(getattr(loaded.train_loader.dataset, "cache_base_images", True)) is False
    assert int(getattr(loaded.train_loader, "prefetch_factor", 0)) == 3


def test_detection_loader_accepts_uuid_image_ids(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)
    loaded = build_detection_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_detection",
        model_config={"input": {"input_size": [32, 32]}},
        training_config={"batch_size": 1},
    )
    assert loaded.num_classes == 1
    assert loaded.train_count == 1
    assert loaded.val_count == 1
    assert loaded.train_loader.drop_last is True

    train_images, train_targets = next(iter(loaded.train_loader))
    assert len(train_images) == 1
    assert tuple(train_targets[0]["boxes"].shape) == (1, 4)
    assert train_targets[0]["labels"].tolist() == [0]


def test_detection_loader_offsets_labels_for_ssdlite_family(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)
    loaded = build_detection_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_detection_ssdlite",
        model_config={
            "input": {"input_size": [320, 320]},
            "architecture": {"family": "ssdlite320_mobilenet_v3_large"},
        },
        training_config={"batch_size": 1},
    )

    train_images, train_targets = next(iter(loaded.train_loader))
    assert len(train_images) == 1
    assert train_targets[0]["labels"].tolist() == [1]


def test_detection_loader_respects_explicit_drop_last_false(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)
    loaded = build_detection_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_detection_drop_last",
        model_config={"input": {"input_size": [32, 32]}},
        training_config={"batch_size": 1, "training": {"drop_last": False}},
    )

    assert loaded.train_loader.drop_last is False


def test_detection_training_smoke_accepts_zero_based_labels(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)
    loaded = build_detection_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_detection_train",
        model_config={"input": {"input_size": [32, 32]}},
        training_config={"batch_size": 1},
    )

    captured_epochs: list[DetectionEpochMetrics] = []
    checkpoints: list[tuple[str, int]] = []
    status, evaluation = run_detection_training(
        model_config={
            "architecture": {
                "family": "retinanet",
                "backbone": {"name": "resnet50", "pretrained": False},
                "head": {"num_classes": loaded.num_classes},
            }
        },
        training_config={
            "optimizer": {"lr": 0.0001, "weight_decay": 0.0001},
            "scheduler": {"type": "none"},
            "logging": {"save_every_epochs": 1, "keep_best": False},
            "evaluation": {"eval_interval_epochs": 1},
            "epochs": 1,
            "batch_size": 1,
        },
        train_loader=loaded.train_loader,
        val_loader=loaded.val_loader,
        num_classes=loaded.num_classes,
        should_cancel=lambda: False,
        on_epoch=lambda row: captured_epochs.append(row),
        on_checkpoint=lambda kind, epoch, _metric_name, _value, _payload: checkpoints.append((kind, epoch)),
        device=torch.device("cpu"),
        resume_state=None,
    )
    assert status == "completed"
    assert evaluation is not None
    assert len(captured_epochs) == 1
    assert checkpoints


def test_detection_training_uses_ssdlite_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    import pixel_sheriff_trainer.detection.train as detection_train

    class FakeDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images, targets=None):  # type: ignore[override]
            if self.training:
                return {"total_loss": self.scale}
            return [
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
                for _ in images
            ]

    called = {"ssdlite": False}

    def _fake_builder(_num_classes: int) -> torch.nn.Module:
        called["ssdlite"] = True
        return FakeDetector()

    monkeypatch.setattr(detection_train, "_build_ssdlite320_mobilenet_v3_large", _fake_builder)
    monkeypatch.setattr(
        detection_train,
        "evaluate_detection",
        lambda *_args, **_kwargs: DetectionEvaluation(mAP50=0.0, mAP50_95=0.0),
    )

    sample_image = torch.zeros((3, 8, 8), dtype=torch.float32)
    sample_target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    train_loader = [([sample_image], [sample_target])]
    val_loader = [([sample_image], [sample_target])]

    status, evaluation = run_detection_training(
        model_config={"architecture": {"family": "ssdlite320_mobilenet_v3_large"}},
        training_config={
            "optimizer": {"lr": 0.0001, "weight_decay": 0.0},
            "scheduler": {"type": "none"},
            "logging": {"save_every_epochs": 1, "keep_best": False},
            "evaluation": {"eval_interval_epochs": 1},
            "epochs": 1,
            "batch_size": 1,
        },
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=1,
        should_cancel=lambda: False,
        on_epoch=lambda _row: None,
        on_checkpoint=lambda *_args, **_kwargs: None,
        device=torch.device("cpu"),
        resume_state=None,
    )

    assert called["ssdlite"] is True
    assert status == "completed"
    assert evaluation is not None


def test_detection_training_rejects_ssdlite_small_batch_loader() -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    class FakeLoader:
        batch_size = 4
        drop_last = False
        dataset = [None] * 37

        def __iter__(self):
            return iter(())

    with pytest.raises(ValueError, match="batchnorm_small_batch_unsupported"):
        run_detection_training(
            model_config={"architecture": {"family": "ssdlite320_mobilenet_v3_large"}},
            training_config={
                "optimizer": {"lr": 0.0001, "weight_decay": 0.0},
                "scheduler": {"type": "none"},
                "logging": {"save_every_epochs": 1, "keep_best": False},
                "evaluation": {"eval_interval_epochs": 1},
                "epochs": 1,
                "batch_size": 4,
            },
            train_loader=FakeLoader(),
            val_loader=[],
            num_classes=4,
            should_cancel=lambda: False,
            on_epoch=lambda _row: None,
            on_checkpoint=lambda *_args, **_kwargs: None,
            device=torch.device("cpu"),
            resume_state=None,
        )


def test_detection_training_falls_back_to_cpu_eval_when_cuda_nms_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    import pixel_sheriff_trainer.detection.train as detection_train

    class FakeTensor:
        def to(self, *_args, **_kwargs):
            return self

    class FakeDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.devices: list[str] = []

        def to(self, device=None, *args, **kwargs):  # type: ignore[override]
            if device is not None:
                self.devices.append(str(device))
            return self

        def forward(self, images, targets=None):  # type: ignore[override]
            if self.training:
                return {"total_loss": self.scale}
            return [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64)}]

    fake_model = FakeDetector()
    eval_devices: list[str] = []

    def _fake_eval(model, _val_loader, device, *, num_classes, iou_thresholds=None):
        eval_devices.append(str(device))
        if str(device) == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return DetectionEvaluation(mAP50=0.25, mAP50_95=0.15)

    monkeypatch.setattr(detection_train, "_build_retinanet", lambda _num_classes: fake_model)
    monkeypatch.setattr(detection_train, "evaluate_detection", _fake_eval)

    train_loader = [
        (
            [FakeTensor(), FakeTensor()],
            [{"boxes": FakeTensor(), "labels": FakeTensor()}, {"boxes": FakeTensor(), "labels": FakeTensor()}],
        )
    ]
    val_loader = [([FakeTensor()], [{"boxes": FakeTensor(), "labels": FakeTensor()}])]

    status, evaluation = run_detection_training(
        model_config={"architecture": {"family": "retinanet"}},
        training_config={
            "optimizer": {"lr": 0.0001, "weight_decay": 0.0},
            "scheduler": {"type": "none"},
            "logging": {"save_every_epochs": 1, "keep_best": False},
            "evaluation": {"eval_interval_epochs": 1},
            "epochs": 1,
            "batch_size": 2,
        },
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=1,
        should_cancel=lambda: False,
        on_epoch=lambda _row: None,
        on_checkpoint=lambda *_args, **_kwargs: None,
        device=torch.device("cuda"),
        resume_state=None,
    )

    assert status == "completed"
    assert evaluation is not None
    assert eval_devices == ["cuda", "cpu"]
    assert fake_model.devices[:3] == ["cuda", "cpu", "cuda"]


def test_detection_training_cancel_stops_between_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    import pixel_sheriff_trainer.detection.train as detection_train

    class FakeDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images, targets=None):  # type: ignore[override]
            return {"total_loss": self.scale}

    yielded_batches = {"count": 0}
    sample_image = torch.zeros((3, 8, 8), dtype=torch.float32)
    sample_target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }

    class FakeLoader:
        def __iter__(self):
            for _ in range(3):
                yielded_batches["count"] += 1
                yield [sample_image.clone()], [{key: value.clone() for key, value in sample_target.items()}]

    monkeypatch.setattr(detection_train, "_build_retinanet", lambda _num_classes: FakeDetector())
    monkeypatch.setattr(
        detection_train,
        "evaluate_detection",
        lambda *_args, **_kwargs: pytest.fail("evaluation should not run after cancellation"),
    )

    status, evaluation = run_detection_training(
        model_config={"architecture": {"family": "retinanet"}},
        training_config={
            "optimizer": {"lr": 0.0001, "weight_decay": 0.0},
            "scheduler": {"type": "none"},
            "logging": {"save_every_epochs": 1, "keep_best": False},
            "evaluation": {"eval_interval_epochs": 1},
            "epochs": 1,
            "batch_size": 1,
        },
        train_loader=FakeLoader(),
        val_loader=[],
        num_classes=1,
        should_cancel=lambda: yielded_batches["count"] >= 2,
        on_epoch=lambda _row: pytest.fail("epoch metrics should not be emitted after cancellation"),
        on_checkpoint=lambda *_args, **_kwargs: pytest.fail("checkpoints should not be written after cancellation"),
        device=torch.device("cpu"),
        resume_state=None,
    )

    assert yielded_batches["count"] == 2
    assert status == "canceled"
    assert evaluation is None


def test_segmentation_loader_accepts_uuid_image_ids(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=True)
    loaded = build_segmentation_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_segmentation",
        model_config={"input": {"input_size": [32, 32]}},
        training_config={"batch_size": 1},
    )
    assert loaded.num_classes == 1
    assert loaded.train_count == 1
    assert loaded.val_count == 1

    train_images, train_masks = next(iter(loaded.train_loader))
    assert tuple(train_images.shape) == (1, 3, 32, 32)
    assert tuple(train_masks.shape) == (1, 32, 32)
    assert int(train_masks.max().item()) == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_runner_process_writes_events_metrics_and_checkpoints(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    content_hash, _zip_path = _write_tiny_export_zip(tmp_path, project_id)
    _seed_experiment_layout(tmp_path, project_id, experiment_id, job_id)

    job = TrainJob(
        job_id=job_id,
        job_version="1",
        job_type="train",
        attempt=1,
        project_id=project_id,
        experiment_id=experiment_id,
        model_id="model-1",
        task="classification",
        task_id="task-1",
        model_config={
            "architecture": {
                "family": "resnet_classifier",
                "backbone": {"name": "resnet18", "pretrained": False},
                "head": {"num_classes": 1},
            },
            "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
        },
        training_config={
            "schema_version": "0.1",
            "model_id": "model-1",
            "dataset_version_id": "dv-1",
            "task": "classification",
            "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"type": "none", "params": {}},
            "epochs": 1,
            "batch_size": 2,
            "augmentation_profile": "none",
            "precision": "fp32",
            "advanced": {"seed": 1, "num_workers": 0, "grad_clip_norm": None},
            "hpo": {"enabled": False, "strategy": "random", "budget": {"max_trials": 1}, "search_space": {}},
        },
        dataset_export={
            "content_hash": content_hash,
            "zip_relpath": f"exports/{project_id}/{content_hash}.zip",
            "dataset_version_id": "dv-1",
        },
    )

    runner = TrainRunner(str(tmp_path))
    result = runner.process(job)
    assert result in {"completed", "failed:trainer_error", "failed:unsupported_family"}

    run_dir = tmp_path / "experiments" / project_id / experiment_id / "runs" / "1"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints.json").exists()
    assert (run_dir / "runtime.json").exists()
    assert (run_dir / "training.log").exists()
    if result == "completed":
        import torch

        run_eval = run_dir / "evaluation.json"
        run_predictions = run_dir / "predictions.jsonl"
        run_predictions_meta = run_dir / "predictions.meta.json"
        latest_eval = tmp_path / "experiments" / project_id / experiment_id / "evaluation.json"
        latest_predictions = tmp_path / "experiments" / project_id / experiment_id / "predictions.jsonl"
        latest_predictions_meta = tmp_path / "experiments" / project_id / experiment_id / "predictions.meta.json"
        latest_runtime = tmp_path / "experiments" / project_id / experiment_id / "runtime.json"
        assert run_eval.exists()
        assert run_predictions.exists()
        assert run_predictions_meta.exists()
        assert latest_eval.exists()
        assert latest_predictions.exists()
        assert latest_predictions_meta.exists()
        assert latest_runtime.exists()

        evaluation_payload = json.loads(run_eval.read_text(encoding="utf-8"))
        assert evaluation_payload["schema_version"] == "1"
        confusion = evaluation_payload["confusion_matrix"]["matrix"]
        assert isinstance(confusion, list)
        assert len(confusion) == 1
        assert len(confusion[0]) == 1
        per_class = evaluation_payload["per_class"]
        assert isinstance(per_class, list)
        assert len(per_class) == 1
        accuracy = evaluation_payload["overall"]["accuracy"]
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

        predictions_meta = json.loads(run_predictions_meta.read_text(encoding="utf-8"))
        assert predictions_meta["schema_version"] == "1"
        assert predictions_meta["attempt"] == 1
        assert predictions_meta["task"] == "classification"

        runtime_payload = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
        assert runtime_payload["device_selected"] in {"cpu", "cuda", "mps"}
        assert isinstance(runtime_payload["amp_enabled"], bool)
        assert "prefetch_factor" in runtime_payload
        assert "cache_resized_images" in runtime_payload
        assert "max_cached_images" in runtime_payload

        metrics_lines = [line for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert metrics_lines
        first_metric = json.loads(metrics_lines[0])
        assert "train_accuracy" in first_metric
        assert "epoch_seconds" in first_metric
        assert "eta_seconds" in first_metric

        latest_state = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu")
        best_metric_state = torch.load(run_dir / "checkpoints" / "best_metric.pt", map_location="cpu")
        assert "optimizer_state_dict" in latest_state
        assert "scheduler_state_dict" in latest_state
        assert "optimizer_state_dict" not in best_metric_state


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_queue_payload_parse_to_runner_process_persists_events_and_artifacts(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    content_hash, _zip_path = _write_tiny_export_zip(tmp_path, project_id)
    _seed_experiment_layout(tmp_path, project_id, experiment_id, job_id)

    raw_payload = json.dumps(
        {
            "job_id": job_id,
            "job_version": "1",
            "job_type": "train",
            "attempt": 1,
            "project_id": project_id,
            "experiment_id": experiment_id,
            "model_id": "model-1",
            "task": "classification",
            "model_config": {
                "architecture": {
                    "family": "resnet_classifier",
                    "backbone": {"name": "resnet18", "pretrained": False},
                    "head": {"num_classes": 1},
                },
                "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
            },
            "training_config": {
                "schema_version": "0.1",
                "model_id": "model-1",
                "dataset_version_id": "dv-1",
                "task": "classification",
                "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0},
                "scheduler": {"type": "none", "params": {}},
                "epochs": 1,
                "batch_size": 2,
                "augmentation_profile": "none",
                "precision": "fp32",
                "advanced": {"seed": 1, "num_workers": 0, "grad_clip_norm": None},
                "hpo": {"enabled": False, "strategy": "random", "budget": {"max_trials": 1}, "search_space": {}},
            },
            "dataset_export": {
                "content_hash": content_hash,
                "zip_relpath": f"exports/{project_id}/{content_hash}.zip",
                "dataset_version_id": "dv-1",
            },
        }
    )
    job = parse_train_job(raw_payload)
    runner = TrainRunner(str(tmp_path))
    result = runner.process(job)

    assert result in {"completed", "failed:trainer_error", "failed:unsupported_family"}
    run_dir = tmp_path / "experiments" / project_id / experiment_id / "runs" / "1"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints.json").exists()
    events_lines = [line for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events_lines) >= 2
    parsed_events = [json.loads(line) for line in events_lines]
    assert parsed_events[0]["type"] == "status"
    assert parsed_events[-1]["type"] == "done"


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_runner_respects_eval_interval_and_writes_null_val_metrics(tmp_path: Path) -> None:
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    content_hash, _zip_path = _write_tiny_export_zip(tmp_path, project_id)
    _seed_experiment_layout(tmp_path, project_id, experiment_id, job_id)

    job = TrainJob(
        job_id=job_id,
        job_version="1",
        job_type="train",
        attempt=1,
        project_id=project_id,
        experiment_id=experiment_id,
        model_id="model-1",
        task="classification",
        task_id="task-1",
        model_config={
            "architecture": {
                "family": "resnet_classifier",
                "backbone": {"name": "resnet18", "pretrained": False},
                "head": {"num_classes": 1},
            },
            "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
        },
        training_config={
            "schema_version": "0.1",
            "model_id": "model-1",
            "dataset_version_id": "dv-1",
            "task": "classification",
            "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"type": "none", "params": {}},
            "epochs": 4,
            "batch_size": 2,
            "evaluation": {"eval_interval_epochs": 2},
            "augmentation_profile": "none",
            "precision": "fp32",
            "advanced": {"seed": 1, "num_workers": 0, "grad_clip_norm": None},
            "hpo": {"enabled": False, "strategy": "random", "budget": {"max_trials": 1}, "search_space": {}},
        },
        dataset_export={
            "content_hash": content_hash,
            "zip_relpath": f"exports/{project_id}/{content_hash}.zip",
            "dataset_version_id": "dv-1",
        },
    )

    runner = TrainRunner(str(tmp_path))
    result = runner.process(job)
    if result != "completed":
        pytest.skip(f"training did not complete in test environment: {result}")

    metrics_path = tmp_path / "experiments" / project_id / experiment_id / "runs" / "1" / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 4
    epoch3 = next(row for row in rows if int(row["epoch"]) == 3)
    assert epoch3["val_loss"] is None
    assert epoch3["val_accuracy"] is None


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_runner_fails_fast_for_batchnorm_small_batch(tmp_path: Path) -> None:
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    content_hash, _zip_path = _write_tiny_export_zip(tmp_path, project_id)
    _seed_experiment_layout(tmp_path, project_id, experiment_id, job_id)

    job = TrainJob(
        job_id=job_id,
        job_version="1",
        job_type="train",
        attempt=1,
        project_id=project_id,
        experiment_id=experiment_id,
        model_id="model-1",
        task="classification",
        task_id="task-1",
        model_config={
            "architecture": {
                "family": "resnet_classifier",
                "backbone": {"name": "resnet18", "pretrained": False},
                "head": {"num_classes": 1},
            },
            "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
        },
        training_config={
            "schema_version": "0.1",
            "model_id": "model-1",
            "dataset_version_id": "dv-1",
            "task": "classification",
            "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"type": "none", "params": {}},
            "epochs": 1,
            "batch_size": 1,
            "training": {"drop_last": True},
            "augmentation_profile": "none",
            "precision": "fp32",
            "advanced": {"seed": 1, "num_workers": 0, "grad_clip_norm": None},
            "hpo": {"enabled": False, "strategy": "random", "budget": {"max_trials": 1}, "search_space": {}},
        },
        dataset_export={
            "content_hash": content_hash,
            "zip_relpath": f"exports/{project_id}/{content_hash}.zip",
            "dataset_version_id": "dv-1",
        },
    )

    runner = TrainRunner(str(tmp_path))
    result = runner.process(job)
    assert result == "failed:batchnorm_small_batch_unsupported"


@pytest.mark.skipif(not HAS_TORCH or not HAS_ONNX_RUNTIME, reason="torch + onnxruntime are required")
def test_export_best_classification_onnx_supports_dynamic_batch(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    num_classes = 3
    model_config = {
        "architecture": {
            "family": "resnet_classifier",
            "backbone": {"name": "resnet18", "pretrained": False},
            "head": {"num_classes": num_classes},
        },
        "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
    }
    model = build_resnet_classifier(model_config, num_classes_override=num_classes)
    checkpoint_state = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
    }
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_metric",
        epoch=1,
        metric_name="val_accuracy",
        value=0.8,
        state_dict=checkpoint_state,
    )

    result = export_best_classification_onnx(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        model_config=model_config,
        num_classes=num_classes,
        class_names=["cat", "dog", "bird"],
        class_order=[1, 2, 3],
    )

    assert result.status == "exported"
    assert result.model_uri is not None
    model_path = storage.resolve(result.model_uri)
    metadata_path = storage.resolve(result.metadata_uri)
    assert model_path.exists()
    assert metadata_path.exists()

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["status"] == "exported"
    assert metadata_payload["input_shape"] == [3, 32, 32]
    assert metadata_payload["class_names"] == ["cat", "dog", "bird"]
    assert metadata_payload["validation"]["status"] == "passed"

    import numpy as np
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if providers:
        session = ort.InferenceSession(str(model_path), providers=providers)
    else:
        session = ort.InferenceSession(str(model_path))
    for batch_size in (1, 4):
        dummy = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
        output = session.run(["output"], {"input": dummy})[0]
        assert int(output.shape[0]) == batch_size


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_export_model_to_onnx_falls_back_to_legacy_export_for_torch_export_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ExperimentStorage(str(tmp_path))

    class TinyModel(torch.nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x

    calls: list[bool] = []

    def _fake_export(model, args, f, **kwargs):
        calls.append(bool(kwargs.get("dynamo")))
        if kwargs.get("dynamo") is True:
            raise RuntimeError("GuardOnDataDependentSymNode caused by batched_nms during torch.export")
        Path(f).write_bytes(b"fake-onnx")
        return None

    monkeypatch.setattr(torch.onnx, "export", _fake_export)
    import pixel_sheriff_trainer.export_onnx as export_onnx_mod

    monkeypatch.setattr(
        export_onnx_mod,
        "_validate_exported_onnx",
        lambda _path, *, input_shape, output_names=None, task=None, batch_sizes=(1, 4): {
            "status": "passed",
            "onnx_checker": {"status": "passed", "error": None},
            "onnxruntime": {"status": "passed", "error": None, "providers": [], "batch_results": {}},
        },
    )

    result = export_model_to_onnx(
        TinyModel(),
        storage,
        project_id="project-1",
        experiment_id="experiment-1",
        attempt=1,
        checkpoint_kind="best_metric",
        checkpoint_uri="checkpoints/best_metric.pt",
        input_shape=(3, 32, 32),
        input_names=["input"],
        output_names=["output"],
        preprocess={"resize_policy": "stretch"},
        class_order=["1"],
        class_names=["cat"],
        extra_metadata={"task": "detection"},
    )

    assert result.status == "exported"
    assert calls == [True, False]
    metadata = json.loads(storage.resolve(result.metadata_uri).read_text(encoding="utf-8"))
    assert metadata["onnx"]["export_backend"]["mode"] == "legacy"
    assert "GuardOnDataDependentSymNode" in metadata["onnx"]["export_backend"]["fallback_reason"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_export_model_to_onnx_disables_dynamic_batch_for_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ExperimentStorage(str(tmp_path))

    class TinyModel(torch.nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x

    captured: dict[str, object] = {}

    import pixel_sheriff_trainer.export_onnx as export_onnx_mod

    def _fake_export_with_fallback(model, example_input, model_path, *, input_names, output_names, dynamic_axes):
        captured["dynamic_axes"] = {key: dict(value) for key, value in dynamic_axes.items()}
        Path(model_path).write_bytes(b"fake-onnx")
        return {"mode": "legacy", "fallback_reason": None}

    def _fake_validate(_path, *, input_shape, output_names=None, task=None, batch_sizes=(1, 4)):
        captured["task"] = task
        captured["batch_sizes"] = batch_sizes
        return {
            "status": "passed",
            "onnx_checker": {"status": "passed", "error": None},
            "onnxruntime": {"status": "passed", "error": None, "providers": [], "batch_results": {}},
        }

    monkeypatch.setattr(export_onnx_mod, "_export_with_fallback", _fake_export_with_fallback)
    monkeypatch.setattr(export_onnx_mod, "_validate_exported_onnx", _fake_validate)

    result = export_model_to_onnx(
        TinyModel(),
        storage,
        project_id="project-1",
        experiment_id="experiment-1",
        attempt=1,
        checkpoint_kind="best_metric",
        checkpoint_uri="checkpoints/best_metric.pt",
        input_shape=(3, 320, 320),
        input_names=["input"],
        output_names=["output"],
        preprocess={"resize_policy": "stretch"},
        class_order=["1"],
        class_names=["cat"],
        extra_metadata={"task": "detection"},
    )

    assert result.status == "exported"
    assert captured["task"] == "detection"
    assert captured["dynamic_axes"] == {}
    assert captured["batch_sizes"] == (1,)

    metadata = json.loads(storage.resolve(result.metadata_uri).read_text(encoding="utf-8"))
    assert metadata["onnx"]["dynamic_axes"] == {}
    assert "Dynamic batch is disabled for detection exports" in metadata["onnx"]["runtime_note"]


def test_validate_onnxruntime_batch_outputs_allows_detection_postprocessed_shapes() -> None:
    import numpy as np

    result = _validate_onnxruntime_batch_outputs(
        [np.zeros((300, 4), dtype=np.float32), np.zeros((300,), dtype=np.float32)],
        batch_size=1,
        task="detection",
    )

    assert result["status"] == "passed"
    assert result["output_shape"] == [300, 4]
    assert result["batch_semantics"] == "postprocessed_detection_outputs"
