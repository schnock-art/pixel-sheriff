from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from .trainer_test_helpers import (
    DetectionEpochMetrics,
    DetectionEvaluation,
    HAS_TORCH,
    _write_tiny_coco_export_zip,
    build_detection_loaders,
    build_segmentation_loaders,
    run_detection_training,
    torch,
)

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
    checkpoints: list[tuple[str, int, str | None, float | None]] = []
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
        on_checkpoint=lambda kind, epoch, metric_name, value, _payload: checkpoints.append((kind, epoch, metric_name, value)),
        device=torch.device("cpu"),
        resume_state=None,
    )
    assert status == "completed"
    assert evaluation is not None
    assert len(captured_epochs) == 1
    assert captured_epochs[0].precision is not None
    assert captured_epochs[0].recall is not None
    assert captured_epochs[0].tp is not None
    assert captured_epochs[0].fp is not None
    assert captured_epochs[0].fn is not None
    assert captured_epochs[0].duplicate_fp is not None
    assert checkpoints
    assert checkpoints[0][2] == "val_map_50_95"


def test_detection_training_uses_map50_95_for_best_metric_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    import pixel_sheriff_trainer.detection.train as detection_train
    from pixel_sheriff_trainer.detection.evaluation.types import DetectionOverallMetrics

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

    evaluations = iter(
        [
            DetectionEvaluation(
                mAP50=0.90,
                mAP50_95=0.40,
                overall=DetectionOverallMetrics(
                    mAP50=0.90,
                    mAP50_95=0.40,
                    precision=0.8,
                    recall=0.7,
                    tp=7,
                    fp=2,
                    fn=3,
                    duplicate_fp=1,
                    matched_mean_iou=0.6,
                ),
            ),
            DetectionEvaluation(
                mAP50=0.80,
                mAP50_95=0.50,
                overall=DetectionOverallMetrics(
                    mAP50=0.80,
                    mAP50_95=0.50,
                    precision=0.85,
                    recall=0.75,
                    tp=8,
                    fp=1,
                    fn=2,
                    duplicate_fp=0,
                    matched_mean_iou=0.7,
                ),
            ),
        ]
    )

    monkeypatch.setattr(detection_train, "_build_retinanet", lambda _num_classes, *, pretrained=False: FakeDetector())
    monkeypatch.setattr(detection_train, "evaluate_detection", lambda *_args, **_kwargs: next(evaluations))

    sample_image = torch.zeros((3, 8, 8), dtype=torch.float32)
    sample_target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    train_loader = [([sample_image], [sample_target])]
    val_loader = [([sample_image], [sample_target])]

    checkpoints: list[tuple[str, int, str | None, float | None]] = []
    status, evaluation = run_detection_training(
        model_config={"architecture": {"family": "retinanet"}},
        training_config={
            "optimizer": {"lr": 0.0001, "weight_decay": 0.0},
            "scheduler": {"type": "none"},
            "logging": {"save_every_epochs": 1, "keep_best": True},
            "evaluation": {"eval_interval_epochs": 1},
            "epochs": 2,
            "batch_size": 1,
        },
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=1,
        should_cancel=lambda: False,
        on_epoch=lambda _row: None,
        on_checkpoint=lambda kind, epoch, metric_name, value, _payload: checkpoints.append((kind, epoch, metric_name, value)),
        device=torch.device("cpu"),
        resume_state=None,
    )

    best_metric_events = [row for row in checkpoints if row[0] == "best_metric"]
    latest_events = [row for row in checkpoints if row[0] == "latest"]

    assert status == "completed"
    assert evaluation is not None
    assert best_metric_events[-1] == ("best_metric", 2, "val_map_50_95", 0.5)
    assert latest_events[-1] == ("latest", 2, "val_map_50_95", 0.5)


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

    def _fake_builder(_num_classes: int, *, pretrained: bool = False) -> torch.nn.Module:
        called["ssdlite"] = True
        called["pretrained"] = pretrained
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
    assert called["pretrained"] is False
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

    def _fake_eval(model, _val_loader, device, **_kwargs):
        eval_devices.append(str(device))
        if str(device) == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return DetectionEvaluation(mAP50=0.25, mAP50_95=0.15)

    monkeypatch.setattr(detection_train, "_build_retinanet", lambda _num_classes, *, pretrained=False: fake_model)
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


@pytest.mark.parametrize(
    ("raw_label", "label_offset"),
    [
        (0, 0),
        (1, 1),
    ],
)
def test_evaluate_detection_normalizes_prediction_and_ground_truth_labels(
    raw_label: int,
    label_offset: int,
) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    from pixel_sheriff_trainer.detection.eval import evaluate_detection

    class FakeModel(torch.nn.Module):
        def eval(self):  # type: ignore[override]
            return self

        def forward(self, _images):  # type: ignore[override]
            return [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 30.0, 25.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.95], dtype=torch.float32),
                    "labels": torch.tensor([raw_label], dtype=torch.int64),
                }
            ]

    class FakeLoader(list):
        def __init__(self, dataset, batches):
            super().__init__(batches)
            self.dataset = dataset

    dataset = SimpleNamespace(
        samples=[
            SimpleNamespace(
                image_id="img-1",
                asset_id="asset-1",
                relative_path="assets/img-1.jpg",
                width=100,
                height=80,
            )
        ],
        annotations={
            "img-1": [
                {
                    "id": "ann-1",
                    "category_id": raw_label,
                    "bbox": [10.0, 10.0, 20.0, 15.0],
                    "area": 300.0,
                }
            ]
        },
        target_width=100,
        target_height=80,
    )
    loader = FakeLoader(
        dataset,
        [
            (
                [torch.zeros((3, 100, 100), dtype=torch.float32)],
                [{"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}],
            )
        ],
    )

    evaluation = evaluate_detection(
        FakeModel(),
        loader,
        torch.device("cpu"),
        num_classes=1,
        class_names=["Boat"],
        class_order=["boat"],
        label_offset=label_offset,
    )

    assert evaluation.mAP50 == pytest.approx(1.0)
    assert evaluation.diagnostics is not None
    assert evaluation.diagnostics.prediction_rows[0].class_index == 0
    assert evaluation.diagnostics.prediction_rows[0].status == "matched_tp"


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

    monkeypatch.setattr(detection_train, "_build_retinanet", lambda _num_classes, *, pretrained=False: FakeDetector())
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
