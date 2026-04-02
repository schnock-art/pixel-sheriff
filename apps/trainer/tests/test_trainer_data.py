from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from .trainer_test_helpers import (
    HAS_TORCH,
    _write_tiny_coco_export_zip,
    _write_tiny_export_zip,
    apply_detection_augmentation,
    apply_segmentation_augmentation,
    build_classification_loaders,
    build_detection_loaders,
    resolve_training_augmentation,
)

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


def test_resolve_training_augmentation_preserves_legacy_non_classification_behavior() -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    legacy_detection_mode, legacy_detection_steps = resolve_training_augmentation(
        {"task": "detection", "augmentation_profile": "light"},
        "detection",
    )
    assert legacy_detection_mode == "none"
    assert legacy_detection_steps == []

    classification_mode, classification_steps = resolve_training_augmentation(
        {"task": "classification", "augmentation_profile": "medium"},
        "classification",
    )
    assert classification_mode == "medium"
    assert [step.type for step in classification_steps] == ["horizontal_flip", "color_jitter"]

    custom_mode, custom_steps = resolve_training_augmentation(
        {
            "task": "detection",
            "augmentation_profile": "custom",
            "augmentation_spec_version": 1,
            "augmentation_steps": [{"type": "rotate", "p": 1.0, "params": {"degrees": 6}}],
        },
        "detection",
    )
    assert custom_mode == "custom"
    assert custom_steps[0].params == {"min_degrees": -6.0, "max_degrees": 6.0}


def test_apply_detection_augmentation_rotates_boxes_without_dropping_valid_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    from PIL import Image

    monkeypatch.setattr("pixel_sheriff_trainer.augmentation.random.random", lambda: 0.0)
    monkeypatch.setattr("pixel_sheriff_trainer.augmentation.random.uniform", lambda _low, _high: 8.0)

    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    rotated_image, rotated_boxes, rotated_labels = apply_detection_augmentation(
        image,
        [[8.0, 8.0, 20.0, 20.0]],
        [1],
        [resolve_training_augmentation(
            {
                "task": "detection",
                "augmentation_profile": "custom",
                "augmentation_spec_version": 1,
                "augmentation_steps": [{"type": "rotate", "p": 1.0, "params": {"min_degrees": -8, "max_degrees": 8}}],
            },
            "detection",
        )[1][0]],
    )
    assert rotated_image.size == (32, 32)
    assert rotated_labels == [1]
    assert len(rotated_boxes) == 1
    assert rotated_boxes[0][2] > rotated_boxes[0][0]
    assert rotated_boxes[0][3] > rotated_boxes[0][1]


def test_apply_segmentation_augmentation_preserves_mask_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")

    from PIL import Image

    monkeypatch.setattr("pixel_sheriff_trainer.augmentation.random.random", lambda: 0.0)
    monkeypatch.setattr("pixel_sheriff_trainer.augmentation.random.uniform", lambda _low, _high: 6.0)

    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    mask = Image.new("L", (16, 16), color=0)
    for x in range(4, 10):
        for y in range(4, 10):
            mask.putpixel((x, y), 1)

    rotated_image, rotated_mask = apply_segmentation_augmentation(
        image,
        mask,
        resolve_training_augmentation(
            {
                "task": "segmentation",
                "augmentation_profile": "custom",
                "augmentation_spec_version": 1,
                "augmentation_steps": [{"type": "rotate", "p": 1.0, "params": {"degrees": 6}}],
            },
            "segmentation",
        )[1],
    )
    assert rotated_image.size == (16, 16)
    assert rotated_mask.size == (16, 16)
    assert set(rotated_mask.getdata()).issubset({0, 1})


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


def test_detection_loader_honors_runtime_worker_settings(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch/torchvision not available")
    project_id = str(uuid.uuid4())
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)
    loaded = build_detection_loaders(
        export_zip_path=zip_path,
        workdir=tmp_path / "workdir_detection_runtime",
        model_config={"input": {"input_size": [32, 32]}},
        training_config={
            "batch_size": 1,
            "runtime": {"num_workers": 1, "prefetch_factor": 3, "persistent_workers": True},
            "training": {"drop_last": False},
        },
    )
    assert loaded.train_loader.drop_last is False
    assert int(getattr(loaded.train_loader, "prefetch_factor", 0)) == 3
    assert bool(getattr(loaded.train_loader, "persistent_workers", False)) is True


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
