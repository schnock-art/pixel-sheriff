from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pixel_sheriff_trainer.inference.app import _top_k_predictions
from pixel_sheriff_trainer.inference.preprocess import (
    PreprocessContext,
    preprocess_asset,
    preprocess_asset_with_context,
    remap_bbox_xyxy_to_original_xywh,
)


def test_preprocess_outputs_nchw_float32(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 16), color=(255, 0, 0)).save(image_path)
    metadata = {
        "preprocess": {
            "resize_policy": "stretch",
            "resize": {"width": 20, "height": 10},
            "normalization": {"type": "none"},
        }
    }
    tensor = preprocess_asset(image_path, metadata)
    assert tensor.shape == (1, 3, 10, 20)
    assert tensor.dtype == np.float32


def test_preprocess_asset_with_context_tracks_stretch_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), color=(255, 0, 0)).save(image_path)
    metadata = {
        "preprocess": {
            "resize_policy": "stretch",
            "resize": {"width": 20, "height": 10},
            "normalization": {"type": "none"},
        }
    }

    tensor, context = preprocess_asset_with_context(image_path, metadata)

    assert tensor.shape == (1, 3, 10, 20)
    assert context == PreprocessContext(
        original_width=64,
        original_height=48,
        target_width=20,
        target_height=10,
        resize_policy="stretch",
        resized_width=20,
        resized_height=10,
        offset_x=0,
        offset_y=0,
    )


def test_remap_bbox_xyxy_to_original_xywh_reverses_stretch_resize() -> None:
    context = PreprocessContext(
        original_width=640,
        original_height=480,
        target_width=320,
        target_height=320,
        resize_policy="stretch",
        resized_width=320,
        resized_height=320,
        offset_x=0,
        offset_y=0,
    )

    bbox = remap_bbox_xyxy_to_original_xywh([32.0, 64.0, 160.0, 224.0], context)

    assert bbox == [64.0, 96.0, 256.0, 240.0]


def test_remap_bbox_xyxy_to_original_xywh_reverses_letterbox_resize() -> None:
    context = PreprocessContext(
        original_width=640,
        original_height=480,
        target_width=320,
        target_height=320,
        resize_policy="letterbox",
        resized_width=320,
        resized_height=240,
        offset_x=0,
        offset_y=40,
    )

    bbox = remap_bbox_xyxy_to_original_xywh([32.0, 72.0, 160.0, 200.0], context)

    assert bbox == [64.0, 64.0, 256.0, 256.0]


def test_top_k_predictions_are_deterministically_sorted() -> None:
    logits = np.array([[1.0, 3.0, 3.0, 2.0]], dtype=np.float32)
    rows, output_dim = _top_k_predictions(logits, top_k=3)
    assert output_dim == 4
    assert [row.class_index for row in rows] == [1, 2, 3]
