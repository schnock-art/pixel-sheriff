from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pixel_sheriff_trainer.inference.app import _top_k_predictions
from pixel_sheriff_trainer.inference.preprocess import preprocess_asset


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


def test_top_k_predictions_are_deterministically_sorted() -> None:
    logits = np.array([[1.0, 3.0, 3.0, 2.0]], dtype=np.float32)
    rows, output_dim = _top_k_predictions(logits, top_k=3)
    assert output_dim == 4
    assert [row.class_index for row in rows] == [1, 2, 3]
