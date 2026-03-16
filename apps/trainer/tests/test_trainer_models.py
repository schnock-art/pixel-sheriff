from __future__ import annotations

import os
from pathlib import Path

import pytest

from .trainer_test_helpers import (
    HAS_TORCH,
    build_classifier_model,
    configure_torchvision_cache,
    resolve_torchvision_cache_root,
    torch,
)

def test_build_classifier_model_uses_pretrained_weights_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.models as tv_models

    captured: dict[str, object] = {}
    expected_default = object()

    class FakeWeights:
        DEFAULT = expected_default

    class FakeResNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):  # type: ignore[override]
            return self.fc(torch.ones((x.shape[0], 4), dtype=x.dtype))

    def _fake_ctor(*, weights=None, **_kwargs):
        captured["weights"] = weights
        return FakeResNet()

    monkeypatch.setattr(tv_models, "ResNet18_Weights", FakeWeights)
    monkeypatch.setattr(tv_models, "resnet18", _fake_ctor)

    model = build_classifier_model(
        {
            "architecture": {
                "family": "resnet_classifier",
                "backbone": {"name": "resnet18", "pretrained": True},
                "head": {"num_classes": 3},
            }
        },
        num_classes_override=3,
    )

    assert captured["weights"] is expected_default
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.fc.out_features == 3


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_build_classifier_model_supports_efficientnet_v2_family() -> None:
    model = build_classifier_model(
        {
            "architecture": {
                "family": "efficientnet_v2_classifier",
                "backbone": {"name": "efficientnet_v2_s", "pretrained": False},
                "head": {"num_classes": 5},
            }
        },
        num_classes_override=5,
    )
    outputs = model(torch.randn(1, 3, 128, 128))
    assert tuple(outputs.shape) == (1, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_build_classifier_model_fails_when_pretrained_weights_are_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.models as tv_models

    class FakeWeights:
        DEFAULT = object()

    def _failing_ctor(*, weights=None, **_kwargs):
        raise RuntimeError(f"download failed for weights={weights}")

    monkeypatch.setattr(tv_models, "ResNet18_Weights", FakeWeights)
    monkeypatch.setattr(tv_models, "resnet18", _failing_ctor)

    with pytest.raises(ValueError, match="pretrained_weights_unavailable"):
        build_classifier_model(
            {
                "architecture": {
                    "family": "resnet_classifier",
                    "backbone": {"name": "resnet18", "pretrained": True},
                    "head": {"num_classes": 2},
                }
            },
            num_classes_override=2,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_detection_builder_uses_backbone_weights_when_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.detection.train as detection_train
    import torchvision.models as tv_models
    import torchvision.models.detection as tv_det

    captured: dict[str, object] = {}
    expected_default = object()

    class FakeWeights:
        DEFAULT = expected_default

    def _fake_builder(*, weights=None, weights_backbone=None, num_classes=None, **_kwargs):
        captured["weights"] = weights
        captured["weights_backbone"] = weights_backbone
        captured["num_classes"] = num_classes
        return torch.nn.Identity()

    monkeypatch.setattr(tv_models, "ResNet50_Weights", FakeWeights)
    monkeypatch.setattr(tv_det, "retinanet_resnet50_fpn", _fake_builder)

    detection_train._build_detection_model(
        {
            "architecture": {
                "family": "retinanet",
                "backbone": {"name": "resnet50", "pretrained": True},
            }
        },
        num_classes=4,
    )

    assert captured["weights"] is None
    assert captured["weights_backbone"] is expected_default
    assert captured["num_classes"] == 4


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_segmentation_builder_uses_backbone_weights_when_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.segmentation.train as segmentation_train
    import torchvision.models as tv_models
    import torchvision.models.segmentation as tv_seg

    captured: dict[str, object] = {}
    expected_default = object()

    class FakeWeights:
        DEFAULT = expected_default

    def _fake_builder(*, weights=None, weights_backbone=None, num_classes=None, **_kwargs):
        captured["weights"] = weights
        captured["weights_backbone"] = weights_backbone
        captured["num_classes"] = num_classes
        return torch.nn.Identity()

    monkeypatch.setattr(tv_models, "ResNet101_Weights", FakeWeights)
    monkeypatch.setattr(tv_seg, "deeplabv3_resnet101", _fake_builder)

    segmentation_train._build_deeplabv3(
        {
            "architecture": {
                "backbone": {"name": "resnet101", "pretrained": True},
            }
        },
        num_classes=3,
    )

    assert captured["weights"] is None
    assert captured["weights_backbone"] is expected_default
    assert captured["num_classes"] == 4


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_configure_torchvision_cache_uses_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCHVISION_CACHE_ROOT", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)

    expected_root = (tmp_path / "model_weights" / "torchvision").resolve()
    assert resolve_torchvision_cache_root(str(tmp_path)) == expected_root

    configured_root = configure_torchvision_cache(str(tmp_path))
    assert configured_root == expected_root
    assert Path(os.environ["TORCH_HOME"]) == expected_root
    assert Path(torch.hub.get_dir()) == expected_root / "hub"
