from __future__ import annotations

import copy

import pytest
import torch

from sheriff_api.ml.model_factory import ModelFactoryValidationError, build_model


def _base_classifier_config() -> dict:
    return {
        "schema_version": "1.0",
        "name": "classifier-with-embedding",
        "created_at": "2026-03-01T00:00:00Z",
        "source_dataset": {
            "manifest_id": "manifest-123",
            "task": "classification",
            "num_classes": 3,
            "class_order": ["10", "11", "12"],
            "class_names": ["car", "bus", "truck"],
        },
        "input": {
            "image_channels": 3,
            "input_size": [224, 224],
            "resize_policy": "letterbox",
            "normalization": {"type": "imagenet"},
        },
        "architecture": {
            "family": "resnet_classifier",
            "framework": "torchvision",
            "precision": "fp32",
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"type": "none"},
            "head": {"type": "linear", "num_classes": 3},
        },
        "loss": {"type": "classification_cross_entropy"},
        "outputs": {
            "primary": {
                "name": "classification_logits",
                "type": "task_output",
                "task": "classification",
                "format": "classification_logits",
            },
            "aux": [
                {
                    "name": "embedding_vec",
                    "type": "embedding",
                    "source": {"block": "backbone", "tap": "avgpool"},
                    "projection": {"type": "linear", "out_dim": 128, "normalize": "l2"},
                }
            ],
        },
        "export": {
            "onnx": {
                "enabled": True,
                "opset": 17,
                "dynamic_shapes": {"enabled": False, "batch": False, "height_width": False},
                "output_names": ["classification_logits", "embedding_vec"],
            }
        },
    }


def _efficientnet_classifier_config() -> dict:
    config = copy.deepcopy(_base_classifier_config())
    config["input"]["input_size"] = [384, 384]
    config["architecture"]["family"] = "efficientnet_v2_classifier"
    config["architecture"]["backbone"]["name"] = "efficientnet_v2_s"
    return config


def test_model_factory_builds_classifier_with_embedding_aux() -> None:
    config = _base_classifier_config()
    built = build_model(config)
    assert built.output_names == ["classification_logits", "embedding_vec"]

    x = torch.randn(2, 3, 224, 224)
    outputs = built.model(x)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    logits, embedding = outputs
    assert logits.shape == (2, 3)
    assert embedding.shape == (2, 128)


def test_avgpool_alias_resolves_to_global_pool() -> None:
    config = _base_classifier_config()
    config["outputs"]["aux"][0]["source"]["tap"] = "avgpool"
    built = build_model(config, verify_metadata=True)
    outputs = built.model(torch.randn(1, 3, 224, 224))
    assert outputs[1].shape == (1, 128)


def test_model_factory_builds_efficientnet_classifier_with_embedding_aux() -> None:
    config = _efficientnet_classifier_config()
    built = build_model(config)
    outputs = built.model(torch.randn(1, 3, 384, 384))
    assert outputs[0].shape == (1, 3)
    assert outputs[1].shape == (1, 128)


def test_model_factory_rejects_resnet_only_tap_for_efficientnet() -> None:
    config = _efficientnet_classifier_config()
    config["outputs"]["aux"][0]["source"]["tap"] = "c4"
    with pytest.raises(ModelFactoryValidationError, match="Unsupported tap"):
        build_model(config)


def test_model_factory_rejects_invalid_tap_for_family() -> None:
    config = _base_classifier_config()
    config["outputs"]["aux"][0]["source"]["tap"] = "does_not_exist"
    with pytest.raises(ModelFactoryValidationError, match="Unsupported tap"):
        build_model(config)


def test_model_factory_rejects_output_name_mismatch() -> None:
    config = _base_classifier_config()
    config["export"]["onnx"]["output_names"] = ["classification_logits"]
    with pytest.raises(ModelFactoryValidationError, match="must be included in export.onnx.output_names"):
        build_model(config)


def test_model_factory_rejects_head_num_classes_mismatch() -> None:
    config = copy.deepcopy(_base_classifier_config())
    config["architecture"]["head"]["num_classes"] = 2
    with pytest.raises(ModelFactoryValidationError, match="must match source_dataset.num_classes"):
        build_model(config)
