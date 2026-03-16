from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from .trainer_test_helpers import (
    ExperimentStorage,
    HAS_ONNX_RUNTIME,
    HAS_TORCH,
    _validate_onnxruntime_batch_outputs,
    build_resnet_classifier,
    export_best_classification_onnx,
    export_model_to_onnx,
    torch,
)

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
