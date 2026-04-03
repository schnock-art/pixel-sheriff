from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import uuid

import numpy as np
import pytest

from .trainer_test_helpers import (
    ExperimentStorage,
    HAS_TORCH,
    _write_tiny_coco_export_zip,
    _write_tiny_export_zip,
    save_checkpoint,
    torch,
)


def _install_fake_onnxruntime_quantization(
    monkeypatch: pytest.MonkeyPatch,
    *,
    quantize_static,
) -> None:
    fake_onnxruntime = ModuleType("onnxruntime")
    fake_quantization = ModuleType("onnxruntime.quantization")
    fake_quantization.QuantFormat = SimpleNamespace(QDQ="QDQ")
    fake_quantization.QuantType = SimpleNamespace(QInt8="QInt8")
    fake_quantization.quantize_static = quantize_static
    fake_onnxruntime.quantization = fake_quantization
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", fake_quantization)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_variant_task_support_is_family_aware_for_real_qat() -> None:
    import pixel_sheriff_trainer.variants as variants_mod

    classification_support = variants_mod.variant_task_support(
        "classification",
        {"architecture": {"family": "resnet_classifier", "backbone": {"name": "resnet18"}}},
    )
    assert classification_support == {
        "fp16_supported": True,
        "fp16_reason": None,
        "ptq_supported": True,
        "qat_supported": True,
        "qat_reason": None,
        "qat_mode": "fake_quant",
        "qat_experimental": False,
        "qat_warning": None,
    }

    ssdlite_support = variants_mod.variant_task_support(
        "detection",
        {"architecture": {"family": "ssdlite320_mobilenet_v3_large", "backbone": {"name": "mobilenet_v3_large"}}},
    )
    assert ssdlite_support == {
        "fp16_supported": True,
        "fp16_reason": None,
        "ptq_supported": True,
        "qat_supported": True,
        "qat_reason": None,
        "qat_mode": "fake_quant",
        "qat_experimental": True,
        "qat_warning": "Real fake-quant QAT is experimental for SSDLite and still exports through float ONNX + ORT QDQ.",
    }

    retinanet_support = variants_mod.variant_task_support(
        "detection",
        {"architecture": {"family": "retinanet", "backbone": {"name": "resnet50"}}},
    )
    assert retinanet_support == {
        "fp16_supported": True,
        "fp16_reason": None,
        "ptq_supported": True,
        "qat_supported": False,
        "qat_reason": "Real fake-quant QAT v1 is not supported for detection family 'retinanet'",
        "qat_mode": None,
        "qat_experimental": False,
        "qat_warning": None,
    }

    segmentation_support = variants_mod.variant_task_support("segmentation")
    assert segmentation_support == {
        "fp16_supported": False,
        "fp16_reason": "FP16 is not supported for this task",
        "ptq_supported": False,
        "qat_supported": False,
        "qat_reason": "QAT is not supported for this task",
        "qat_mode": None,
        "qat_experimental": False,
        "qat_warning": None,
    }


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_ptq_variant_quantizes_detection_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.variants as variants_mod

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)

    base_model_path = storage.variant_onnx_model_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP32)
    base_metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP32)
    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    base_model_path.write_bytes(b"fake-fp32-model")
    base_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "task": "detection",
                "class_order": ["1"],
                "class_names": ["flower"],
                "preprocess": {
                    "resize_policy": "stretch",
                    "resize": {"width": 32, "height": 32},
                    "normalization": {"type": "none"},
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

    quantize_calls: list[tuple[str, str]] = []
    preprocess_calls: list[str] = []

    def _fake_quantize_static(model_input, model_output, calibration_reader, **_kwargs):
        quantize_calls.append((str(model_input), str(model_output)))
        assert calibration_reader.get_next() is not None
        calibration_reader.rewind()
        Path(model_output).write_bytes(b"fake-ptq-model")

    _install_fake_onnxruntime_quantization(monkeypatch, quantize_static=_fake_quantize_static)
    monkeypatch.setattr(variants_mod, "_onnx_session", lambda _path, providers=None, cpu_only=False: FakeSession())
    monkeypatch.setattr(
        variants_mod,
        "preprocess_asset",
        lambda path, _metadata: preprocess_calls.append(Path(path).name) or np.zeros((1, 3, 32, 32), dtype=np.float32),
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_evaluations",
        lambda *args, **kwargs: {"val": {"status": "ready", "overall": {"mAP50_95": 0.61}}},
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_benchmark",
        lambda *args, **kwargs: {"status": "ready", "benchmark": {"status": "ready", "mean_latency_ms": 12.5}, "devices": {"cpu": {"status": "ready", "mean_latency_ms": 12.5}}},
    )

    row = variants_mod.run_ptq_variant(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        task="detection",
        dataset_export={"zip_relpath": str(zip_path.relative_to(tmp_path)).replace("\\", "/")},
        checkpoint_kind="best_metric",
        calibration_max_samples=8,
    )

    assert row["status"] == "ready"
    assert row["quantized"] is True
    assert row["quantization_strategy"] == "static_ptq"
    assert row["preferred"] is True
    assert preprocess_calls == ["coco_0.png"]
    assert len(quantize_calls) == 1

    metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_PTQ_INT8)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["variant_key"] == variants_mod.VARIANT_PTQ_INT8
    assert metadata_payload["variant_kind"] == "ptq"
    assert metadata_payload["quantized"] is True
    assert metadata_payload["quantization"]["mode"] == "static_ptq"
    assert metadata_payload["quantization"]["calibration_max_samples"] == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_fp16_variant_converts_detection_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.variants as variants_mod

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)

    base_model_path = storage.variant_onnx_model_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP32)
    base_metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP32)
    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    base_model_path.write_bytes(b"fake-fp32-model")
    base_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "task": "detection",
                "class_order": ["1"],
                "class_names": ["flower"],
                "checkpoint_kind": "best_metric",
                "preprocess": {
                    "resize_policy": "stretch",
                    "resize": {"width": 32, "height": 32},
                    "normalization": {"type": "none"},
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    saved_models: list[tuple[object, str]] = []
    output_tensor_type = SimpleNamespace(elem_type=1)
    inferred_output_tensor_type = SimpleNamespace(elem_type=10)
    fake_onnx = ModuleType("onnx")
    fake_onnx.load_model = lambda path: {"path": path}
    fake_onnx.save_model = lambda model, path: saved_models.append((model, path)) or Path(path).write_bytes(b"fake-fp16-model")
    fake_onnx.shape_inference = SimpleNamespace(
        infer_shapes=lambda model: SimpleNamespace(
            graph=SimpleNamespace(
                value_info=[],
                output=[SimpleNamespace(name="/transform/Cast_3_output_0", type=SimpleNamespace(tensor_type=inferred_output_tensor_type))],
            )
        )
    )
    fake_float16 = ModuleType("onnxconverter_common.float16")
    fake_float16.convert_float_to_float16 = lambda model, keep_io_types=True: SimpleNamespace(
        converted=model,
        keep_io_types=keep_io_types,
        graph=SimpleNamespace(
            output=[SimpleNamespace(name="/transform/Cast_3_output_0", type=SimpleNamespace(tensor_type=output_tensor_type))]
        ),
    )
    fake_package = ModuleType("onnxconverter_common")
    fake_package.float16 = fake_float16
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxconverter_common", fake_package)
    monkeypatch.setitem(sys.modules, "onnxconverter_common.float16", fake_float16)
    monkeypatch.setattr(
        variants_mod,
        "_validate_fp16_variant_artifact",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "onnx_checker": {"status": "passed", "error": None},
            "onnxruntime": {"status": "passed", "error": None, "providers": ["CPUExecutionProvider"], "batch_results": {"1": {"status": "passed"}}},
        },
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_evaluations",
        lambda *args, **kwargs: {"val": {"status": "ready", "overall": {"mAP50_95": 0.62}}},
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_benchmark",
        lambda *args, **kwargs: {
            "status": "ready",
            "benchmark": {"status": "ready", "mean_latency_ms": 7.1},
            "devices": {"cpu": {"status": "ready", "mean_latency_ms": 7.1}},
        },
    )

    row = variants_mod.run_fp16_variant(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        task="detection",
        dataset_export={"zip_relpath": str(zip_path.relative_to(tmp_path)).replace("\\", "/")},
        checkpoint_kind="best_metric",
    )

    assert row["status"] == "ready"
    assert row["variant_key"] == variants_mod.VARIANT_FP16
    assert row["benchmarks"]["cpu"]["mean_latency_ms"] == 7.1
    assert len(saved_models) == 1
    assert output_tensor_type.elem_type == 10

    metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP16)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["variant_key"] == variants_mod.VARIANT_FP16
    assert metadata_payload["variant_kind"] == "fp16"
    assert metadata_payload["numeric_precision"] == "fp16"
    assert metadata_payload["quantized"] is False
    assert metadata_payload["status"] == "exported"
    assert metadata_payload["validation"]["status"] == "passed"
    assert metadata_payload["onnx"]["fp16_conversion"]["detection_output_types_repaired"] is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_fp16_variant_fails_fast_when_validation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.variants as variants_mod

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)

    base_model_path = storage.variant_onnx_model_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP32)
    base_metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP32)
    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    base_model_path.write_bytes(b"fake-fp32-model")
    base_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "task": "detection",
                "class_order": ["1"],
                "class_names": ["flower"],
                "checkpoint_kind": "best_metric",
                "preprocess": {
                    "resize_policy": "stretch",
                    "resize": {"width": 32, "height": 32},
                    "normalization": {"type": "none"},
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fake_onnx = ModuleType("onnx")
    fake_onnx.load_model = lambda path: SimpleNamespace(path=path, graph=SimpleNamespace(output=[]))
    fake_onnx.save_model = lambda _model, path: Path(path).write_bytes(b"broken-fp16-model")
    fake_onnx.shape_inference = SimpleNamespace(infer_shapes=lambda model: model)
    fake_float16 = ModuleType("onnxconverter_common.float16")
    fake_float16.convert_float_to_float16 = lambda model, keep_io_types=True: model
    fake_package = ModuleType("onnxconverter_common")
    fake_package.float16 = fake_float16
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxconverter_common", fake_package)
    monkeypatch.setitem(sys.modules, "onnxconverter_common.float16", fake_float16)
    monkeypatch.setattr(
        variants_mod,
        "_validate_fp16_variant_artifact",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "onnx_checker": {"status": "passed", "error": None},
            "onnxruntime": {"status": "failed", "error": "broken fp16 output signature"},
        },
    )

    row = variants_mod.run_fp16_variant(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        task="detection",
        dataset_export={"zip_relpath": str(zip_path.relative_to(tmp_path)).replace("\\", "/")},
        checkpoint_kind="best_metric",
    )

    assert row["status"] == "failed"
    assert row["error"] == "broken fp16 output signature"
    assert row["onnx"]["model_relpath"] is None

    model_path = storage.variant_onnx_model_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP16)
    metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP16)
    assert not model_path.exists()
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["status"] == "failed"
    assert metadata_payload["error"] == "broken fp16 output signature"
    assert metadata_payload["model_uri"] is None
    assert metadata_payload["validation"]["status"] == "failed"


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_qat_variant_uses_fake_quant_classification_flow_and_rebuilds_float_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pixel_sheriff_trainer.variants as variants_mod
    from pixel_sheriff_trainer.export_onnx import OnnxExportResult

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    _content_hash, zip_path = _write_tiny_export_zip(tmp_path, project_id)

    class TinyObserver(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.int64))
            self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.int64))
            self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float32))
            self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int32))
            inner = torch.nn.Module()
            inner.register_buffer("eps", torch.tensor([1e-5], dtype=torch.float32))
            inner.register_buffer("min_val", torch.tensor([0.0], dtype=torch.float32))
            inner.register_buffer("max_val", torch.tensor([0.0], dtype=torch.float32))
            self.add_module("activation_post_process", inner)

    class TinyQatConvBn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[[[1.0]]], [[[2.0]]]], dtype=torch.float32))
            self.bn = torch.nn.BatchNorm2d(2)
            self.add_module("weight_fake_quant", TinyObserver())

    class TinyFloatClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(2)
            self.fc = torch.nn.Linear(2, 1)

        def forward(self, x):  # type: ignore[override]
            return self.fc(torch.flatten(self.bn1(self.conv1(x)), 1))

    class TinyQatClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("activation_post_process_0", TinyObserver())
            self.add_module("conv1", TinyQatConvBn())
            self.fc = torch.nn.Linear(2, 1)

        def forward(self, x):  # type: ignore[override]
            return self.fc(torch.flatten(x, 1))

    seed_model = TinyFloatClassifier()
    with torch.no_grad():
        seed_model.conv1.weight.copy_(torch.tensor([[[[0.25]]], [[[0.75]]]], dtype=torch.float32))
        seed_model.bn1.weight.copy_(torch.tensor([1.2, 0.8], dtype=torch.float32))
        seed_model.bn1.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
        seed_model.fc.weight.copy_(torch.tensor([[0.5, -0.5]], dtype=torch.float32))
        seed_model.fc.bias.copy_(torch.tensor([0.05], dtype=torch.float32))
    seed_state = seed_model.state_dict()
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=2,
        metric_name=None,
        value=None,
        state_dict={"epoch": 2, "model_state_dict": seed_state},
    )

    class FakePipeline:
        def build_loaders(self, _job, _workdir, _storage):
            return SimpleNamespace(
                train="train-loader",
                val="val-loader",
                num_classes=1,
                class_names=["class-a"],
                class_order=["0"],
            )

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

    prepared_calls: list[tuple[str, int]] = []
    quantize_calls: list[tuple[str, str]] = []
    calibration_inputs: list[str] = []
    export_state_keys: list[str] = []
    export_bn_weight: torch.Tensor | None = None

    def _fake_quantize_static(model_input, model_output, calibration_reader, **_kwargs):
        quantize_calls.append((str(model_input), str(model_output)))
        assert calibration_reader.get_next() is not None
        calibration_reader.rewind()
        Path(model_output).write_bytes(b"fake-qat-model")

    def _fake_prepare_qat_training_model(*, task, model_config, num_classes):
        assert task == "classification"
        assert model_config["architecture"]["family"] == "resnet_classifier"
        prepared_calls.append((task, num_classes))
        return TinyQatClassifier()

    def _fake_run_training(**kwargs):
        prepared_model = kwargs["model"]
        assert isinstance(prepared_model, TinyQatClassifier)
        assert torch.equal(prepared_model.state_dict()["conv1.weight"], seed_state["conv1.weight"])
        assert torch.equal(prepared_model.state_dict()["conv1.bn.weight"], seed_state["bn1.weight"])
        epoch_row = SimpleNamespace(
            epoch=1,
            train_loss=0.22,
            train_accuracy=0.81,
            val_loss=0.18,
            val_accuracy=0.88,
            val_macro_f1=0.86,
            val_macro_precision=0.89,
            val_macro_recall=0.84,
            lr=0.001,
            epoch_seconds=1.2,
            eta_seconds=0.0,
            evaluated=True,
        )
        kwargs["on_epoch"](epoch_row)
        checkpoint_state = {
            "epoch": 1,
            "model_state_dict": prepared_model.state_dict(),
            "metrics": {"val_accuracy": 0.88},
        }
        kwargs["on_checkpoint"]("latest", 1, "val_accuracy", 0.88, checkpoint_state)
        kwargs["on_checkpoint"]("best_metric", 1, "val_accuracy", 0.88, checkpoint_state)
        return "completed", object()

    def _fake_export_model_to_onnx(
        _model,
        _storage,
        *,
        project_id,
        experiment_id,
        attempt,
        checkpoint_kind,
        checkpoint_uri,
        input_shape,
        input_names,
        output_names,
        preprocess,
        class_order,
        class_names,
        extra_metadata,
        output_dir,
    ):
        nonlocal export_bn_weight
        export_state = _model.state_dict()
        export_state_keys.extend(export_state.keys())
        export_bn_weight = export_state["bn1.weight"].detach().clone()
        assert "conv1.bn.weight" not in export_state
        assert not any("weight_fake_quant" in key or "activation_post_process" in key for key in export_state)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.onnx"
        metadata_path = output_dir / "onnx.metadata.json"
        model_path.write_bytes(b"fake-intermediate-model")
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "exported",
                    "task": extra_metadata["task"],
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_uri": checkpoint_uri,
                    "class_order": class_order,
                    "class_names": class_names,
                    "input_shape": [int(value) for value in input_shape],
                    "preprocess": preprocess,
                    "qat_export_flow": "float_export_then_ort_qdq",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return OnnxExportResult(
            status="exported",
            attempt=attempt,
            model_uri=str(model_path.relative_to(_storage.root)).replace("\\", "/"),
            metadata_uri=str(metadata_path.relative_to(_storage.root)).replace("\\", "/"),
            error=None,
            validation={"status": "passed"},
        )

    _install_fake_onnxruntime_quantization(monkeypatch, quantize_static=_fake_quantize_static)
    monkeypatch.setitem(variants_mod.PIPELINE_REGISTRY, "classification", FakePipeline())
    monkeypatch.setattr(variants_mod, "_prepare_qat_training_model", _fake_prepare_qat_training_model)
    monkeypatch.setattr(variants_mod, "run_training", _fake_run_training)
    monkeypatch.setattr(variants_mod, "_build_clean_variant_model", lambda **_kwargs: TinyFloatClassifier())
    monkeypatch.setattr(variants_mod, "export_model_to_onnx", _fake_export_model_to_onnx)
    monkeypatch.setattr(variants_mod, "_onnx_session", lambda _path, providers=None, cpu_only=False: FakeSession())
    monkeypatch.setattr(
        variants_mod,
        "preprocess_asset",
        lambda path, _metadata: calibration_inputs.append(Path(path).name) or np.zeros((1, 1, 8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_evaluations",
        lambda *args, **kwargs: {"val": {"status": "ready", "overall": {"val_accuracy": 0.88}}},
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_benchmark",
        lambda *args, **kwargs: {
            "status": "ready",
            "benchmark": {"status": "ready", "mean_latency_ms": 8.1},
            "devices": {"cpu": {"status": "ready", "mean_latency_ms": 8.1}},
        },
    )

    row = variants_mod.run_qat_variant(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        task="classification",
        model_config={
            "architecture": {
                "family": "resnet_classifier",
                "backbone": {"name": "resnet18", "pretrained": False},
                "head": {"num_classes": 1},
            },
            "input": {"input_size": [1, 8, 8], "normalization": {"type": "none"}},
        },
        training_config={
            "epochs": 5,
            "optimizer": {"lr": 0.01},
            "logging": {"keep_last": 2},
        },
        dataset_export={"zip_relpath": str(zip_path.relative_to(tmp_path)).replace("\\", "/")},
        checkpoint_kind="latest",
        epochs_override=None,
        learning_rate_override=None,
        calibration_max_samples=8,
    )

    assert row["status"] == "ready"
    assert row["quantized"] is True
    assert row["preferred"] is True
    assert prepared_calls == [("classification", 1)]
    assert row["qat"]["epochs"] == 3
    assert row["qat"]["learning_rate"] == pytest.approx(0.001)
    assert row["qat"]["strategy"] == "fake_quant_qat_then_ort_qdq"
    assert row["qat"]["mode"] == "fake_quant"
    assert row["qat"]["export_flow"] == "float_export_then_ort_qdq"
    assert row["qat"]["experimental"] is False
    assert row["qat"]["family"] == "resnet_classifier"
    assert row["quantization_strategy"] == "fake_quant_qat_then_ort_qdq"
    assert calibration_inputs == ["img0.png", "img1.png"]
    assert len(quantize_calls) == 1
    assert export_bn_weight is not None and torch.equal(export_bn_weight, seed_state["bn1.weight"])
    assert "bn1.weight" in export_state_keys

    metrics_path = storage.variant_metrics_path(project_id, experiment_id, attempt, variants_mod.VARIANT_QAT_INT8)
    metrics_rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert metrics_rows == [
        {
            "attempt": attempt,
            "epoch": 1,
            "epoch_seconds": 1.2,
            "eta_seconds": 0.0,
            "lr": 0.001,
            "train_accuracy": 0.81,
            "train_loss": 0.22,
            "val_accuracy": 0.88,
            "val_loss": 0.18,
            "val_macro_f1": 0.86,
            "val_macro_precision": 0.89,
            "val_macro_recall": 0.84,
        }
    ]

    metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_QAT_INT8)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["task"] == "classification"
    assert metadata_payload["variant_key"] == variants_mod.VARIANT_QAT_INT8
    assert metadata_payload["variant_kind"] == "qat"
    assert metadata_payload["quantization"]["mode"] == "fake_quant_qat_then_ort_qdq"
    assert metadata_payload["quantization"]["calibration_max_samples"] == 2
    assert metadata_payload["quantization"]["strategy"] == "fake_quant_qat_then_ort_qdq"
    assert metadata_payload["qat"]["mode"] == "fake_quant"
    assert metadata_payload["qat"]["export_flow"] == "float_export_then_ort_qdq"
    assert metadata_payload["qat"]["experimental"] is False
    assert metadata_payload["qat"]["family"] == "resnet_classifier"
    assert metadata_payload["qat"]["checkpoint_kind"] == "latest"
    assert metadata_payload["qat"]["best_epoch"] == 1
    assert metadata_payload["qat"]["best_metric"] == 0.88

    index_path = storage.variants_index_path(project_id, experiment_id, attempt)
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["preferred_variant_key"] == variants_mod.VARIANT_QAT_INT8


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_qat_variant_marks_ssdlite_detection_experimental(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.variants as variants_mod
    from pixel_sheriff_trainer.export_onnx import OnnxExportResult

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)

    class TinyObserver(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.int64))

    class TinyDetectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.add_module("activation_post_process_0", TinyObserver())

        def forward(self, x):  # type: ignore[override]
            return x

    seed_state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=2,
        metric_name=None,
        value=None,
        state_dict={"epoch": 2, "model_state_dict": seed_state},
    )

    class FakePipeline:
        def build_loaders(self, _job, _workdir, _storage):
            return SimpleNamespace(
                train="train-loader",
                val="val-loader",
                num_classes=1,
                class_names=["flower"],
                class_order=["1"],
            )

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

    def _fake_prepare_qat_training_model(*, task, model_config, num_classes):
        assert task == "detection"
        assert model_config["architecture"]["family"] == "ssdlite320_mobilenet_v3_large"
        return TinyDetectionModel()

    def _fake_run_detection_training(**kwargs):
        kwargs["on_epoch"](
            variants_mod.DetectionEpochMetrics(
                epoch=1,
                train_loss=0.42,
                mAP50=0.70,
                mAP50_95=0.55,
                precision=0.8,
                recall=0.75,
                matched_mean_iou=0.63,
                tp=8,
                fp=2,
                fn=1,
                duplicate_fp=0,
                lr=0.001,
                epoch_seconds=1.5,
                eta_seconds=0.0,
                evaluated=True,
            )
        )
        checkpoint_state = {
            "epoch": 1,
            "model_state_dict": kwargs["model"].state_dict(),
            "metrics": {"val_map_50_95": 0.55},
        }
        kwargs["on_checkpoint"]("latest", 1, "val_map_50_95", 0.55, checkpoint_state)
        kwargs["on_checkpoint"]("best_metric", 1, "val_map_50_95", 0.55, checkpoint_state)
        return "completed", object()

    def _fake_export_model_to_onnx(
        _model,
        _storage,
        *,
        project_id,
        experiment_id,
        attempt,
        checkpoint_kind,
        checkpoint_uri,
        input_shape,
        input_names,
        output_names,
        preprocess,
        class_order,
        class_names,
        extra_metadata,
        output_dir,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.onnx"
        metadata_path = output_dir / "onnx.metadata.json"
        model_path.write_bytes(b"fake-intermediate-model")
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "exported",
                    "task": extra_metadata["task"],
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_uri": checkpoint_uri,
                    "class_order": class_order,
                    "class_names": class_names,
                    "input_shape": [int(value) for value in input_shape],
                    "preprocess": preprocess,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return OnnxExportResult(
            status="exported",
            attempt=attempt,
            model_uri=str(model_path.relative_to(_storage.root)).replace("\\", "/"),
            metadata_uri=str(metadata_path.relative_to(_storage.root)).replace("\\", "/"),
            error=None,
            validation={"status": "passed"},
        )

    _install_fake_onnxruntime_quantization(monkeypatch, quantize_static=lambda *_args, **_kwargs: Path(_args[1]).write_bytes(b"fake-qat-model"))
    monkeypatch.setitem(variants_mod.PIPELINE_REGISTRY, "detection", FakePipeline())
    monkeypatch.setattr(variants_mod, "_prepare_qat_training_model", _fake_prepare_qat_training_model)
    monkeypatch.setattr(variants_mod, "_build_clean_variant_model", lambda **_kwargs: TinyDetectionModel())
    monkeypatch.setattr(variants_mod, "run_detection_training", _fake_run_detection_training)
    monkeypatch.setattr(variants_mod, "export_model_to_onnx", _fake_export_model_to_onnx)
    monkeypatch.setattr(variants_mod, "_onnx_session", lambda _path, providers=None, cpu_only=False: FakeSession())
    monkeypatch.setattr(
        variants_mod,
        "preprocess_asset",
        lambda path, _metadata: np.zeros((1, 3, 32, 32), dtype=np.float32),
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_evaluations",
        lambda *args, **kwargs: {"val": {"status": "ready", "overall": {"mAP50_95": 0.55}}},
    )
    monkeypatch.setattr(
        variants_mod,
        "_write_variant_benchmark",
        lambda *args, **kwargs: {"status": "ready", "benchmark": {"status": "ready", "mean_latency_ms": 9.4}, "devices": {"cpu": {"status": "ready", "mean_latency_ms": 9.4}}},
    )

    row = variants_mod.run_qat_variant(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        task="detection",
        model_config={
            "architecture": {
                "family": "ssdlite320_mobilenet_v3_large",
                "backbone": {"name": "mobilenet_v3_large", "pretrained": False},
            },
            "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
        },
        training_config={
            "epochs": 5,
            "optimizer": {"lr": 0.01},
            "logging": {"keep_last": 2},
        },
        dataset_export={"zip_relpath": str(zip_path.relative_to(tmp_path)).replace("\\", "/")},
        checkpoint_kind="latest",
        epochs_override=None,
        learning_rate_override=None,
        calibration_max_samples=8,
    )

    assert row["status"] == "ready"
    assert row["quantization_strategy"] == "fake_quant_qat_then_ort_qdq"
    assert row["qat"]["mode"] == "fake_quant"
    assert row["qat"]["export_flow"] == "float_export_then_ort_qdq"
    assert row["qat"]["experimental"] is True
    assert row["qat"]["family"] == "ssdlite320_mobilenet_v3_large"
    assert "SSDLite" in row["qat"]["warning"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_qat_variant_rejects_retinanet_real_qat(tmp_path: Path) -> None:
    import pixel_sheriff_trainer.variants as variants_mod

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1

    row = variants_mod.run_qat_variant(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        task="detection",
        model_config={"architecture": {"family": "retinanet", "backbone": {"name": "resnet50"}}},
        training_config={"epochs": 5, "optimizer": {"lr": 0.01}},
        dataset_export={"zip_relpath": "exports/example.zip"},
        checkpoint_kind="latest",
        epochs_override=None,
        learning_rate_override=None,
        calibration_max_samples=8,
    )

    assert row["status"] == "unsupported"
    assert row["error"] == "Real fake-quant QAT v1 is not supported for detection family 'retinanet'"
