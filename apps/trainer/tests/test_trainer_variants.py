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
def test_variant_task_support_reports_detection_qat_supported() -> None:
    import pixel_sheriff_trainer.variants as variants_mod

    detection_support = variants_mod.variant_task_support("detection")
    assert detection_support == {
        "fp16_supported": True,
        "fp16_reason": None,
        "ptq_supported": True,
        "qat_supported": True,
        "qat_reason": None,
    }

    segmentation_support = variants_mod.variant_task_support("segmentation")
    assert segmentation_support == {
        "fp16_supported": False,
        "fp16_reason": "FP16 is not supported for this task",
        "ptq_supported": False,
        "qat_supported": False,
        "qat_reason": "QAT is not supported for this task",
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
    fake_onnx = ModuleType("onnx")
    fake_onnx.load_model = lambda path: {"path": path}
    fake_onnx.save_model = lambda model, path: saved_models.append((model, path)) or Path(path).write_bytes(b"fake-fp16-model")
    fake_float16 = ModuleType("onnxconverter_common.float16")
    fake_float16.convert_float_to_float16 = lambda model, keep_io_types=True: {"converted": model, "keep_io_types": keep_io_types}
    fake_package = ModuleType("onnxconverter_common")
    fake_package.float16 = fake_float16
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxconverter_common", fake_package)
    monkeypatch.setitem(sys.modules, "onnxconverter_common.float16", fake_float16)
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

    metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_FP16)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["variant_key"] == variants_mod.VARIANT_FP16
    assert metadata_payload["variant_kind"] == "fp16"
    assert metadata_payload["numeric_precision"] == "fp16"
    assert metadata_payload["quantized"] is False


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_run_qat_variant_supports_detection_and_records_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pixel_sheriff_trainer.variants as variants_mod
    from pixel_sheriff_trainer.export_onnx import OnnxExportResult

    storage = ExperimentStorage(str(tmp_path))
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    attempt = 1
    zip_path = _write_tiny_coco_export_zip(tmp_path, project_id, include_segmentation=False)

    class TinyDetectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, x):  # type: ignore[override]
            return x

    seed_state = TinyDetectionModel().state_dict()
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

    quantize_calls: list[tuple[str, str]] = []
    calibration_inputs: list[str] = []

    def _fake_quantize_static(model_input, model_output, calibration_reader, **_kwargs):
        quantize_calls.append((str(model_input), str(model_output)))
        assert calibration_reader.get_next() is not None
        calibration_reader.rewind()
        Path(model_output).write_bytes(b"fake-qat-model")

    def _fake_run_detection_training(**kwargs):
        resume_state = kwargs["resume_state"]["model_state_dict"]
        assert set(resume_state.keys()) == set(seed_state.keys())
        for key, value in seed_state.items():
            assert torch.equal(resume_state[key], value)
        epoch_row = variants_mod.DetectionEpochMetrics(
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
        kwargs["on_epoch"](epoch_row)
        checkpoint_state = {
            "epoch": 1,
            "model_state_dict": TinyDetectionModel().state_dict(),
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

    _install_fake_onnxruntime_quantization(monkeypatch, quantize_static=_fake_quantize_static)
    monkeypatch.setitem(variants_mod.PIPELINE_REGISTRY, "detection", FakePipeline())
    monkeypatch.setattr(variants_mod, "run_detection_training", _fake_run_detection_training)
    monkeypatch.setattr(variants_mod, "_build_detection_model", lambda _config, *, num_classes: TinyDetectionModel())
    monkeypatch.setattr(variants_mod, "export_model_to_onnx", _fake_export_model_to_onnx)
    monkeypatch.setattr(variants_mod, "_onnx_session", lambda _path, providers=None, cpu_only=False: FakeSession())
    monkeypatch.setattr(
        variants_mod,
        "preprocess_asset",
        lambda path, _metadata: calibration_inputs.append(Path(path).name) or np.zeros((1, 3, 32, 32), dtype=np.float32),
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
                "family": "retinanet",
                "backbone": {"name": "resnet50", "pretrained": False},
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
    assert row["quantized"] is True
    assert row["preferred"] is True
    assert row["qat"]["epochs"] == 3
    assert row["qat"]["learning_rate"] == pytest.approx(0.001)
    assert row["qat"]["strategy"] == "finetune_then_ptq"
    assert calibration_inputs == ["coco_0.png"]
    assert len(quantize_calls) == 1

    metrics_path = storage.variant_metrics_path(project_id, experiment_id, attempt, variants_mod.VARIANT_QAT_INT8)
    metrics_rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert metrics_rows == [
        {
            "attempt": attempt,
            "epoch": 1,
            "epoch_seconds": 1.5,
            "eta_seconds": 0.0,
            "evaluated": True,
            "lr": 0.001,
            "train_loss": 0.42,
            "val_duplicate_fp": 0,
            "val_fn": 1,
            "val_fp": 2,
            "val_map": 0.7,
            "val_map_50_95": 0.55,
            "val_matched_mean_iou": 0.63,
            "val_precision": 0.8,
            "val_recall": 0.75,
            "val_tp": 8,
        }
    ]

    metadata_path = storage.variant_onnx_metadata_path(project_id, experiment_id, attempt, variants_mod.VARIANT_QAT_INT8)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["task"] == "detection"
    assert metadata_payload["variant_key"] == variants_mod.VARIANT_QAT_INT8
    assert metadata_payload["variant_kind"] == "qat"
    assert metadata_payload["quantization"]["mode"] == "static_int8_after_finetune"
    assert metadata_payload["quantization"]["calibration_max_samples"] == 1
    assert metadata_payload["qat"]["checkpoint_kind"] == "latest"
    assert metadata_payload["qat"]["best_epoch"] == 1
    assert metadata_payload["qat"]["best_metric"] == 0.55

    index_path = storage.variants_index_path(project_id, experiment_id, attempt)
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["preferred_variant_key"] == variants_mod.VARIANT_QAT_INT8
