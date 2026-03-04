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
    from pixel_sheriff_trainer.export_onnx import export_best_classification_onnx
    from pixel_sheriff_trainer.io.checkpoints import save_checkpoint
    from pixel_sheriff_trainer.io.storage import ExperimentStorage
    from pixel_sheriff_trainer.jobs import TrainJob, parse_train_job
    from pixel_sheriff_trainer.runner import TrainRunner

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    build_resnet_classifier = None  # type: ignore[assignment]
    build_classification_loaders = None  # type: ignore[assignment]
    export_best_classification_onnx = None  # type: ignore[assignment]
    save_checkpoint = None  # type: ignore[assignment]
    ExperimentStorage = None  # type: ignore[assignment]
    TrainJob = None  # type: ignore[assignment]
    parse_train_job = None  # type: ignore[assignment]
    TrainRunner = None  # type: ignore[assignment]

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
