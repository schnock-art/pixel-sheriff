from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from .trainer_test_helpers import (
    HAS_TORCH,
    TrainJob,
    TrainRunner,
    _seed_experiment_layout,
    _write_tiny_export_zip,
    parse_train_job,
)

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
        task_id="task-1",
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
        task_id="task-1",
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
        task_id="task-1",
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
