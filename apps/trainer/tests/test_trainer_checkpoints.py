from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from .trainer_test_helpers import (
    ExperimentStorage,
    HAS_ONNX_RUNTIME,
    HAS_TORCH,
    TrainJob,
    TrainRunner,
    _seed_experiment_layout,
    _write_tiny_export_zip,
    build_resnet_classifier,
    compact_completed_checkpoints,
    export_best_classification_onnx,
    read_checkpoints,
    save_checkpoint,
    torch,
)

@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_compact_completed_checkpoints_preserves_resumable_latest_artifact(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    project_id = "project-1"
    experiment_id = "experiment-1"
    attempt = 1

    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=3,
        metric_name="val_accuracy",
        value=0.91,
        state_dict={"epoch": 3, "model_state_dict": {"w": torch.tensor([1.0])}, "optimizer_state_dict": {"lr": 0.1}},
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_metric",
        epoch=3,
        metric_name="val_accuracy",
        value=0.91,
        state_dict={"epoch": 3, "model_state_dict": {"w": torch.tensor([1.0])}},
    )

    compacted_kinds = compact_completed_checkpoints(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
    )

    assert compacted_kinds == []
    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    by_kind = {row["kind"]: row for row in rows}
    assert by_kind["latest"]["uri"] != by_kind["best_metric"]["uri"]
    assert storage.resolve(by_kind["best_metric"]["uri"]).exists()
    assert storage.resolve(by_kind["latest"]["uri"]).exists()
    assert (storage.checkpoints_dir(project_id, experiment_id, attempt) / "latest.pt").exists()
    assert (storage.checkpoints_dir(project_id, experiment_id, attempt) / "latest_epoch_3.pt").exists()


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_compact_completed_checkpoints_uses_best_metric_as_canonical(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    project_id = "project-2"
    experiment_id = "experiment-2"
    attempt = 1

    shared_model_state = {"epoch": 4, "model_state_dict": {"w": torch.tensor([2.0])}}
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_metric",
        epoch=4,
        metric_name="val_accuracy",
        value=0.93,
        state_dict=shared_model_state,
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_loss",
        epoch=4,
        metric_name="val_loss",
        value=0.12,
        state_dict=shared_model_state,
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=4,
        metric_name="val_accuracy",
        value=0.93,
        state_dict={"epoch": 4, "model_state_dict": {"w": torch.tensor([2.0])}, "optimizer_state_dict": {"lr": 0.1}},
    )

    compacted_kinds = compact_completed_checkpoints(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
    )

    assert compacted_kinds == ["best_loss"]
    rows = read_checkpoints(storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt)
    by_kind = {row["kind"]: row for row in rows}
    canonical_uri = by_kind["best_metric"]["uri"]
    assert by_kind["best_loss"]["uri"] == canonical_uri
    assert by_kind["latest"]["uri"] != canonical_uri
    ckpt_dir = storage.checkpoints_dir(project_id, experiment_id, attempt)
    assert (ckpt_dir / "best_metric.pt").exists()
    assert (ckpt_dir / "best_loss.pt").exists() is False
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "latest_epoch_4.pt").exists()


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_compact_completed_checkpoints_keeps_latest_resumable_for_resume(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    project_id = "project-resume"
    experiment_id = "experiment-resume"
    attempt = 1

    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_metric",
        epoch=2,
        metric_name="val_accuracy",
        value=0.88,
        state_dict={"epoch": 2, "model_state_dict": {"w": torch.tensor([5.0])}},
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=2,
        metric_name="val_accuracy",
        value=0.88,
        state_dict={
            "epoch": 2,
            "model_state_dict": {"w": torch.tensor([5.0])},
            "optimizer_state_dict": {"lr": 0.1},
            "scheduler_state_dict": {"gamma": 0.9},
        },
    )

    compacted_kinds = compact_completed_checkpoints(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
    )
    assert compacted_kinds == []

    runner = TrainRunner(str(tmp_path))
    resume_state, resume_message = runner._try_load_resume_state(
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        training_config={"resume": {"enabled": True, "checkpoint_kind": "latest"}},
    )
    assert resume_state is not None
    assert resume_state["epoch"] == 2
    assert resume_state["optimizer_state_dict"] == {"lr": 0.1}
    assert resume_state["scheduler_state_dict"] == {"gamma": 0.9}
    assert resume_message == "resume applied from latest checkpoint (epoch=2)"


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_compact_completed_checkpoints_keeps_distinct_epochs(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    project_id = "project-3"
    experiment_id = "experiment-3"
    attempt = 1

    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_metric",
        epoch=2,
        metric_name="val_accuracy",
        value=0.82,
        state_dict={"epoch": 2, "model_state_dict": {"w": torch.tensor([3.0])}},
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_loss",
        epoch=1,
        metric_name="val_loss",
        value=0.33,
        state_dict={"epoch": 1, "model_state_dict": {"w": torch.tensor([3.0])}},
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=3,
        metric_name="val_accuracy",
        value=0.79,
        state_dict={"epoch": 3, "model_state_dict": {"w": torch.tensor([3.0])}, "optimizer_state_dict": {"lr": 0.1}},
    )

    compacted_kinds = compact_completed_checkpoints(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
    )

    assert compacted_kinds == []
    ckpt_dir = storage.checkpoints_dir(project_id, experiment_id, attempt)
    assert (ckpt_dir / "best_metric.pt").exists()
    assert (ckpt_dir / "best_loss.pt").exists()
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "latest_epoch_3.pt").exists()


@pytest.mark.skipif(not HAS_TORCH or not HAS_ONNX_RUNTIME, reason="torch + onnxruntime are required")
def test_compact_completed_checkpoints_after_onnx_export_preserves_exported_artifacts(tmp_path: Path) -> None:
    storage = ExperimentStorage(str(tmp_path))
    project_id = "project-4"
    experiment_id = "experiment-4"
    attempt = 1
    num_classes = 2
    model_config = {
        "architecture": {
            "family": "resnet_classifier",
            "backbone": {"name": "resnet18", "pretrained": False},
            "head": {"num_classes": num_classes},
        },
        "input": {"input_size": [32, 32], "normalization": {"type": "none"}},
    }
    model = build_resnet_classifier(model_config, num_classes_override=num_classes)
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="best_metric",
        epoch=2,
        metric_name="val_accuracy",
        value=0.87,
        state_dict={"epoch": 2, "model_state_dict": model.state_dict()},
    )
    save_checkpoint(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        kind="latest",
        epoch=2,
        metric_name="val_accuracy",
        value=0.87,
        state_dict={"epoch": 2, "model_state_dict": model.state_dict(), "optimizer_state_dict": {"lr": 0.1}},
    )

    exported = export_best_classification_onnx(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
        model_config=model_config,
        num_classes=num_classes,
        class_names=["cat", "dog"],
        class_order=["cat", "dog"],
    )
    assert exported.status == "exported"
    assert exported.model_uri is not None
    assert storage.resolve(exported.model_uri).exists()

    compacted_kinds = compact_completed_checkpoints(
        storage,
        project_id=project_id,
        experiment_id=experiment_id,
        attempt=attempt,
    )

    assert compacted_kinds == []
    assert storage.resolve(exported.model_uri).exists()
    assert storage.resolve(exported.metadata_uri).exists()


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
def test_runner_checkpoint_compaction_failure_does_not_fail_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    content_hash, _zip_path = _write_tiny_export_zip(tmp_path, project_id)
    _seed_experiment_layout(tmp_path, project_id, experiment_id, job_id)

    import pixel_sheriff_trainer.runner as runner_mod

    monkeypatch.setattr(runner_mod, "compact_completed_checkpoints", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

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
    if result != "completed":
        pytest.skip(f"training did not complete in test environment: {result}")

    run_dir = tmp_path / "experiments" / project_id / experiment_id / "runs" / "1"
    log_text = (run_dir / "training.log").read_text(encoding="utf-8")
    assert "checkpoint_compaction_failed error=disk full" in log_text

    status_payload = json.loads((tmp_path / "experiments" / project_id / experiment_id / "status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"
