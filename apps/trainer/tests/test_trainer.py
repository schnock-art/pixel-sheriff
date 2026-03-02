from __future__ import annotations

import json
from pathlib import Path
import uuid
import zipfile

import pytest

try:
    import torch  # noqa: F401
    from pixel_sheriff_trainer.classification.dataset import build_classification_loaders
    from pixel_sheriff_trainer.jobs import TrainJob
    from pixel_sheriff_trainer.runner import TrainRunner

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    build_classification_loaders = None  # type: ignore[assignment]
    TrainJob = None  # type: ignore[assignment]
    TrainRunner = None  # type: ignore[assignment]

def _write_tiny_export_zip(root: Path, project_id: str) -> tuple[str, Path]:
    from PIL import Image

    assets_dir = root / "tmp_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_path = assets_dir / "img0.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)

    manifest = {
        "schema_version": "1.2",
        "label_schema": {
            "classes": [{"id": 1, "name": "cat"}],
            "class_order": [1],
        },
        "splits": {
            "train": {"asset_ids": ["asset-0"]},
            "val": {"asset_ids": ["asset-0"]},
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
            }
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
            }
        ],
    }
    content_hash = "tinyhash123"
    zip_relpath = Path("exports") / project_id / f"{content_hash}.zip"
    zip_path = root / zip_relpath
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        bundle.write(image_path, arcname="assets/img0.png")
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
        training_config={"batch_size": 1, "advanced": {"num_workers": 0, "seed": 1}},
    )
    assert loaded.num_classes == 1
    assert loaded.train_count >= 1
    assert loaded.val_count >= 1


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
            "batch_size": 1,
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
    if result == "completed":
        run_eval = run_dir / "evaluation.json"
        run_predictions = run_dir / "predictions.jsonl"
        run_predictions_meta = run_dir / "predictions.meta.json"
        latest_eval = tmp_path / "experiments" / project_id / experiment_id / "evaluation.json"
        latest_predictions = tmp_path / "experiments" / project_id / experiment_id / "predictions.jsonl"
        latest_predictions_meta = tmp_path / "experiments" / project_id / experiment_id / "predictions.meta.json"
        assert run_eval.exists()
        assert run_predictions.exists()
        assert run_predictions_meta.exists()
        assert latest_eval.exists()
        assert latest_predictions.exists()
        assert latest_predictions_meta.exists()

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
