from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from sheriff_api.config import get_settings
from sheriff_api.db.models import Project
from sheriff_api.db.session import SessionLocal
from sheriff_api.errors import api_error
from sheriff_api.routers.experiments.shared import (
    default_training_config,
    ensure_dataset_export_zip,
    experiment_store,
    model_store,
    normalize_task,
    shared_architecture_family,
    utc_now_iso,
)
from sheriff_api.services.dataset_store import DatasetStore
from sheriff_api.services.storage import LocalStorage


DEMO_ATTEMPT = 1
DEMO_JOB_ID = "demo-experiment-job-1"
DEMO_METRIC_NAME = "val_map_50_95"
DEMO_INPUT_SHAPE = [3, 512, 512]
DEMO_RESIZE = {"width": 512, "height": 512}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            rows.append(item.strip())
    return rows


def _dataset_classes(dataset_version: dict[str, Any]) -> tuple[list[str], list[str]]:
    label_schema = dataset_version.get("labels", {}).get("label_schema", {})
    if not isinstance(label_schema, dict):
        raise ValueError("dataset_label_schema_missing")

    class_ids = _as_str_list(label_schema.get("class_order"))
    if not class_ids:
        raise ValueError("dataset_class_order_missing")

    class_name_by_id: dict[str, str] = {}
    classes = label_schema.get("classes")
    if isinstance(classes, list):
        for row in classes:
            if not isinstance(row, dict):
                continue
            category_id = row.get("category_id")
            if not isinstance(category_id, str) or not category_id.strip():
                continue
            export_name = row.get("export_name")
            name = export_name if isinstance(export_name, str) and export_name.strip() else row.get("name")
            if isinstance(name, str) and name.strip():
                class_name_by_id[category_id] = name.strip()

    class_names = [class_name_by_id.get(class_id, class_id) for class_id in class_ids]
    return class_ids, class_names


def _model_name(model_record: dict[str, Any], model_id: str) -> str:
    name = model_record.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return model_id


def _experiment_name(model_record: dict[str, Any], model_id: str) -> str:
    return f"{_model_name(model_record, model_id)} training_run_1"


def _experiment_dir(storage_root: str, project_id: str, experiment_id: str) -> Path:
    return Path(storage_root) / "experiments" / project_id / experiment_id


def _run_dir(storage_root: str, project_id: str, experiment_id: str, attempt: int) -> Path:
    return _experiment_dir(storage_root, project_id, experiment_id) / "runs" / str(attempt)


def _storage_relpath(storage_root: str, path: Path) -> str:
    return str(path.relative_to(Path(storage_root).resolve())).replace("\\", "/")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metrics_rows() -> list[dict[str, Any]]:
    epochs = [
        (1, 1.26, 1.08, 0.46, 0.28, 0.61, 0.53, 0.57, 7, 5, 6, 1, 8.8, 44.0),
        (2, 0.98, 0.84, 0.59, 0.39, 0.70, 0.64, 0.63, 9, 4, 5, 1, 8.2, 32.8),
        (3, 0.78, 0.66, 0.71, 0.52, 0.79, 0.73, 0.70, 11, 4, 4, 1, 7.7, 23.1),
        (4, 0.62, 0.54, 0.82, 0.63, 0.86, 0.80, 0.76, 14, 3, 3, 1, 7.1, 14.2),
        (5, 0.51, 0.43, 0.88, 0.69, 0.90, 0.85, 0.80, 16, 2, 2, 0, 6.6, 6.6),
        (6, 0.44, 0.37, 0.92, 0.74, 0.94, 0.89, 0.84, 18, 2, 1, 0, 6.1, 0.0),
    ]
    rows: list[dict[str, Any]] = []
    for (
        epoch,
        train_loss,
        val_loss,
        val_map,
        val_map_50_95,
        val_precision,
        val_recall,
        val_matched_mean_iou,
        val_tp,
        val_fp,
        val_fn,
        val_duplicate_fp,
        epoch_seconds,
        eta_seconds,
    ) in epochs:
        rows.append(
            {
                "attempt": DEMO_ATTEMPT,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_map": val_map,
                "val_map_50_95": val_map_50_95,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_matched_mean_iou": val_matched_mean_iou,
                "val_tp": val_tp,
                "val_fp": val_fp,
                "val_fn": val_fn,
                "val_duplicate_fp": val_duplicate_fp,
                "epoch_seconds": epoch_seconds,
                "eta_seconds": eta_seconds,
                "created_at": utc_now_iso(),
            }
        )
    return rows


def _checkpoint_rows(storage_root: str, project_id: str, experiment_id: str) -> list[dict[str, Any]]:
    checkpoints_dir = _run_dir(storage_root, project_id, experiment_id, DEMO_ATTEMPT) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_specs = [
        ("best_metric", 6, DEMO_METRIC_NAME, 0.74),
        ("best_loss", 6, "val_loss", 0.37),
        ("latest", 6, DEMO_METRIC_NAME, 0.74),
    ]
    rows: list[dict[str, Any]] = []
    for kind, epoch, metric_name, value in checkpoint_specs:
        checkpoint_path = checkpoints_dir / f"{kind}.pt"
        checkpoint_path.write_bytes(f"demo checkpoint {kind}\n".encode("utf-8"))
        rows.append(
            {
                "kind": kind,
                "epoch": epoch,
                "metric_name": metric_name,
                "value": value,
                "updated_at": utc_now_iso(),
                "uri": _storage_relpath(storage_root, checkpoint_path),
                "status": "ok",
                "error": None,
            }
        )
    return rows


def _build_detection_evaluation(
    *,
    project_id: str,
    experiment_id: str,
    model_id: str,
    task_id: str,
    dataset_version_id: str,
    dataset_export: dict[str, Any],
    class_ids: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []
    for index, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
        precision = max(0.72, 0.93 - (index * 0.05))
        recall = max(0.69, 0.89 - (index * 0.04))
        ap50 = max(0.76, 0.94 - (index * 0.04))
        ap75 = max(0.64, ap50 - 0.12)
        map_50_95 = max(0.58, ap50 - 0.18)
        matched_mean_iou = max(0.62, 0.87 - (index * 0.05))
        support = max(4, 11 - (index * 2))
        tp = max(3, 9 - index)
        fp = min(3, 1 + index)
        fn = max(1, 3 - min(index, 2))
        duplicate_fp = 1 if index == 0 and len(class_ids) > 1 else 0
        per_class.append(
            {
                "class_index": index,
                "class_id": class_id,
                "name": class_name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round((2 * precision * recall) / max(precision + recall, 1e-9), 4),
                "support": support,
                "ap50": round(ap50, 4),
                "ap75": round(ap75, 4),
                "map_50_95": round(map_50_95, 4),
                "ap_by_iou": {
                    "0.50": round(ap50, 4),
                    "0.75": round(ap75, 4),
                },
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "duplicate_fp": duplicate_fp,
                "matched_mean_iou": round(matched_mean_iou, 4),
            }
        )

    overall_precision = mean(row["precision"] for row in per_class)
    overall_recall = mean(row["recall"] for row in per_class)
    overall_map50 = mean(row["ap50"] for row in per_class)
    overall_map50_95 = mean(row["map_50_95"] for row in per_class)
    overall_iou = mean(row["matched_mean_iou"] for row in per_class)

    return {
        "schema_version": "1",
        "task": "detection",
        "computed_at": utc_now_iso(),
        "split": "val",
        "num_samples": 6,
        "provenance": {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "attempt": DEMO_ATTEMPT,
            "model_id": model_id,
            "task_id": task_id,
            "job_id": DEMO_JOB_ID,
            "dataset_version_id": dataset_version_id,
            "dataset_export_hash": str(dataset_export.get("content_hash") or ""),
            "dataset_export_relpath": str(dataset_export.get("zip_relpath") or ""),
        },
        "classes": {
            "class_order": class_ids,
            "class_names": class_names,
            "id_to_index": {class_id: index for index, class_id in enumerate(class_ids)},
        },
        "overall": {
            "mAP50": round(overall_map50, 4),
            "mAP50_95": round(overall_map50_95, 4),
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "tp": sum(int(row["tp"]) for row in per_class),
            "fp": sum(int(row["fp"]) for row in per_class),
            "fn": sum(int(row["fn"]) for row in per_class),
            "duplicate_fp": sum(int(row["duplicate_fp"]) for row in per_class),
            "matched_mean_iou": round(overall_iou, 4),
            "ap_small": 0.58,
            "ap_medium": 0.73,
            "ap_large": 0.88,
            "size_buckets": {
                "small": {
                    "ground_truth_count": 2,
                    "prediction_count": 2,
                    "ap50": 0.59,
                    "map_50_95": 0.44,
                    "precision": 0.67,
                    "recall": 0.62,
                },
                "medium": {
                    "ground_truth_count": 2,
                    "prediction_count": 2,
                    "ap50": 0.78,
                    "map_50_95": 0.66,
                    "precision": 0.83,
                    "recall": 0.79,
                },
                "large": {
                    "ground_truth_count": 2,
                    "prediction_count": 2,
                    "ap50": 0.91,
                    "map_50_95": 0.79,
                    "precision": 0.93,
                    "recall": 0.9,
                },
            },
        },
        "per_class": per_class,
    }


def _runtime_payload() -> dict[str, Any]:
    return {
        "device_selected": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "amp_enabled": False,
        "torch_version": "2.7.1",
        "torchvision_version": "0.22.1",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "cache_resized_images": True,
        "max_cached_images": 256,
    }


def _training_log() -> str:
    return "\n".join(
        [
            "run_started status=running device=cpu task=detection",
            "dataset_export status=ready split=val hash=demo-export-v1",
            "epoch=1 train_loss=1.2600 val_loss=1.0800 val_map=0.4600 val_map_50_95=0.2800",
            "epoch=2 train_loss=0.9800 val_loss=0.8400 val_map=0.5900 val_map_50_95=0.3900",
            "epoch=3 train_loss=0.7800 val_loss=0.6600 val_map=0.7100 val_map_50_95=0.5200",
            "epoch=4 train_loss=0.6200 val_loss=0.5400 val_map=0.8200 val_map_50_95=0.6300",
            "epoch=5 train_loss=0.5100 val_loss=0.4300 val_map=0.8800 val_map_50_95=0.6900",
            "epoch=6 train_loss=0.4400 val_loss=0.3700 val_map=0.9200 val_map_50_95=0.7400",
            "checkpoint kind=best_metric epoch=6 metric=val_map_50_95 value=0.7400",
            "checkpoint kind=best_loss epoch=6 metric=val_loss value=0.3700",
            "onnx_export status=exported model_uri=experiments/demo/model.onnx metadata_uri=experiments/demo/onnx.metadata.json",
            "run_finished status=completed",
            "",
        ]
    )


def _onnx_metadata(class_ids: list[str], class_names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "exported",
        "attempt": DEMO_ATTEMPT,
        "task": "detection",
        "provider": "onnxruntime",
        "input_shape": DEMO_INPUT_SHAPE,
        "class_ids": class_ids,
        "class_order": class_ids,
        "class_names": class_names,
        "preprocess": {
            "resize": DEMO_RESIZE,
            "normalization": {"type": "imagenet"},
            "channels": "rgb",
        },
        "validation": {"status": "passed"},
        "error": None,
    }


async def seed_demo_experiment(
    project_id: str,
    task_id: str,
    model_id: str,
    dataset_version_id: str,
) -> dict[str, Any]:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    dataset_store = DatasetStore(settings.storage_root)

    async with SessionLocal() as db:
        project = await db.get(Project, project_id)
        if project is None:
            raise api_error(status_code=404, code="project_not_found", message="Project not found")

        loaded_dataset = dataset_store.get_version(project_id, dataset_version_id)
        if loaded_dataset is None or not isinstance(loaded_dataset.get("version"), dict):
            raise api_error(status_code=404, code="dataset_version_not_found", message="Dataset version not found in project")
        dataset_version = loaded_dataset["version"]

        model_record = model_store.get(project_id, model_id)
        if model_record is None:
            raise api_error(status_code=404, code="model_not_found", message="Model not found in project")

        model_config = model_record.get("config_json")
        if not isinstance(model_config, dict):
            raise ValueError("model_config_invalid")

        task = normalize_task(str(dataset_version.get("task") or "classification"))
        if task != "detection":
            raise ValueError(f"unsupported_task:{task}")

        class_ids, class_names = _dataset_classes(dataset_version)
        config_json = default_training_config(
            model_id=model_id,
            dataset_version_id=dataset_version_id,
            task_id=task_id,
            task=task,
        )
        config_json.update(
            {
                "epochs": 6,
                "batch_size": 2,
                "precision": "fp32",
                "optimizer": {"type": "adamw", "lr": 0.00025, "weight_decay": 0.0001},
                "scheduler": {"type": "cosine", "params": {}},
                "advanced": {"seed": 2026, "num_workers": 0, "grad_clip_norm": 1.0},
                "evaluation": {"eval_interval_epochs": 1},
                "logging": {"save_every_epochs": 1, "keep_last": 2, "keep_best": True},
                "runtime": {
                    "device": "auto",
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "prefetch_factor": 2,
                    "cache_resized_images": True,
                    "max_cached_images": 256,
                },
            }
        )

        created = experiment_store.create(
            project_id=project_id,
            model_id=model_id,
            task_id=task_id,
            name=_experiment_name(model_record, model_id),
            config_json=config_json,
            status="draft",
        )
        experiment_id = str(created.get("id") or "")
        if not experiment_id:
            raise ValueError("experiment_create_failed")

        dataset_export = await ensure_dataset_export_zip(
            db=db,
            project=project,
            dataset_version=dataset_version,
        )
        initialized = experiment_store.init_run_attempt(
            project_id=project_id,
            experiment_id=experiment_id,
            job_id=DEMO_JOB_ID,
            dataset_export=dataset_export,
            task=task,
            model_family=shared_architecture_family(model_config),
        )
        if initialized is None:
            raise ValueError("experiment_attempt_init_failed")

        run_dir = _run_dir(settings.storage_root, project_id, experiment_id, DEMO_ATTEMPT)
        experiment_store.set_run_started_at(project_id=project_id, experiment_id=experiment_id, attempt=DEMO_ATTEMPT)
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=DEMO_ATTEMPT,
            event={"type": "status", "status": "running", "attempt": DEMO_ATTEMPT, "job_id": DEMO_JOB_ID, "ts": utc_now_iso()},
        )

        metrics_rows = _metrics_rows()
        for row in metrics_rows:
            experiment_store.append_metric(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=DEMO_ATTEMPT,
                metric_row=row,
            )
            experiment_store.append_event(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=DEMO_ATTEMPT,
                event={"type": "metric", **row},
            )

        checkpoints = _checkpoint_rows(settings.storage_root, project_id, experiment_id)
        experiment_store.set_checkpoints(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=DEMO_ATTEMPT,
            checkpoints=checkpoints,
        )
        for checkpoint in checkpoints:
            experiment_store.append_event(
                project_id=project_id,
                experiment_id=experiment_id,
                attempt=DEMO_ATTEMPT,
                event={"type": "checkpoint", **checkpoint},
            )

        experiment_store.set_summary(
            project_id=project_id,
            experiment_id=experiment_id,
            summary_json={
                "best_metric_name": DEMO_METRIC_NAME,
                "best_metric_value": checkpoints[0]["value"],
                "best_epoch": checkpoints[0]["epoch"],
                "last_epoch": metrics_rows[-1]["epoch"],
            },
        )

        evaluation_payload = _build_detection_evaluation(
            project_id=project_id,
            experiment_id=experiment_id,
            model_id=model_id,
            task_id=task_id,
            dataset_version_id=dataset_version_id,
            dataset_export=dataset_export,
            class_ids=class_ids,
            class_names=class_names,
        )
        _write_json(run_dir / "evaluation.json", evaluation_payload)
        _write_json(_experiment_dir(settings.storage_root, project_id, experiment_id) / "evaluation.json", evaluation_payload)

        runtime_payload = _runtime_payload()
        _write_json(run_dir / "runtime.json", runtime_payload)
        _write_json(_experiment_dir(settings.storage_root, project_id, experiment_id) / "runtime.json", runtime_payload)

        (run_dir / "training.log").write_text(_training_log(), encoding="utf-8")

        onnx_dir = run_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        (onnx_dir / "model.onnx").write_bytes(b"demo-onnx-binary-content")
        _write_json(onnx_dir / "onnx.metadata.json", _onnx_metadata(class_ids, class_names))
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=DEMO_ATTEMPT,
            event={
                "type": "onnx_export",
                "status": "exported",
                "attempt": DEMO_ATTEMPT,
                "metadata_uri": _storage_relpath(settings.storage_root, onnx_dir / "onnx.metadata.json"),
                "model_uri": _storage_relpath(settings.storage_root, onnx_dir / "model.onnx"),
                "ts": utc_now_iso(),
            },
        )

        experiment_store.set_run_ended_at(project_id=project_id, experiment_id=experiment_id, attempt=DEMO_ATTEMPT)
        experiment_store.set_status(project_id=project_id, experiment_id=experiment_id, status="completed")
        experiment_store.append_event(
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=DEMO_ATTEMPT,
            event={"type": "done", "status": "completed", "attempt": DEMO_ATTEMPT, "job_id": DEMO_JOB_ID, "ts": utc_now_iso()},
        )

        return {
            "projectId": project_id,
            "taskId": task_id,
            "modelId": model_id,
            "datasetVersionId": dataset_version_id,
            "experimentId": experiment_id,
            "experimentName": _experiment_name(model_record, model_id),
            "attempt": DEMO_ATTEMPT,
            "task": task,
            "classIds": class_ids,
            "classNames": class_names,
            "onnxRelpath": _storage_relpath(settings.storage_root, onnx_dir / "model.onnx"),
            "onnxMetadataRelpath": _storage_relpath(settings.storage_root, onnx_dir / "onnx.metadata.json"),
            "datasetExportRelpath": str(dataset_export.get("zip_relpath") or ""),
            "storageRoot": str(storage.root),
        }


async def _async_main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "Usage: python -m sheriff_api.demo_experiment_seed <project_id> <task_id> <model_id> <dataset_version_id>",
            file=sys.stderr,
        )
        return 1
    metadata = await seed_demo_experiment(argv[1], argv[2], argv[3], argv[4])
    print(json.dumps(metadata, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_async_main(sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
