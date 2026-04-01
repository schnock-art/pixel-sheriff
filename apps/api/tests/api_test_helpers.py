from __future__ import annotations

import json
from pathlib import Path
import uuid

from httpx import AsyncClient
import sheriff_api.routers.models as models_router
from sheriff_api.config import get_settings

def assert_api_error(response, *, status_code: int, code: str, message: str | None = None) -> dict:
    assert response.status_code == status_code
    payload = response.json()
    assert "error" in payload
    error = payload["error"]
    assert error["code"] == code
    if message is not None:
        assert error["message"] == message
    details = error.get("details")
    assert isinstance(details, dict)
    assert details["request_path"]
    assert details["request_method"]
    return payload


async def _create_default_task_project(
    client: AsyncClient,
    *,
    name: str,
    task_type: str | None = None,
) -> dict:
    payload: dict[str, str] = {"name": name}
    if task_type is not None:
        payload["task_type"] = task_type
    response = await client.post("/api/v1/projects", json=payload)
    assert response.status_code == 200
    project = response.json()
    assert isinstance(project.get("default_task_id"), str) and project["default_task_id"]
    return project


async def _create_task_scoped_category(
    client: AsyncClient,
    *,
    project_id: str,
    task_id: str,
    name: str,
    display_order: int = 0,
) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": name, "display_order": display_order},
    )
    assert response.status_code == 200
    return response.json()


async def _create_dataset_export(
    client: AsyncClient,
    *,
    project_id: str,
    task_id: str,
    version_name: str,
    selection_filters: dict[str, object] | None = None,
) -> tuple[str, dict]:
    dataset_version_id = await _create_dataset_version_for_task(
        client,
        project_id=project_id,
        task_id=task_id,
        name=version_name,
        selection_filters=selection_filters,
    )
    response = await client.post(f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/export")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_version_id"] == dataset_version_id
    return dataset_version_id, payload

async def _create_detection_project_with_manifest(client: AsyncClient, *, project_name: str) -> tuple[str, dict]:
    project = (await client.post("/api/v1/projects", json={"name": project_name, "task_type": "bbox"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    category = (
        await client.post(
            f"/api/v1/projects/{project_id}/categories",
            json={"task_id": task_id, "name": "boat"},
        )
    ).json()
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category["id"]], "primary_category_id": category["id"]},
                "image_basis": {"width": 100, "height": 80},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 15]},
                ],
            },
        },
    )
    assert annotation.status_code == 200

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "detection-v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": True}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created_dataset.status_code == 200
    dataset_version = created_dataset.json()["version"]
    return project_id, {"label_schema": dataset_version["labels"]["label_schema"]}

async def _create_project_model(client: AsyncClient, *, project_name: str) -> tuple[str, str]:
    project_id, _manifest = await _create_detection_project_with_manifest(client, project_name=project_name)
    created = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created.status_code == 200
    return project_id, created.json()["id"]


async def _create_detection_project_model_with_categories(
    client: AsyncClient,
    *,
    project_name: str,
    category_names: list[str],
) -> tuple[str, str, str, list[str]]:
    project = (await client.post("/api/v1/projects", json={"name": project_name, "task_type": "bbox"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    category_ids: list[str] = []
    for index, name in enumerate(category_names):
        category_response = await client.post(
            f"/api/v1/projects/{project_id}/categories",
            json={"task_id": task_id, "name": name, "display_order": index},
        )
        assert category_response.status_code == 200
        category_ids.append(category_response.json()["id"])

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category_ids[0]], "primary_category_id": category_ids[0]},
                "image_basis": {"width": 100, "height": 80},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category_ids[0], "bbox": [10, 10, 20, 15]},
                ],
            },
        },
    )
    assert annotation.status_code == 200

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "detection-v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": True}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created_dataset.status_code == 200

    created_model = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created_model.status_code == 200
    return project_id, created_model.json()["id"], task_id, category_ids


async def _create_classification_project_model(client: AsyncClient, *, project_name: str) -> tuple[str, str, str]:
    project_id, model_id, task_id, _category_ids = await _create_classification_project_model_with_categories(
        client,
        project_name=project_name,
        category_names=["class-a"],
    )
    return project_id, model_id, task_id


async def _create_classification_project_model_with_categories(
    client: AsyncClient,
    *,
    project_name: str,
    category_names: list[str],
) -> tuple[str, str, str, list[str]]:
    project = (await client.post("/api/v1/projects", json={"name": project_name, "task_type": "classification_single"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    category_ids: list[str] = []
    for index, name in enumerate(category_names):
        category_response = await client.post(
            f"/api/v1/projects/{project_id}/categories",
            json={"task_id": task_id, "name": name, "display_order": index},
        )
        assert category_response.status_code == 200
        category_ids.append(category_response.json()["id"])

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "category_ids": [category_ids[0]],
                "classification": {"category_ids": [category_ids[0]], "primary_category_id": category_ids[0]},
                "image_basis": {"width": 100, "height": 80},
            },
        },
    )
    assert annotation.status_code == 200

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "classification-v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": True}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created_dataset.status_code == 200

    created_model = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created_model.status_code == 200
    return project_id, created_model.json()["id"], task_id, category_ids


async def _create_segmentation_project_model(client: AsyncClient, *, project_name: str) -> tuple[str, str, str]:
    project = (await client.post("/api/v1/projects", json={"name": project_name, "task_type": "segmentation"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    category_response = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": "pet"},
    )
    assert category_response.status_code == 200
    category_id = category_response.json()["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "classification": {"category_ids": [category_id], "primary_category_id": category_id},
                "image_basis": {"width": 100, "height": 80},
                "objects": [
                    {
                        "id": "poly-1",
                        "kind": "polygon",
                        "category_id": category_id,
                        "segmentation": [[10, 10, 30, 10, 30, 25, 10, 25]],
                    }
                ],
            },
        },
    )
    assert annotation.status_code == 200

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "segmentation-v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": True}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": True,
        },
    )
    assert created_dataset.status_code == 200

    created_model = await client.post(f"/api/v1/projects/{project_id}/models", json={})
    assert created_model.status_code == 200
    return project_id, created_model.json()["id"], task_id


def _seed_experiment_run_artifacts(
    *,
    project_id: str,
    experiment_id: str,
    attempt: int = 1,
    metrics_rows: list[dict] | None = None,
    log_content: str | None = None,
    include_onnx: bool = False,
    onnx_status: str = "exported",
) -> None:
    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    run_dir = experiment_dir / "runs" / str(attempt)
    run_dir.mkdir(parents=True, exist_ok=True)

    if metrics_rows is None:
        metrics_rows = [
            {"attempt": attempt, "epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.5},
            {"attempt": attempt, "epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "val_accuracy": 0.65},
            {"attempt": attempt, "epoch": 3, "train_loss": 0.6, "val_loss": 0.5, "val_accuracy": 0.72},
        ]

    metrics_content = "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in metrics_rows)
    (run_dir / "metrics.jsonl").write_text(metrics_content, encoding="utf-8")

    evaluation_payload = {
        "schema_version": "1",
        "task": "classification",
        "computed_at": "2025-01-01T00:00:00Z",
        "split": "val",
        "num_samples": 4,
        "provenance": {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "attempt": attempt,
            "model_id": "model-1",
            "task_id": "task-1",
            "job_id": "job-1",
            "dataset_version_id": "dv-1",
            "dataset_export_hash": "hash-1",
            "dataset_export_relpath": f"exports/{project_id}/hash-1.zip",
        },
        "classes": {
            "class_order": [1, 2],
            "class_names": ["one", "two"],
            "id_to_index": {"1": 0, "2": 1},
        },
        "overall": {
            "accuracy": 0.75,
            "macro_f1": 0.73,
            "macro_precision": 0.72,
            "macro_recall": 0.74,
        },
        "per_class": [
            {"class_index": 0, "class_id": 1, "name": "one", "precision": 0.8, "recall": 0.67, "f1": 0.73, "support": 3},
            {"class_index": 1, "class_id": 2, "name": "two", "precision": 0.67, "recall": 1.0, "f1": 0.8, "support": 1},
        ],
        "confusion_matrix": {
            "matrix": [[2, 1], [0, 1]],
            "normalized_by": "none",
            "labels": {"axis": "true_rows_pred_cols"},
        },
        "samples": {
            "misclassified": [
                {
                    "asset_id": "asset-1",
                    "relative_path": "assets/a1.jpg",
                    "true_class_index": 0,
                    "pred_class_index": 1,
                    "confidence": 0.95,
                    "margin": 0.70,
                }
            ],
            "lowest_confidence_correct": [],
            "highest_confidence_wrong": [
                {
                    "asset_id": "asset-1",
                    "relative_path": "assets/a1.jpg",
                    "true_class_index": 0,
                    "pred_class_index": 1,
                    "confidence": 0.95,
                    "margin": 0.70,
                }
            ],
        },
    }
    predictions_rows = [
        {"asset_id": "asset-0", "relative_path": "assets/a0.jpg", "true_class_index": 0, "pred_class_index": 0, "confidence": 0.81, "margin": 0.52},
        {"asset_id": "asset-1", "relative_path": "assets/a1.jpg", "true_class_index": 0, "pred_class_index": 1, "confidence": 0.95, "margin": 0.70},
        {"asset_id": "asset-2", "relative_path": "assets/a2.jpg", "true_class_index": 0, "pred_class_index": 0, "confidence": 0.55, "margin": 0.11},
        {"asset_id": "asset-3", "relative_path": "assets/a3.jpg", "true_class_index": 1, "pred_class_index": 1, "confidence": 0.60, "margin": 0.22},
    ]
    predictions_content = "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in predictions_rows)
    predictions_meta = {
        "schema_version": "1",
        "attempt": attempt,
        "num_samples": len(predictions_rows),
        "task": "classification",
        "split": "val",
        "computed_at": "2025-01-01T00:00:00Z",
        "provenance": {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "attempt": attempt,
            "model_id": "model-1",
            "task_id": "task-1",
            "job_id": "job-1",
            "dataset_version_id": "dv-1",
            "dataset_export_hash": "hash-1",
            "dataset_export_relpath": f"exports/{project_id}/hash-1.zip",
        },
    }

    for target in [run_dir / "evaluation.json", experiment_dir / "evaluation.json"]:
        target.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")
    for target in [run_dir / "predictions.jsonl", experiment_dir / "predictions.jsonl"]:
        target.write_text(predictions_content, encoding="utf-8")
    for target in [run_dir / "predictions.meta.json", experiment_dir / "predictions.meta.json"]:
        target.write_text(json.dumps(predictions_meta, indent=2, sort_keys=True), encoding="utf-8")

    runtime_payload = {
        "device_selected": "cuda",
        "cuda_available": True,
        "mps_available": False,
        "amp_enabled": True,
        "torch_version": "2.x",
        "torchvision_version": "0.x",
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }
    for target in [run_dir / "runtime.json", experiment_dir / "runtime.json"]:
        target.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True), encoding="utf-8")

    if log_content is None:
        log_content = (
            "epoch=1 train_loss=0.90 val_loss=0.80 val_accuracy=0.50\n"
            "epoch=2 train_loss=0.70 val_loss=0.60 val_accuracy=0.65\n"
        )
    (run_dir / "training.log").write_text(log_content, encoding="utf-8")

    if include_onnx:
        onnx_dir = run_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        if onnx_status == "exported":
            (onnx_dir / "model.onnx").write_bytes(b"fake-onnx-binary-content")
        (onnx_dir / "onnx.metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": onnx_status,
                    "attempt": attempt,
                    "input_shape": [3, 224, 224],
                    "class_order": ["one", "two"],
                    "class_names": ["one", "two"],
                    "preprocess": {
                        "resize": {"width": 224, "height": 224},
                        "normalization": {"type": "imagenet"},
                    },
                    "validation": {"status": "passed" if onnx_status == "exported" else "failed"},
                    "error": None if onnx_status == "exported" else "onnxruntime validation failed",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    status_path = experiment_dir / "status.json"
    status_payload = {}
    if status_path.exists():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload.update(
        {
            "status": "completed",
            "current_run_attempt": attempt,
            "last_completed_attempt": attempt,
            "active_job_id": None,
            "error": None,
        }
    )
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_experiment_variant_artifacts(
    *,
    project_id: str,
    experiment_id: str,
    attempt: int = 1,
    variant_key: str,
    preferred_variant_key: str | None = None,
    status: str = "ready",
    task: str = "classification",
) -> None:
    settings = get_settings()
    experiment_dir = Path(settings.storage_root) / "experiments" / project_id / experiment_id
    variant_dir = experiment_dir / "runs" / str(attempt) / "variants" / variant_key
    onnx_dir = variant_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    (onnx_dir / "model.onnx").write_bytes(f"fake-{variant_key}-onnx".encode("utf-8"))
    (onnx_dir / "onnx.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "status": "exported" if status == "ready" else "failed",
                "attempt": attempt,
                "variant_key": variant_key,
                "input_shape": [3, 224, 224],
                "class_order": ["one", "two"],
                "class_names": ["one", "two"],
                "preprocess": {
                    "resize": {"width": 224, "height": 224},
                    "normalization": {"type": "imagenet"},
                },
                "validation": {"status": "passed" if status == "ready" else "failed"},
                "task": task,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for split in ["val", "test"]:
        (variant_dir / f"evaluation.{split}.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "task": task,
                    "split": split,
                    "status": "ready",
                    "overall": {"accuracy": 0.75, "macro_f1": 0.73, "mAP50": 0.66, "mAP50_95": 0.61},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    (variant_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "status": "ready",
                "provider": "CPUExecutionProvider",
                "mean_latency_ms": 12.3,
                "throughput_items_per_second": 81.3,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    index_path = experiment_dir / "runs" / str(attempt) / "variants" / "index.json"
    if index_path.exists():
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_payload = {"schema_version": "1", "attempt": attempt, "preferred_variant_key": preferred_variant_key or variant_key, "variants": {}}
    if not isinstance(index_payload.get("variants"), dict):
        index_payload["variants"] = {}
    index_payload["variants"][variant_key] = {
        "variant_key": variant_key,
        "label": variant_key,
        "kind": "baseline" if variant_key == "fp32" else ("ptq" if variant_key == "ptq_int8" else "qat"),
        "attempt": attempt,
        "status": status,
        "preferred": preferred_variant_key == variant_key if preferred_variant_key else variant_key == index_payload.get("preferred_variant_key"),
        "onnx": {
            "model_relpath": str((onnx_dir / "model.onnx").relative_to(Path(settings.storage_root))).replace("\\", "/"),
            "metadata_relpath": str((onnx_dir / "onnx.metadata.json").relative_to(Path(settings.storage_root))).replace("\\", "/"),
            "size_bytes": int((onnx_dir / "model.onnx").stat().st_size),
        },
        "evaluation": {
            split: {
                "status": "ready",
                "relpath": str((variant_dir / f"evaluation.{split}.json").relative_to(Path(settings.storage_root))).replace("\\", "/"),
                "overall": {"accuracy": 0.75, "macro_f1": 0.73, "mAP50": 0.66, "mAP50_95": 0.61},
            }
            for split in ["val", "test"]
        },
        "benchmark": {
            "status": "ready",
            "provider": "CPUExecutionProvider",
            "mean_latency_ms": 12.3,
            "throughput_items_per_second": 81.3,
        },
    }
    index_payload["preferred_variant_key"] = preferred_variant_key or index_payload.get("preferred_variant_key") or variant_key
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")

async def _create_detection_project_with_dataset_version(
    client: AsyncClient, *, project_name: str
) -> tuple[str, str, str]:
    """Return (project_id, task_id, dataset_version_id) for a bbox project with one annotated asset."""
    project = (await client.post("/api/v1/projects", json={"name": project_name, "task_type": "bbox"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    cat_resp = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"name": "boat", "task_id": task_id},
    )
    assert cat_resp.status_code == 200
    category = cat_resp.json()

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset = upload.json()

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "asset_id": asset["id"],
            "task_id": task_id,
            "status": "approved",
            "payload_json": {
                "version": "2.0",
                "image_basis": {"width": 100, "height": 80},
                "objects": [
                    {"id": "bbox-1", "kind": "bbox", "category_id": category["id"], "bbox": [10, 10, 20, 15]},
                ],
            },
        },
    )
    assert annotation.status_code == 200

    version_resp = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": {}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
        },
    )
    assert version_resp.status_code == 200
    dataset_version_id = version_resp.json()["version"]["dataset_version_id"]

    return project_id, task_id, dataset_version_id


async def _create_dataset_version_for_task(
    client: AsyncClient,
    *,
    project_id: str,
    task_id: str,
    name: str,
    set_active: bool = True,
    selection_filters: dict[str, object] | None = None,
) -> str:
    version_resp = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": name,
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot", "filters": selection_filters or {}},
            "split": {
                "seed": 42,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": set_active,
        },
    )
    assert version_resp.status_code == 200
    return version_resp.json()["version"]["dataset_version_id"]
