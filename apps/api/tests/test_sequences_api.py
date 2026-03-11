from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import zipfile

from httpx import AsyncClient
import pytest
import sheriff_api.routers.video_imports as video_imports_router
from sheriff_api.config import get_settings


async def _create_project(client: AsyncClient, *, name: str) -> dict:
    response = await client.post("/api/v1/projects", json={"name": name})
    assert response.status_code == 200
    return response.json()


async def _create_category(client: AsyncClient, *, project_id: str, task_id: str, name: str) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": name},
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_asset_upload_populates_canonical_folder_fields_and_lists_folders(client: AsyncClient) -> None:
    project = await _create_project(client, name="folder-fields")
    project_id = project["id"]

    uploaded = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": "train/cats/kitten.jpg"},
        files={"file": ("kitten.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert uploaded.status_code == 200
    asset = uploaded.json()

    assert asset["folder_path"] == "train/cats"
    assert asset["file_name"] == "kitten.jpg"
    assert asset["relative_path"] == "train/cats/kitten.jpg"
    assert asset["source_kind"] == "image"

    listed_assets = await client.get(f"/api/v1/projects/{project_id}/assets")
    assert listed_assets.status_code == 200
    assert listed_assets.json()[0]["relative_path"] == "train/cats/kitten.jpg"

    folders = await client.get(f"/api/v1/projects/{project_id}/folders")
    assert folders.status_code == 200
    assert [folder["path"] for folder in folders.json()] == ["train", "train/cats"]
    assert folders.json()[-1]["asset_count"] == 1


@pytest.mark.asyncio
async def test_video_import_creates_processing_sequence_and_enqueues_media_job(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued: dict[str, object] = {}

    async def fake_enqueue(payload: dict[str, object]) -> None:
        enqueued["payload"] = payload

    monkeypatch.setattr(video_imports_router.media_queue, "enqueue_extract_video_job", fake_enqueue)

    project = await _create_project(client, name="video-import")
    project_id = project["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/video-imports",
        data={"task_id": project["default_task_id"], "fps": "2", "max_frames": "12", "name": "clip-session"},
        files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    sequence = response.json()["sequence"]

    assert sequence["status"] == "processing"
    assert sequence["source_type"] == "video_file"
    assert sequence["name"] == "clip-session"
    assert isinstance(sequence["folder_id"], str)

    payload = enqueued["payload"]
    assert payload["sequence_id"] == sequence["id"]
    assert payload["folder_id"] == sequence["folder_id"]
    assert payload["project_id"] == project_id
    assert payload["fps"] == 2.0
    assert payload["max_frames"] == 12

    status_response = await client.get(f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "processing"

    sequence_list = await client.get(
        f"/api/v1/projects/{project_id}/sequences",
        params={"task_id": project["default_task_id"]},
    )
    assert sequence_list.status_code == 200
    assert [item["id"] for item in sequence_list.json()] == [sequence["id"]]

    folders = await client.get(f"/api/v1/projects/{project_id}/folders")
    assert folders.status_code == 200
    assert folders.json()[0]["sequence_status"] == "processing"
    assert folders.json()[0]["sequence_id"] == sequence["id"]


@pytest.mark.asyncio
async def test_webcam_sequence_detail_sorts_frames_and_marks_annotation_coverage(client: AsyncClient) -> None:
    project = await _create_project(client, name="webcam-sequence")
    project_id = project["id"]
    task_id = project["default_task_id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={"task_id": task_id, "name": "webcam-session", "fps": 2},
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    upload_b = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "1", "timestamp_seconds": "0.5"},
        files={"file": ("frame_000002.jpg", b"frame-b", "image/jpeg")},
    )
    assert upload_b.status_code == 200

    upload_a = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "0", "timestamp_seconds": "0.0"},
        files={"file": ("frame_000001.jpg", b"frame-a", "image/jpeg")},
    )
    assert upload_a.status_code == 200

    category = await _create_category(client, project_id=project_id, task_id=task_id, name="person")
    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": upload_b.json()["id"],
            "status": "approved",
            "payload_json": {"category_ids": [category["id"]]},
        },
    )
    assert annotation.status_code == 200

    detail = await client.get(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}",
        params={"task_id": task_id},
    )
    assert detail.status_code == 200
    payload = detail.json()

    assert payload["status"] == "ready"
    assert [asset["frame_index"] for asset in payload["assets"]] == [0, 1]
    assert [asset["source_kind"] for asset in payload["assets"]] == ["webcam_frame", "webcam_frame"]
    assert payload["assets"][0]["has_annotations"] is False
    assert payload["assets"][1]["has_annotations"] is True

    duplicate = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "1", "timestamp_seconds": "1.0"},
        files={"file": ("frame_000002.jpg", b"dup", "image/jpeg")},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "sequence_frame_exists"


@pytest.mark.asyncio
async def test_webcam_session_create_accepts_explicit_folder_path(client: AsyncClient) -> None:
    project = await _create_project(client, name="webcam-folder-path")
    project_id = project["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={
            "task_id": project["default_task_id"],
            "name": "line-a",
            "fps": 2,
            "folder_path": "captures/loading-bay/cam-a",
        },
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    assert sequence["folder_path"] == "captures/loading-bay/cam-a"

    folders = await client.get(f"/api/v1/projects/{project_id}/folders")
    assert folders.status_code == 200
    assert [folder["path"] for folder in folders.json()] == [
        "captures",
        "captures/loading-bay",
        "captures/loading-bay/cam-a",
    ]
    assert folders.json()[-1]["sequence_id"] == sequence["id"]


@pytest.mark.asyncio
async def test_delete_folder_removes_empty_sequence_folder(client: AsyncClient) -> None:
    project = await _create_project(client, name="delete-empty-folder")
    project_id = project["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={"task_id": project["default_task_id"], "name": "empty-webcam", "fps": 2},
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    deleted = await client.delete(f"/api/v1/projects/{project_id}/folders/{sequence['folder_id']}")
    assert deleted.status_code == 204

    folders = await client.get(f"/api/v1/projects/{project_id}/folders")
    assert folders.status_code == 200
    assert folders.json() == []

    sequences = await client.get(f"/api/v1/projects/{project_id}/sequences")
    assert sequences.status_code == 200
    assert sequences.json() == []


@pytest.mark.asyncio
async def test_export_manifest_includes_sequence_lineage_metadata(client: AsyncClient) -> None:
    project = await _create_project(client, name="export-lineage")
    project_id = project["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/webcam-sessions",
        json={"task_id": project["default_task_id"], "name": "lineage-cam", "fps": 2},
    )
    assert created.status_code == 200
    sequence = created.json()["sequence"]

    uploaded = await client.post(
        f"/api/v1/projects/{project_id}/sequences/{sequence['id']}/frames",
        data={"frame_index": "3", "timestamp_seconds": "1.5"},
        files={"file": ("frame_000004.jpg", b"frame-export", "image/jpeg")},
    )
    assert uploaded.status_code == 200
    await _create_category(client, project_id=project_id, task_id=project["default_task_id"], name="frame")

    created_dataset = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "lineage-dataset",
            "task_id": project["default_task_id"],
            "selection": {"mode": "filter_snapshot", "filters": {}},
            "split": {
                "seed": 7,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": False, "by": "label_primary"},
            },
            "set_active": False,
        },
    )
    assert created_dataset.status_code == 200
    dataset_version = created_dataset.json()["version"]

    export_response = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version['dataset_version_id']}/export"
    )
    assert export_response.status_code == 200

    export_zip = Path(get_settings().storage_root) / "exports" / project_id / f"{export_response.json()['hash']}.zip"
    assert export_zip.exists()

    with zipfile.ZipFile(BytesIO(export_zip.read_bytes()), "r") as bundle:
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))

    assert len(manifest["assets"]) == 1
    source_meta = manifest["assets"][0]["meta"]["source"]
    assert source_meta == {
        "kind": "webcam_frame",
        "sequence_id": sequence["id"],
        "sequence_name": "lineage-cam",
        "frame_index": 3,
        "timestamp_seconds": 1.5,
    }


@pytest.mark.asyncio
async def test_delete_project_removes_pending_video_import_files(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_enqueue(_payload: dict[str, object]) -> None:
        return None

    monkeypatch.setattr(video_imports_router.media_queue, "enqueue_extract_video_job", fake_enqueue)

    project = await _create_project(client, name="delete-import-root")
    project_id = project["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/video-imports",
        data={"fps": "2", "max_frames": "8", "name": "pending-video"},
        files={"file": ("pending.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    sequence = response.json()["sequence"]

    imports_root = Path(get_settings().storage_root) / "imports" / project_id / sequence["id"]
    assert imports_root.exists()

    deleted = await client.delete(f"/api/v1/projects/{project_id}")
    assert deleted.status_code == 204
    assert not imports_root.exists()
