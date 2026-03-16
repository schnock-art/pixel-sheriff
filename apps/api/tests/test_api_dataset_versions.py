from __future__ import annotations

from pathlib import Path

from httpx import AsyncClient
import pytest

@pytest.mark.asyncio
async def test_dataset_preview_filters_respect_exclude_statuses_and_exclude_folder_precedence(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "dataset-filter-precedence"})).json()
    project_id = project["id"]

    async def _upload(relative_path: str) -> dict:
        response = await client.post(
            f"/api/v1/projects/{project_id}/assets/upload",
            data={"relative_path": relative_path},
            files={"file": (Path(relative_path).name, b"fake-image-bytes", "image/jpeg")},
        )
        assert response.status_code == 200
        return response.json()

    cat_a = await _upload("animals/cats/a.jpg")
    cat_b = await _upload("animals/cats/b.jpg")
    dog_c = await _upload("animals/dogs/c.jpg")
    misc_d = await _upload("misc/d.jpg")

    await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": cat_a["id"], "task_id": project["default_task_id"], "status": "labeled", "payload_json": {}},
    )
    await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": cat_b["id"], "task_id": project["default_task_id"], "status": "approved", "payload_json": {}},
    )
    await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": dog_c["id"], "task_id": project["default_task_id"], "status": "needs_review", "payload_json": {}},
    )
    await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={"asset_id": misc_d["id"], "task_id": project["default_task_id"], "status": "skipped", "payload_json": {}},
    )

    preview = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions/preview",
        json={
            "task_id": project["default_task_id"],
            "selection": {
                "mode": "filter_snapshot",
                "filters": {
                    "include_statuses": ["labeled", "approved", "needs_review"],
                    "exclude_statuses": ["approved"],
                    "include_folder_paths": ["animals"],
                    "exclude_folder_paths": ["animals/cats"],
                },
            },
            "split": {
                "seed": 1337,
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "stratify": {"enabled": True, "by": "label_primary", "strict_stratify": False},
            },
        },
    )
    assert preview.status_code == 200
    payload = preview.json()

    # include/exclude folders use: final_membership = included_set - excluded_set (exclude wins)
    assert payload["counts"]["total"] == 1
    assert payload["counts"]["split_counts"]["train"] + payload["counts"]["split_counts"]["val"] + payload["counts"]["split_counts"]["test"] == 1


@pytest.mark.asyncio
async def test_dataset_preview_include_folder_empty_means_no_restriction(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "dataset-folder-include-empty"})).json()
    project_id = project["id"]

    async def _upload(relative_path: str) -> None:
        response = await client.post(
            f"/api/v1/projects/{project_id}/assets/upload",
            data={"relative_path": relative_path},
            files={"file": (Path(relative_path).name, b"fake-image-bytes", "image/jpeg")},
        )
        assert response.status_code == 200

    await _upload("animals/cats/a.jpg")
    await _upload("animals/dogs/b.jpg")
    await _upload("misc/c.jpg")

    preview = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions/preview",
        json={
            "task_id": project["default_task_id"],
            "selection": {
                "mode": "filter_snapshot",
                "filters": {
                    "include_folder_paths": [],
                    "exclude_folder_paths": ["misc"],
                },
            },
        },
    )
    assert preview.status_code == 200
    payload = preview.json()

    assert payload["counts"]["total"] == 2


@pytest.mark.asyncio
async def test_dataset_preview_returns_sample_asset_metadata_and_class_names(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "dataset-preview-samples"})).json()
    project_id = project["id"]
    task_id = project["default_task_id"]

    category = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": "flower"},
    )
    assert category.status_code == 200
    category_id = category.json()["id"]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        data={"relative_path": "flowers/rose.jpg"},
        files={"file": ("rose.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    annotation = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset_id,
            "status": "approved",
            "payload_json": {
                "category_id": category_id,
                "category_ids": [category_id],
                "classification": {"category_ids": [category_id], "primary_category_id": category_id},
            },
        },
    )
    assert annotation.status_code == 200

    preview = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions/preview",
        json={
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot"},
            "split": {
                "seed": 1337,
                "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                "stratify": {"enabled": True, "by": "label_primary", "strict_stratify": False},
            },
        },
    )
    assert preview.status_code == 200
    payload = preview.json()

    assert payload["class_names"][category_id] == "flower"
    assert payload["sample_asset_ids"] == [asset_id]
    assert len(payload["sample_assets"]) == 1
    assert payload["sample_assets"][0]["asset_id"] == asset_id
    assert payload["sample_assets"][0]["relative_path"] == "flowers/rose.jpg"
    assert payload["sample_assets"][0]["status"] == "approved"
    assert payload["sample_assets"][0]["split"] == "train"
    assert payload["sample_assets"][0]["label_summary"]["primary_category_id"] == category_id


@pytest.mark.asyncio
async def test_dataset_saved_split_membership_comes_from_stored_split_map(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/projects", json={"name": "dataset-split-membership"})).json()
    project_id = project["id"]

    uploaded_assets: list[dict] = []
    for index in range(12):
        response = await client.post(
            f"/api/v1/projects/{project_id}/assets/upload",
            data={"relative_path": f"set/sample_{index}.jpg"},
            files={"file": (f"sample_{index}.jpg", b"fake-image-bytes", "image/jpeg")},
        )
        assert response.status_code == 200
        uploaded_assets.append(response.json())

    created = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "v1",
            "task_id": project["default_task_id"],
            "selection": {"mode": "filter_snapshot", "filters": {"include_labeled_only": False}},
            "split": {
                "seed": 1337,
                "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                "stratify": {"enabled": True, "by": "label_primary", "strict_stratify": False},
            },
        },
    )
    assert created.status_code == 200
    payload = created.json()
    dataset_version_id = payload["version"]["dataset_version_id"]
    split_counts = payload["version"]["stats"]["split_counts"]

    # Change live annotation status after version creation; split membership should remain stable.
    for asset in uploaded_assets[:4]:
        response = await client.post(
            f"/api/v1/projects/{project_id}/annotations",
            json={"asset_id": asset["id"], "task_id": project["default_task_id"], "status": "approved", "payload_json": {}},
        )
        assert response.status_code == 200

    train_assets = await client.get(
        f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/assets",
        params={"split": "train", "page": 1, "page_size": 250},
    )
    val_assets = await client.get(
        f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/assets",
        params={"split": "val", "page": 1, "page_size": 250},
    )
    test_assets = await client.get(
        f"/api/v1/projects/{project_id}/datasets/versions/{dataset_version_id}/assets",
        params={"split": "test", "page": 1, "page_size": 250},
    )
    assert train_assets.status_code == 200
    assert val_assets.status_code == 200
    assert test_assets.status_code == 200

    assert train_assets.json()["total"] == split_counts["train"]
    assert val_assets.json()["total"] == split_counts["val"]
    assert test_assets.json()["total"] == split_counts["test"]
