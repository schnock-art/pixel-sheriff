import uuid

from httpx import AsyncClient
import pytest


def assert_error_code(response, status_code: int, code: str) -> dict:
    assert response.status_code == status_code
    payload = response.json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get("error"), dict)
    assert payload["error"].get("code") == code
    return payload


async def _create_project(client: AsyncClient, name: str = "multi-task") -> dict:
    response = await client.post("/api/v1/projects", json={"name": name})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("default_task_id"), str)
    return payload


async def _create_category(client: AsyncClient, project_id: str, task_id: str, name: str) -> dict:
    response = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": name},
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_project_has_default_task_and_tasks_list(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-default")
    project_id = project["id"]
    default_task_id = project["default_task_id"]

    listed = await client.get(f"/api/v1/projects/{project_id}/tasks")
    assert listed.status_code == 200
    tasks = listed.json()
    assert isinstance(tasks, list) and len(tasks) == 1
    assert tasks[0]["id"] == default_task_id
    assert tasks[0]["is_default"] is True
    assert tasks[0]["kind"] == "classification"
    assert tasks[0]["label_mode"] in {"single_label", "multi_label"}


@pytest.mark.asyncio
async def test_categories_and_annotations_are_task_scoped(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-scoped")
    project_id = project["id"]
    task_a = project["default_task_id"]

    created_task = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json={"name": "Boxes", "kind": "bbox"},
    )
    assert created_task.status_code == 200
    task_b = created_task.json()["id"]

    missing_task_create_category = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"name": "cat"},
    )
    assert missing_task_create_category.status_code == 422

    category_a = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_a, "name": "cat-a"},
    )
    assert category_a.status_code == 200
    category_a_id = category_a.json()["id"]

    category_b = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_b, "name": "cat-b"},
    )
    assert category_b.status_code == 200
    category_b_id = category_b.json()["id"]

    listed_a = await client.get(f"/api/v1/projects/{project_id}/categories", params={"task_id": task_a})
    listed_b = await client.get(f"/api/v1/projects/{project_id}/categories", params={"task_id": task_b})
    assert listed_a.status_code == 200
    assert listed_b.status_code == 200
    assert [row["id"] for row in listed_a.json()] == [category_a_id]
    assert [row["id"] for row in listed_b.json()] == [category_b_id]

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    ann_a = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_a,
            "asset_id": asset_id,
            "status": "labeled",
            "payload_json": {
                "category_ids": [category_a_id],
                "classification": {"category_ids": [category_a_id], "primary_category_id": category_a_id},
            },
        },
    )
    assert ann_a.status_code == 200

    ann_b = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_b,
            "asset_id": asset_id,
            "status": "labeled",
            "payload_json": {"objects": [{"id": str(uuid.uuid4()), "kind": "bbox", "category_id": category_b_id, "bbox": [1, 1, 10, 10]}]},
        },
    )
    assert ann_b.status_code == 200

    by_task_a = await client.get(f"/api/v1/projects/{project_id}/annotations", params={"task_id": task_a})
    by_task_b = await client.get(f"/api/v1/projects/{project_id}/annotations", params={"task_id": task_b})
    assert by_task_a.status_code == 200
    assert by_task_b.status_code == 200
    assert len(by_task_a.json()) == 1
    assert len(by_task_b.json()) == 1
    assert by_task_a.json()[0]["task_id"] == task_a
    assert by_task_b.json()[0]["task_id"] == task_b


@pytest.mark.asyncio
async def test_dataset_preview_requires_task_id(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-preview")
    project_id = project["id"]
    task_id = project["default_task_id"]

    missing_task_id = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions/preview",
        json={
            "selection": {"mode": "filter_snapshot"},
            "split": {"seed": 1337, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "stratify": {"enabled": False, "by": "label_primary"}},
        },
    )
    assert missing_task_id.status_code == 422

    valid = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions/preview",
        json={
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot"},
            "split": {"seed": 1337, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "stratify": {"enabled": False, "by": "label_primary"}},
        },
    )
    assert valid.status_code == 200
    payload = valid.json()
    assert isinstance(payload.get("counts"), dict)


@pytest.mark.asyncio
async def test_task_create_validation_and_label_mode_default(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-validation")
    project_id = project["id"]

    created = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json={"name": "Classifier B", "kind": "classification"},
    )
    assert created.status_code == 200
    assert created.json()["label_mode"] == "single_label"

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json={"name": "Boxes bad", "kind": "bbox", "label_mode": "single_label"},
    )
    assert_error_code(invalid, 422, "validation_error")


@pytest.mark.asyncio
async def test_task_delete_guardrails_and_default_reassignment(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-delete")
    project_id = project["id"]
    default_task_id = project["default_task_id"]

    delete_last = await client.delete(f"/api/v1/projects/{project_id}/tasks/{default_task_id}")
    assert_error_code(delete_last, 409, "project_must_have_task")

    next_task = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json={"name": "Boxes", "kind": "bbox"},
    )
    assert next_task.status_code == 200
    next_task_id = next_task.json()["id"]

    # Default task is empty in this test, so deleting it should reassign default deterministically.
    deleted_default = await client.delete(f"/api/v1/projects/{project_id}/tasks/{default_task_id}")
    assert deleted_default.status_code == 200
    assert deleted_default.json()["ok"] is True

    refreshed_project = await client.get(f"/api/v1/projects/{project_id}")
    assert refreshed_project.status_code == 200
    assert refreshed_project.json()["default_task_id"] == next_task_id

    # Add another empty task so "last task" guard does not mask "non-empty" guard.
    keepalive = await client.post(
        f"/api/v1/projects/{project_id}/tasks",
        json={"name": "Seg", "kind": "segmentation"},
    )
    assert keepalive.status_code == 200

    # Non-empty task cannot be deleted.
    await _create_category(client, project_id, next_task_id, "box-class")
    delete_non_empty = await client.delete(f"/api/v1/projects/{project_id}/tasks/{next_task_id}")
    payload = assert_error_code(delete_non_empty, 409, "task_not_empty")
    references = payload["error"]["details"]["references"]
    assert references["categories"] >= 1


@pytest.mark.asyncio
async def test_single_label_task_rejects_multilabel_annotation_payload(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-single-label")
    project_id = project["id"]
    task_id = project["default_task_id"]

    cat_a = await _create_category(client, project_id, task_id, "A")
    cat_b = await _create_category(client, project_id, task_id, "B")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    invalid = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset_id,
            "status": "labeled",
            "payload_json": {
                "classification": {
                    "category_ids": [cat_a["id"], cat_b["id"]],
                    "primary_category_id": cat_a["id"],
                }
            },
        },
    )
    assert_error_code(invalid, 422, "validation_error")


@pytest.mark.asyncio
async def test_dataset_create_persists_task_id_and_label_mode(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-dataset")
    project_id = project["id"]
    task_id = project["default_task_id"]

    category = await _create_category(client, project_id, task_id, "cat")
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    labeled = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset_id,
            "status": "labeled",
            "payload_json": {
                "classification": {
                    "category_ids": [category["id"]],
                    "primary_category_id": category["id"],
                }
            },
        },
    )
    assert labeled.status_code == 200

    created = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot"},
            "split": {"seed": 1337, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "stratify": {"enabled": False, "by": "label_primary"}},
        },
    )
    assert created.status_code == 200
    version = created.json()["version"]
    assert version["task_id"] == task_id
    assert version["task"] == "classification"
    assert version["labels"]["label_mode"] == "single_label"


@pytest.mark.asyncio
async def test_category_mutations_are_locked_once_task_has_dataset_versions(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-category-lock")
    project_id = project["id"]
    task_id = project["default_task_id"]

    category = await _create_category(client, project_id, task_id, "cat")
    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    labeled = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset_id,
            "status": "labeled",
            "payload_json": {
                "classification": {
                    "category_ids": [category["id"]],
                    "primary_category_id": category["id"],
                }
            },
        },
    )
    assert labeled.status_code == 200

    created = await client.post(
        f"/api/v1/projects/{project_id}/datasets/versions",
        json={
            "name": "v1",
            "task_id": task_id,
            "selection": {"mode": "filter_snapshot"},
            "split": {"seed": 1337, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "stratify": {"enabled": False, "by": "label_primary"}},
        },
    )
    assert created.status_code == 200

    create_locked = await client.post(
        f"/api/v1/projects/{project_id}/categories",
        json={"task_id": task_id, "name": "new-class"},
    )
    assert_error_code(create_locked, 409, "task_locked_by_dataset")

    patch_locked = await client.patch(f"/api/v1/categories/{category['id']}", json={"name": "cat-renamed"})
    assert_error_code(patch_locked, 409, "task_locked_by_dataset")

    delete_locked = await client.delete(f"/api/v1/categories/{category['id']}")
    assert_error_code(delete_locked, 409, "task_locked_by_dataset")


@pytest.mark.asyncio
async def test_delete_unused_category_succeeds(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-delete-unused")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id, task_id, "temp-class")

    deleted = await client.delete(f"/api/v1/categories/{category['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True

    listed = await client.get(f"/api/v1/projects/{project_id}/categories", params={"task_id": task_id})
    assert listed.status_code == 200
    assert all(row["id"] != category["id"] for row in listed.json())


@pytest.mark.asyncio
async def test_delete_category_fails_when_annotation_references_exist(client: AsyncClient) -> None:
    project = await _create_project(client, name="multi-task-delete-in-use")
    project_id = project["id"]
    task_id = project["default_task_id"]
    category = await _create_category(client, project_id, task_id, "kept-class")

    upload = await client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]
    labeled = await client.post(
        f"/api/v1/projects/{project_id}/annotations",
        json={
            "task_id": task_id,
            "asset_id": asset_id,
            "status": "labeled",
            "payload_json": {
                "classification": {
                    "category_ids": [category["id"]],
                    "primary_category_id": category["id"],
                }
            },
        },
    )
    assert labeled.status_code == 200

    delete_in_use = await client.delete(f"/api/v1/categories/{category['id']}")
    payload = assert_error_code(delete_in_use, 409, "category_in_use")
    assert int(payload["error"]["details"].get("annotation_references", 0)) >= 1
