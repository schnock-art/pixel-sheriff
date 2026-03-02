import json

import sheriff_api.services.model_store as model_store_module


def test_project_model_store_project_isolation_and_update_contract(tmp_path) -> None:
    store = model_store_module.create_project_model_store(str(tmp_path))
    assert isinstance(store, model_store_module.FileProjectModelStore)

    project_a_record = store.create(project_id="project-a", name="model-a", config_json={"epochs": 3})
    project_b_record = store.create(project_id="project-b", name="model-b", config_json={"epochs": 5})

    assert store.get("project-a", project_a_record["id"]) is not None
    assert store.get("project-a", project_b_record["id"]) is None
    assert store.get("project-b", project_a_record["id"]) is None
    assert [row["id"] for row in store.list_by_project("project-a")] == [project_a_record["id"]]

    updated = store.update_config(project_id="project-a", model_id=project_a_record["id"], config_json={"epochs": 7})
    assert updated is not None
    assert updated["config_json"] == {"epochs": 7}
    assert store.get("project-a", project_a_record["id"])["config_json"] == {"epochs": 7}


def test_project_model_store_lists_descending_by_created_at(tmp_path, monkeypatch) -> None:
    timestamps = iter(
        [
            "2025-01-01T00:00:01Z",
            "2025-01-01T00:00:02Z",
            "2025-01-01T00:00:03Z",
        ]
    )
    monkeypatch.setattr(model_store_module, "_utc_now_iso", lambda: next(timestamps))
    store = model_store_module.FileProjectModelStore(str(tmp_path))

    first = store.create(project_id="project-a", name="first", config_json={"v": 1})
    second = store.create(project_id="project-a", name="second", config_json={"v": 2})
    third = store.create(project_id="project-a", name="third", config_json={"v": 3})

    rows = store.list_by_project("project-a")
    assert [row["id"] for row in rows] == [third["id"], second["id"], first["id"]]


def test_project_model_store_read_records_falls_back_to_empty_on_corrupt_or_non_list_payload(tmp_path) -> None:
    store = model_store_module.FileProjectModelStore(str(tmp_path))
    records_path = tmp_path / "models" / "project-a" / "records.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    records_path.write_text("{invalid-json", encoding="utf-8")
    assert store.list_by_project("project-a") == []

    records_path.write_text(json.dumps({"not": "a-list"}), encoding="utf-8")
    assert store.list_by_project("project-a") == []


def test_project_model_store_update_returns_none_for_missing_model(tmp_path) -> None:
    store = model_store_module.FileProjectModelStore(str(tmp_path))
    updated = store.update_config(project_id="project-a", model_id="missing-model", config_json={"epochs": 1})
    assert updated is None
