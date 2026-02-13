def run(payload: dict) -> dict:
    if "dataset_version_id" not in payload:
        raise ValueError("dataset_version_id is required")
    return {"status": "done", "export_uri": f"exports/{payload['dataset_version_id']}.zip"}
