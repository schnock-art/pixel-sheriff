def run(payload: dict) -> dict:
    return {"status": "queued", "project_id": payload.get("project_id")}
