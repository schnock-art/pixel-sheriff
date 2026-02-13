def run(payload: dict) -> dict:
    if "video_uri" not in payload:
        raise ValueError("video_uri is required")
    return {"status": "done", "frames_extracted": payload.get("fps", 1)}
