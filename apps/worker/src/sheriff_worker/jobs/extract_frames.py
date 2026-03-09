from __future__ import annotations

from sheriff_api.services.video_frames import extract_video_sequence_job


def run(payload: dict) -> dict:
    if "video_storage_uri" in payload:
        raise ValueError("The real extract_frames handler is async; use run_async")
    if "video_uri" not in payload:
        raise ValueError("video_uri is required")
    return {"status": "done", "frames_extracted": payload.get("fps", 1)}


async def run_async(payload: dict) -> dict:
    return await extract_video_sequence_job(payload)
