from __future__ import annotations

from sheriff_api.services.prelabels import process_prelabel_asset_job


def run(payload: dict) -> dict:
    raise ValueError("The real prelabel_asset handler is async; use run_async")


async def run_async(payload: dict) -> dict:
    return await process_prelabel_asset_job(payload)

