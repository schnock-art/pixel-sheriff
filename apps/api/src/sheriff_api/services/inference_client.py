from __future__ import annotations

from typing import Any

import httpx


class InferenceClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)

    async def infer_classification(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/infer/classification", json=payload)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}

    async def warmup_classification(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/infer/classification/warmup", json=payload)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}
