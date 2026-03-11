from __future__ import annotations

from typing import Any

import httpx

_TASK_INFER_ENDPOINT: dict[str, str] = {
    "classification": "/infer/classification",
    "bbox": "/infer/detection",
    "segmentation": "/infer/segmentation",
}

_TASK_WARMUP_ENDPOINT: dict[str, str] = {
    "classification": "/infer/classification/warmup",
    "bbox": "/infer/detection/warmup",
}


class InferenceClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)

    async def infer(self, task_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Route inference request by task kind.

        task_kind: "classification" | "bbox" | "segmentation"
        """
        endpoint = _TASK_INFER_ENDPOINT.get(task_kind)
        if endpoint is None:
            raise ValueError(f"Unsupported task kind for inference: {task_kind!r}")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}{endpoint}", json=payload)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}

    async def infer_classification(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.infer("classification", payload)

    async def infer_detection(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.infer("bbox", payload)

    async def infer_segmentation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.infer("segmentation", payload)

    async def florence_detect(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/infer/florence/detect", json=payload)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}

    async def warmup_florence(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/infer/florence/warmup", json=payload)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}

    async def warmup(self, task_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = _TASK_WARMUP_ENDPOINT.get(task_kind)
        if endpoint is None:
            raise ValueError(f"Unsupported task kind for warmup: {task_kind!r}")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}{endpoint}", json=payload)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}

    async def warmup_classification(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.warmup("classification", payload)

    async def warmup_detection(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.warmup("bbox", payload)
