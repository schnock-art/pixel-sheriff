from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sheriff_api.config import get_settings
from sheriff_api.services.inference_client import InferenceClient


@dataclass
class DetectionResult:
    label_text: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    raw: dict[str, Any]


class PrelabelAdapter(Protocol):
    name: str

    async def warmup(self) -> None:
        ...

    async def detect(
        self,
        *,
        asset_storage_uri: str,
        prompts: list[str],
        threshold: float,
        max_detections: int,
    ) -> list[DetectionResult]:
        ...


class ActiveDeploymentPrelabelAdapter:
    def __init__(
        self,
        *,
        deployment: dict[str, Any],
        metadata_relpath: str,
        onnx_relpath: str,
        model_key: str | None,
    ) -> None:
        settings = get_settings()
        self.name = str(deployment.get("name") or "active-deployment")
        self._deployment = deployment
        self._metadata_relpath = metadata_relpath
        self._onnx_relpath = onnx_relpath
        self._model_key = model_key
        self._client = InferenceClient(
            base_url=settings.trainer_inference_base_url,
            timeout_seconds=float(settings.trainer_inference_timeout_seconds),
        )

    async def warmup(self) -> None:
        await self._client.warmup_detection(
            {
                "onnx_relpath": self._onnx_relpath,
                "metadata_relpath": self._metadata_relpath,
                "device_preference": str(self._deployment.get("device_preference") or "auto"),
                "model_key": self._model_key,
            }
        )

    async def detect(
        self,
        *,
        asset_storage_uri: str,
        prompts: list[str],
        threshold: float,
        max_detections: int,
    ) -> list[DetectionResult]:
        response = await self._client.infer_detection(
            {
                "onnx_relpath": self._onnx_relpath,
                "metadata_relpath": self._metadata_relpath,
                "asset_relpath": asset_storage_uri,
                "device_preference": str(self._deployment.get("device_preference") or "auto"),
                "score_threshold": threshold,
                "model_key": self._model_key,
            }
        )
        detections: list[DetectionResult] = []
        for row in list(response.get("boxes") or [])[:max_detections]:
            if not isinstance(row, dict):
                continue
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if not all(isinstance(value, (int, float)) for value in bbox):
                continue
            x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if width <= 0 or height <= 0:
                continue
            label_text = str(row.get("class_name") or "").strip()
            if not label_text:
                continue
            detections.append(
                DetectionResult(
                    label_text=label_text,
                    score=float(row.get("score") or 0.0),
                    bbox_xyxy=(x, y, x + width, y + height),
                    raw=row,
                )
            )
        return detections


class Florence2PrelabelAdapter:
    def __init__(self, *, model_name: str = "microsoft/Florence-2-base-ft") -> None:
        settings = get_settings()
        self.name = model_name
        self._model_name = model_name
        self._client = InferenceClient(
            base_url=settings.trainer_inference_base_url,
            timeout_seconds=float(settings.trainer_inference_timeout_seconds),
        )

    async def warmup(self) -> None:
        await self._client.warmup_florence({"model_name": self._model_name})

    async def detect(
        self,
        *,
        asset_storage_uri: str,
        prompts: list[str],
        threshold: float,
        max_detections: int,
    ) -> list[DetectionResult]:
        response = await self._client.florence_detect(
            {
                "asset_relpath": asset_storage_uri,
                "model_name": self._model_name,
                "prompts": prompts,
                "score_threshold": threshold,
                "max_detections": max_detections,
            }
        )
        detections: list[DetectionResult] = []
        for row in list(response.get("boxes") or [])[:max_detections]:
            if not isinstance(row, dict):
                continue
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if not all(isinstance(value, (int, float)) for value in bbox):
                continue
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            label_text = str(row.get("label_text") or "").strip()
            if not label_text:
                continue
            detections.append(
                DetectionResult(
                    label_text=label_text,
                    score=float(row.get("score") or 0.0),
                    bbox_xyxy=(x1, y1, x2, y2),
                    raw=row,
                )
            )
        return detections


PrelabelAdapterFactory = Callable[..., PrelabelAdapter]


PRELABEL_ADAPTER_REGISTRY: dict[str, PrelabelAdapterFactory] = {
    "active_deployment": ActiveDeploymentPrelabelAdapter,
    "florence2": Florence2PrelabelAdapter,
}
