from __future__ import annotations
from pathlib import Path

import pytest

import pixel_sheriff_trainer.inference.app as inference_app_module
import pixel_sheriff_trainer.inference.session_cache as session_cache_module
from pixel_sheriff_trainer.inference.schemas import InferDetectionWarmupRequest


@pytest.mark.asyncio
async def test_detection_warmup_endpoint_returns_selected_device(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "storage"
    onnx_path = storage_root / "models" / "demo.onnx"
    metadata_path = storage_root / "models" / "demo.metadata.json"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"fake-onnx")
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    monkeypatch.setattr(inference_app_module, "sha256_file", lambda _path: "model-key")
    monkeypatch.setattr(
        session_cache_module,
        "ort",
        type(
            "_DummyOrt",
            (),
            {
                "get_available_providers": staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]),
                "InferenceSession": object,
            },
        ),
    )

    async def _acquire_session(self, *, model_key: str, onnx_path: Path, device_preference: str):
        assert model_key == "model-key"
        assert onnx_path.name == "demo.onnx"
        assert device_preference == "auto"
        return object(), "cuda"

    async def _release(self, model_key: str, device_selected: str) -> None:
        assert model_key == "model-key"
        assert device_selected == "cuda"

    monkeypatch.setattr(inference_app_module.SessionCache, "acquire_session", _acquire_session)
    monkeypatch.setattr(inference_app_module.SessionCache, "release", _release)

    app = inference_app_module.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/infer/detection/warmup")
    payload = InferDetectionWarmupRequest(
        onnx_relpath="models/demo.onnx",
        metadata_relpath="models/demo.metadata.json",
        device_preference="auto",
        model_key="model-key",
    )

    response = await route.endpoint(payload)
    assert response.model_dump() == {"device_selected": "cuda", "warmed": True}
