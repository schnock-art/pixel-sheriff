from __future__ import annotations

from pathlib import Path

import pytest

from pixel_sheriff_trainer.inference.session_cache import CacheBusyError, SessionCache
import pixel_sheriff_trainer.inference.session_cache as session_cache_module


class _DummyInput:
    name = "input"


class _DummySession:
    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        self.model_path = model_path
        self.providers = providers or []

    def get_inputs(self) -> list[_DummyInput]:
        return [_DummyInput()]

    def run(self, *_args, **_kwargs):
        return []


@pytest.fixture(autouse=True)
def _patch_ort(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyOrt:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        InferenceSession = _DummySession

    monkeypatch.setattr(session_cache_module, "ort", _DummyOrt)


@pytest.mark.asyncio
async def test_cache_keys_by_model_key_and_selected_device() -> None:
    cache = SessionCache(max_models_gpu=1, max_models_cpu=1, ttl_seconds=600)
    session_a, device_a = await cache.acquire_session(model_key="m1", onnx_path=Path("/tmp/m1.onnx"), device_preference="auto")
    assert device_a == "cuda"
    await cache.release("m1", device_a)

    session_b, device_b = await cache.acquire_session(model_key="m1", onnx_path=Path("/tmp/m1.onnx"), device_preference="cuda")
    assert device_b == "cuda"
    await cache.release("m1", device_b)
    assert session_a is session_b


@pytest.mark.asyncio
async def test_cache_falls_back_to_cpu_when_gpu_capacity_is_leased() -> None:
    cache = SessionCache(max_models_gpu=1, max_models_cpu=1, ttl_seconds=600)
    _session_a, device_a = await cache.acquire_session(model_key="m1", onnx_path=Path("/tmp/m1.onnx"), device_preference="cuda")
    assert device_a == "cuda"

    _session_b, device_b = await cache.acquire_session(model_key="m2", onnx_path=Path("/tmp/m2.onnx"), device_preference="cuda")
    assert device_b == "cpu"

    await cache.release("m2", device_b)
    await cache.release("m1", device_a)


@pytest.mark.asyncio
async def test_cache_busy_when_cpu_capacity_is_leased() -> None:
    cache = SessionCache(max_models_gpu=1, max_models_cpu=1, ttl_seconds=600)
    _session_a, device_a = await cache.acquire_session(model_key="m1", onnx_path=Path("/tmp/m1.onnx"), device_preference="cpu")
    assert device_a == "cpu"
    with pytest.raises(CacheBusyError):
        await cache.acquire_session(model_key="m2", onnx_path=Path("/tmp/m2.onnx"), device_preference="cpu")
    await cache.release("m1", device_a)


@pytest.mark.asyncio
async def test_cache_ttl_eviction_reloads_session() -> None:
    cache = SessionCache(max_models_gpu=1, max_models_cpu=1, ttl_seconds=1)
    session_a, device_a = await cache.acquire_session(model_key="m1", onnx_path=Path("/tmp/m1.onnx"), device_preference="cpu")
    await cache.release("m1", device_a)
    import asyncio

    await asyncio.sleep(1.1)
    session_b, device_b = await cache.acquire_session(model_key="m1", onnx_path=Path("/tmp/m1.onnx"), device_preference="cpu")
    await cache.release("m1", device_b)
    assert session_a is not session_b
