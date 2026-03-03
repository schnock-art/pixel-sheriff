from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - exercised only in environments missing ORT
    ort = None  # type: ignore[assignment]


class CacheBusyError(RuntimeError):
    pass


@dataclass
class CacheEntry:
    key: tuple[str, str]
    model_key: str
    device_selected: str
    onnx_path: str
    session: ort.InferenceSession
    last_used: float
    in_use: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SessionCache:
    def __init__(
        self,
        *,
        max_models_gpu: int,
        max_models_cpu: int,
        ttl_seconds: int,
    ) -> None:
        self._max_models_gpu = max(1, int(max_models_gpu))
        self._max_models_cpu = max(1, int(max_models_cpu))
        self._ttl_seconds = max(1, int(ttl_seconds))

        self._entries: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()
        self._global_lock = asyncio.Lock()
        self._key_locks: dict[tuple[str, str], asyncio.Lock] = {}
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        self._providers = set(ort.get_available_providers())

    def _provider_available(self, provider: str) -> bool:
        return provider in self._providers

    def resolve_device(self, device_preference: str) -> str:
        normalized = str(device_preference or "auto").strip().lower()
        cuda_available = self._provider_available("CUDAExecutionProvider")
        if normalized == "cuda":
            return "cuda" if cuda_available else "cpu"
        if normalized == "cpu":
            return "cpu"
        return "cuda" if cuda_available else "cpu"

    def _capacity_for_device(self, device_selected: str) -> int:
        if device_selected == "cuda":
            return self._max_models_gpu
        return self._max_models_cpu

    def _active_count_for_device(self, device_selected: str) -> int:
        return sum(1 for entry in self._entries.values() if entry.device_selected == device_selected)

    def _providers_for_device(self, device_selected: str) -> list[str]:
        if device_selected == "cuda" and self._provider_available("CUDAExecutionProvider"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _key(self, model_key: str, device_selected: str) -> tuple[str, str]:
        return (model_key, device_selected)

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._key_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._key_locks[key] = lock
        return lock

    def _expired(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.last_used) >= self._ttl_seconds

    def _evict_expired(self, now: float) -> None:
        for key in list(self._entries.keys()):
            entry = self._entries.get(key)
            if entry is None:
                continue
            if entry.in_use > 0:
                continue
            if self._expired(entry, now):
                self._entries.pop(key, None)

    def _evict_lru_until_capacity(self, device_selected: str) -> bool:
        capacity = self._capacity_for_device(device_selected)
        while self._active_count_for_device(device_selected) > capacity:
            candidate_key = None
            for key, entry in self._entries.items():
                if entry.device_selected != device_selected:
                    continue
                if entry.in_use > 0:
                    continue
                candidate_key = key
                break
            if candidate_key is None:
                return False
            self._entries.pop(candidate_key, None)
        return True

    async def _touch(self, key: tuple[str, str]) -> CacheEntry | None:
        async with self._global_lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            now = time.time()
            self._evict_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.last_used = now
            self._entries.move_to_end(key)
            return entry

    async def release(self, model_key: str, device_selected: str) -> None:
        key = self._key(model_key, device_selected)
        async with self._global_lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.in_use = max(0, int(entry.in_use) - 1)
            entry.last_used = time.time()
            self._entries.move_to_end(key)

    async def acquire_session(
        self,
        *,
        model_key: str,
        onnx_path: Path,
        device_preference: str,
    ) -> tuple[ort.InferenceSession, str]:
        desired_device = self.resolve_device(device_preference)
        key = self._key(model_key, desired_device)
        lock = self._lock_for(key)
        async with lock:
            entry = await self._touch(key)
            if entry is not None:
                async with self._global_lock:
                    hit = self._entries.get(key)
                    if hit is not None:
                        hit.in_use += 1
                        return hit.session, hit.device_selected

            return await self._create_and_acquire(
                model_key=model_key,
                onnx_path=onnx_path,
                desired_device=desired_device,
                allow_cpu_fallback=True,
            )

    async def _create_and_acquire(
        self,
        *,
        model_key: str,
        onnx_path: Path,
        desired_device: str,
        allow_cpu_fallback: bool,
    ) -> tuple[ort.InferenceSession, str]:
        device_selected = desired_device
        providers = self._providers_for_device(device_selected)
        try:
            session = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception:
            if device_selected == "cuda":
                device_selected = "cpu"
                providers = self._providers_for_device("cpu")
                session = ort.InferenceSession(str(onnx_path), providers=providers)
            else:
                raise

        key = self._key(model_key, device_selected)
        fallback_to_cpu = False
        async with self._global_lock:
            now = time.time()
            self._evict_expired(now)
            existing = self._entries.get(key)
            if existing is not None:
                existing.in_use += 1
                existing.last_used = now
                self._entries.move_to_end(key)
                return existing.session, existing.device_selected

            entry = CacheEntry(
                key=key,
                model_key=model_key,
                device_selected=device_selected,
                onnx_path=str(onnx_path),
                session=session,
                last_used=now,
                in_use=1,
            )
            self._entries[key] = entry
            self._entries.move_to_end(key)
            if not self._evict_lru_until_capacity(device_selected):
                self._entries.pop(key, None)
                if device_selected == "cuda" and allow_cpu_fallback:
                    fallback_to_cpu = True
                else:
                    raise CacheBusyError("cache_busy")
            else:
                return entry.session, entry.device_selected
        if fallback_to_cpu:
            return await self._create_and_acquire(
                model_key=model_key,
                onnx_path=onnx_path,
                desired_device="cpu",
                allow_cpu_fallback=False,
            )
        raise CacheBusyError("cache_busy")
