from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from pixel_sheriff_trainer.inference.tensorrt_engine import load_engine_bundle


class TensorRTCacheBusyError(RuntimeError):
    pass


@dataclass
class TensorRTCacheEntry:
    key: tuple[str, str]
    engine_key: str
    device_selected: str
    engine_path: str
    bundle: Any
    last_used: float
    in_use: int = 0


class TensorRTEngineCache:
    def __init__(
        self,
        *,
        max_engines_gpu: int = 1,
        ttl_seconds: int = 600,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._max_engines_gpu = max(1, int(max_engines_gpu))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._clock = clock or time.monotonic
        self._entries: "OrderedDict[tuple[str, str], TensorRTCacheEntry]" = OrderedDict()
        self._global_lock = asyncio.Lock()
        self._key_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _key(self, engine_key: str, device_selected: str) -> tuple[str, str]:
        return (engine_key, device_selected)

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._key_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._key_locks[key] = lock
        return lock

    def _active_gpu_count(self) -> int:
        return sum(1 for entry in self._entries.values() if entry.device_selected == "cuda")

    def _expired(self, entry: TensorRTCacheEntry, now: float) -> bool:
        return (now - entry.last_used) >= self._ttl_seconds

    def _evict_expired(self, now: float) -> None:
        for key in list(self._entries.keys()):
            entry = self._entries.get(key)
            if entry is None or entry.in_use > 0:
                continue
            if self._expired(entry, now):
                self._entries.pop(key, None)

    def _evict_lru_until_capacity(self) -> bool:
        while self._active_gpu_count() > self._max_engines_gpu:
            candidate_key = None
            for key, entry in self._entries.items():
                if entry.device_selected != "cuda" or entry.in_use > 0:
                    continue
                candidate_key = key
                break
            if candidate_key is None:
                return False
            self._entries.pop(candidate_key, None)
        return True

    async def _touch(self, key: tuple[str, str]) -> TensorRTCacheEntry | None:
        async with self._global_lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            now = self._clock()
            self._evict_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.last_used = now
            self._entries.move_to_end(key)
            return entry

    async def release(self, engine_key: str, device_selected: str) -> None:
        key = self._key(engine_key, device_selected)
        async with self._global_lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.in_use = max(0, int(entry.in_use) - 1)
            entry.last_used = self._clock()
            self._entries.move_to_end(key)

    async def acquire_engine(
        self,
        *,
        engine_key: str,
        engine_path: Path,
        device_preference: str,
    ) -> tuple[Any, str]:
        normalized = str(device_preference or "cuda").strip().lower()
        if normalized == "cpu":
            raise RuntimeError("TensorRT detection requires CUDA")
        desired_device = "cuda"
        key = self._key(engine_key, desired_device)
        lock = self._lock_for(key)
        async with lock:
            entry = await self._touch(key)
            if entry is not None:
                async with self._global_lock:
                    hit = self._entries.get(key)
                    if hit is not None:
                        hit.in_use += 1
                        return hit.bundle, hit.device_selected

            bundle = load_engine_bundle(engine_path)
            async with self._global_lock:
                now = self._clock()
                self._evict_expired(now)
                existing = self._entries.get(key)
                if existing is not None:
                    existing.in_use += 1
                    existing.last_used = now
                    self._entries.move_to_end(key)
                    return existing.bundle, existing.device_selected

                entry = TensorRTCacheEntry(
                    key=key,
                    engine_key=engine_key,
                    device_selected=desired_device,
                    engine_path=str(engine_path),
                    bundle=bundle,
                    last_used=now,
                    in_use=1,
                )
                self._entries[key] = entry
                self._entries.move_to_end(key)
                if not self._evict_lru_until_capacity():
                    self._entries.pop(key, None)
                    raise TensorRTCacheBusyError("TensorRT cache is busy")
                return bundle, desired_device
