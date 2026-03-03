from __future__ import annotations

import threading
from pathlib import Path

from pixel_sheriff_trainer.utils.time import utc_now_iso


class RunLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        line = f"[{utc_now_iso()}] {message.rstrip()}\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
