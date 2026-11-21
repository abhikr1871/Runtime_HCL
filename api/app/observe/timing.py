"""Per-step timings for the trace and the latency report."""
from __future__ import annotations

import time
from contextlib import contextmanager


class Timer:
    def __init__(self) -> None:
        self.ms: dict[str, int] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0) + int((time.perf_counter() - t) * 1000)

    def total(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)
