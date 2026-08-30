"""
utils/fps.py — Real-time FPS and latency tracker.
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    Tracks frames-per-second using a rolling window.

    Usage::

        fps = FPSCounter(window=30)
        while running:
            fps.tick()
            print(fps.get())
    """

    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        """Record a new frame timestamp."""
        self._times.append(time.perf_counter())

    def get(self) -> float:
        """Return current FPS. Returns 0.0 if not enough data."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed


class LatencyTracker:
    """
    Tracks end-to-end processing latency per frame.

    Usage::

        lat = LatencyTracker()
        lat.start()
        # ... processing ...
        ms = lat.stop()   # returns milliseconds
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._last_ms: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        self._last_ms = (time.perf_counter() - self._start) * 1000.0
        self._start = None
        return self._last_ms

    @property
    def last_ms(self) -> float:
        return self._last_ms
