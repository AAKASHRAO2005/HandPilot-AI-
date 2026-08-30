"""
utils/smoothing.py — EMA (Exponential Moving Average) cursor smoother with dead zone.
"""

import math
from typing import Tuple


class EMASmoother:
    """
    Smooths a 2-D point stream using Exponential Moving Average.

    Args:
        alpha: Smoothing factor in (0, 1].
               Lower  = smoother but more lag.
               Higher = more responsive but noisier.
        dead_zone: Pixel radius. Movements smaller than this are ignored.
    """

    def __init__(self, alpha: float = 0.20, dead_zone: float = 8.0) -> None:
        self.alpha = max(0.01, min(1.0, alpha))
        self.dead_zone = dead_zone
        self._x: float | None = None
        self._y: float | None = None

    def update(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """
        Feed a new raw point and receive the smoothed position.

        Returns:
            (smoothed_x, smoothed_y)
        """
        if self._x is None:
            # First sample — initialize without smoothing
            self._x = raw_x
            self._y = raw_y
            return (raw_x, raw_y)

        # Dead-zone check
        dx = raw_x - self._x
        dy = raw_y - self._y
        if math.hypot(dx, dy) < self.dead_zone:
            return (self._x, self._y)

        # EMA update
        self._x = self.alpha * raw_x + (1.0 - self.alpha) * self._x
        self._y = self.alpha * raw_y + (1.0 - self.alpha) * self._y
        return (self._x, self._y)

    def reset(self) -> None:
        """Reset smoother state (e.g. when hand is lost)."""
        self._x = None
        self._y = None

    @property
    def current(self) -> Tuple[float, float] | None:
        if self._x is None:
            return None
        return (self._x, self._y)
