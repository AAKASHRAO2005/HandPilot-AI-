"""
gestures/movement.py — Hand movement and velocity tracker with low-pass velocity smoothing.

Tracks palm/fingertip position history and computes instantaneous velocity
for scroll control and swipe detection.
"""

import time
from collections import deque
from typing import Optional, Tuple

from utils.smoothing import PointOneEuroFilter
from utils.logger import setup_logger

log = setup_logger(__name__)


class MovementTracker:
    """
    Tracks normalized hand position over time and computes smooth velocity.

    Args:
        history_size: Number of recent positions to average velocity over.
        decay: Velocity decay factor per frame.
    """

    def __init__(self, history_size: int = 8, decay: float = 0.85) -> None:
        self._history_size = history_size
        self._positions: deque = deque(maxlen=history_size)
        self._timestamps: deque = deque(maxlen=history_size)
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._decay = decay
        self._vel_filter = PointOneEuroFilter(min_cutoff=1.5, beta=0.1)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Add a new position and compute smoothed velocity.

        Args:
            x, y: Normalized hand position [0, 1].

        Returns:
            (vx, vy) — velocity in normalized units/second.
        """
        now = time.perf_counter()
        self._positions.append((x, y))
        self._timestamps.append(now)

        if len(self._positions) < 2:
            return (0.0, 0.0)

        dt = self._timestamps[-1] - self._timestamps[0]
        if dt < 1e-6:
            return (self._vx, self._vy)

        px0, py0 = self._positions[0]
        px1, py1 = self._positions[-1]
        raw_vx = (px1 - px0) / dt
        raw_vy = (py1 - py0) / dt

        # Apply smooth velocity filter
        self._vx, self._vy = self._vel_filter.filter(raw_vx, raw_vy, now)
        return (self._vx, self._vy)

    def reset(self) -> None:
        self._positions.clear()
        self._timestamps.clear()
        self._vx = 0.0
        self._vy = 0.0
        self._vel_filter.reset()

    @property
    def velocity(self) -> Tuple[float, float]:
        """Most recent (vx, vy) velocity."""
        return (self._vx, self._vy)

    @property
    def current_pos(self) -> Optional[Tuple[float, float]]:
        """Most recent position or None."""
        if self._positions:
            return self._positions[-1]
        return None

    @property
    def prev_pos(self) -> Optional[Tuple[float, float]]:
        """Position one frame ago, or None."""
        if len(self._positions) >= 2:
            return self._positions[-2]
        return None

    @property
    def delta(self) -> Tuple[float, float]:
        """
        Position change between last two frames.
        Useful for per-frame scroll amounts.
        """
        if len(self._positions) >= 2:
            return (
                self._positions[-1][0] - self._positions[-2][0],
                self._positions[-1][1] - self._positions[-2][1],
            )
        return (0.0, 0.0)
