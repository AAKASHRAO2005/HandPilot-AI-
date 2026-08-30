"""
gestures/swipe.py — Swipe gesture detector for keyboard shortcut triggers.

A swipe is detected when the hand moves consistently in one direction
above a velocity threshold within a short time window.
"""

import time
from collections import deque
from typing import Optional

from utils.logger import setup_logger

log = setup_logger(__name__)


class SwipeDetector:
    """
    Detects left/right/up/down swipes from hand velocity history.

    Args:
        velocity_threshold: Minimum normalized velocity to trigger a swipe.
        window_frames: Number of frames the velocity must be sustained.
        cooldown_ms: Minimum ms between consecutive swipe detections.
    """

    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"

    def __init__(
        self,
        velocity_threshold: float = 0.05,
        window_frames: int = 8,
        cooldown_ms: float = 800.0,
    ) -> None:
        self._threshold = velocity_threshold
        self._window = window_frames
        self._cooldown_ms = cooldown_ms

        self._vx_history: deque = deque(maxlen=window_frames)
        self._vy_history: deque = deque(maxlen=window_frames)
        self._last_swipe_time: float = 0.0
        self._last_swipe: Optional[str] = None

    def update(self, vx: float, vy: float) -> Optional[str]:
        """
        Feed velocity and return swipe direction if detected.

        Args:
            vx: Horizontal velocity (positive = right).
            vy: Vertical velocity (positive = down).

        Returns:
            Swipe direction string or None.
        """
        self._vx_history.append(vx)
        self._vy_history.append(vy)

        if len(self._vx_history) < self._window:
            return None

        now = time.perf_counter()
        if (now - self._last_swipe_time) * 1000 < self._cooldown_ms:
            return None

        avg_vx = sum(self._vx_history) / len(self._vx_history)
        avg_vy = sum(self._vy_history) / len(self._vy_history)

        swipe = None
        if abs(avg_vx) > abs(avg_vy):  # Horizontal swipe dominates
            if avg_vx > self._threshold:
                swipe = self.SWIPE_RIGHT
            elif avg_vx < -self._threshold:
                swipe = self.SWIPE_LEFT
        else:  # Vertical swipe dominates
            if avg_vy > self._threshold:
                swipe = self.SWIPE_DOWN
            elif avg_vy < -self._threshold:
                swipe = self.SWIPE_UP

        if swipe:
            log.info(f"SWIPE detected: {swipe} vx={avg_vx:.3f} vy={avg_vy:.3f}")
            self._last_swipe_time = now
            self._last_swipe = swipe
            # Clear history to prevent repeated triggers
            self._vx_history.clear()
            self._vy_history.clear()

        return swipe

    def reset(self) -> None:
        self._vx_history.clear()
        self._vy_history.clear()
