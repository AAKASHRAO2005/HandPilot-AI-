"""
controller/scroll.py — Scroll wheel control with velocity-based speed.
"""

import time
from typing import Optional

from controller import windows_input as wi
from utils.logger import setup_logger

log = setup_logger(__name__)


class ScrollController:
    """
    Converts vertical/horizontal hand velocity to mouse wheel scroll events.

    Args:
        speed_multiplier: Base scroll speed multiplier.
        velocity_scale: Scroll ticks per unit velocity.
        min_velocity: Below this absolute velocity, no scroll is sent.
    """

    def __init__(
        self,
        speed_multiplier: float = 1.0,
        velocity_scale: float = 8.0,
        min_velocity: float = 0.005,
    ) -> None:
        self.speed_multiplier = speed_multiplier
        self.velocity_scale = velocity_scale
        self.min_velocity = min_velocity
        self._last_scroll_time: float = 0.0
        self._min_scroll_interval: float = 0.05  # seconds

    def scroll_vertical(self, velocity_y: float) -> bool:
        """
        Send vertical scroll based on y-axis velocity.

        Args:
            velocity_y: Positive = hand moving down (scroll down).
                        Negative = hand moving up   (scroll up).

        Returns:
            True if scroll was sent.
        """
        if abs(velocity_y) < self.min_velocity:
            return False

        now = time.perf_counter()
        if (now - self._last_scroll_time) < self._min_scroll_interval:
            return False

        # Positive velocity_y = hand moves down = scroll down (negative ticks)
        ticks = -velocity_y * self.velocity_scale * self.speed_multiplier
        ticks = max(-10, min(10, ticks))  # clamp to reasonable range

        if abs(ticks) < 0.5:
            return False

        wi.scroll_vertical(int(ticks))
        self._last_scroll_time = now
        log.debug(f"SCROLL vertical ticks={int(ticks):.1f} vel={velocity_y:.4f}")
        return True

    def scroll_horizontal(self, velocity_x: float) -> bool:
        """
        Send horizontal scroll based on x-axis velocity.

        Args:
            velocity_x: Positive = hand moving right (scroll right).

        Returns:
            True if scroll was sent.
        """
        if abs(velocity_x) < self.min_velocity:
            return False

        now = time.perf_counter()
        if (now - self._last_scroll_time) < self._min_scroll_interval:
            return False

        ticks = velocity_x * self.velocity_scale * self.speed_multiplier
        ticks = max(-10, min(10, ticks))

        if abs(ticks) < 0.5:
            return False

        wi.scroll_horizontal(int(ticks))
        self._last_scroll_time = now
        log.debug(f"SCROLL horizontal ticks={int(ticks):.1f}")
        return True

    def update_config(
        self,
        speed_multiplier: Optional[float] = None,
        velocity_scale: Optional[float] = None,
    ) -> None:
        if speed_multiplier is not None:
            self.speed_multiplier = max(0.1, speed_multiplier)
        if velocity_scale is not None:
            self.velocity_scale = max(1.0, velocity_scale)
