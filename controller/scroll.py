"""
controller/scroll.py — Scroll wheel control with sub-tick accumulation and momentum.
"""

import time
from typing import Optional

from controller import windows_input as wi
from utils.logger import setup_logger

log = setup_logger(__name__)


class ScrollController:
    """
    Converts vertical/horizontal hand velocity to smooth mouse wheel scroll events
    using sub-tick floating point accumulation.

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
        self._min_scroll_interval: float = 0.02

        # Sub-tick accumulators
        self._accum_y: float = 0.0
        self._accum_x: float = 0.0

    def scroll_vertical(self, velocity_y: float) -> bool:
        """
        Send vertical scroll based on y-axis velocity with sub-tick accumulator.

        Args:
            velocity_y: Positive = hand moving down (scroll down).
                        Negative = hand moving up   (scroll up).

        Returns:
            True if scroll was sent.
        """
        if abs(velocity_y) < self.min_velocity:
            self._accum_y *= 0.5  # Decay accumulator
            return False

        now = time.perf_counter()
        if (now - self._last_scroll_time) < self._min_scroll_interval:
            return False

        # Negative ticks = scroll down
        delta_ticks = -velocity_y * self.velocity_scale * self.speed_multiplier
        delta_ticks = max(-8.0, min(8.0, delta_ticks))
        self._accum_y += delta_ticks

        if abs(self._accum_y) >= 1.0:
            ticks_to_send = int(self._accum_y)
            self._accum_y -= float(ticks_to_send)
            wi.scroll_vertical(ticks_to_send)
            self._last_scroll_time = now
            log.debug(f"SCROLL vertical ticks={ticks_to_send} (accum={self._accum_y:.2f})")
            return True

        return False

    def scroll_horizontal(self, velocity_x: float) -> bool:
        """
        Send horizontal scroll based on x-axis velocity with sub-tick accumulator.

        Args:
            velocity_x: Positive = hand moving right (scroll right).

        Returns:
            True if scroll was sent.
        """
        if abs(velocity_x) < self.min_velocity:
            self._accum_x *= 0.5
            return False

        now = time.perf_counter()
        if (now - self._last_scroll_time) < self._min_scroll_interval:
            return False

        delta_ticks = velocity_x * self.velocity_scale * self.speed_multiplier
        delta_ticks = max(-8.0, min(8.0, delta_ticks))
        self._accum_x += delta_ticks

        if abs(self._accum_x) >= 1.0:
            ticks_to_send = int(self._accum_x)
            self._accum_x -= float(ticks_to_send)
            wi.scroll_horizontal(ticks_to_send)
            self._last_scroll_time = now
            log.debug(f"SCROLL horizontal ticks={ticks_to_send}")
            return True

        return False

    def reset(self) -> None:
        self._accum_y = 0.0
        self._accum_x = 0.0

    def update_config(
        self,
        speed_multiplier: Optional[float] = None,
        velocity_scale: Optional[float] = None,
    ) -> None:
        if speed_multiplier is not None:
            self.speed_multiplier = max(0.1, speed_multiplier)
        if velocity_scale is not None:
            self.velocity_scale = max(1.0, velocity_scale)
