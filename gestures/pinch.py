"""
gestures/pinch.py — Pinch gesture detectors for index, middle, and double-click.
"""

import time
from typing import Optional

from vision.landmarks import pinch_distance
from utils.logger import setup_logger

log = setup_logger(__name__)


class PinchDetector:
    """
    Detects a pinch between two specific fingers.

    State machine:
        OPEN → PINCH_START → PINCH_HOLD → PINCH_RELEASE → OPEN

    Args:
        finger_a: First finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
        finger_b: Second finger index.
        threshold: Normalized distance to consider as pinched.
        name: Human-readable name for logging.
    """

    STATE_OPEN = "open"
    STATE_PINCHED = "pinched"

    def __init__(
        self,
        finger_a: int = 0,
        finger_b: int = 1,
        threshold: float = 0.06,
        name: str = "pinch",
    ) -> None:
        self.finger_a = finger_a
        self.finger_b = finger_b
        self.threshold = threshold
        self.name = name

        self._state = self.STATE_OPEN
        self._pinch_start_time: float = 0.0

    def update(self, landmarks: list) -> dict:
        """
        Process one frame of landmarks.

        Returns:
            dict with keys:
                'distance': float — current pinch distance
                'is_pinched': bool
                'event': None | 'pinch_start' | 'pinch_hold' | 'pinch_release'
                'hold_duration': float — seconds pinched
        """
        dist = pinch_distance(landmarks, self.finger_a, self.finger_b)
        is_pinched = dist < self.threshold
        event = None
        hold_duration = 0.0

        if is_pinched and self._state == self.STATE_OPEN:
            self._state = self.STATE_PINCHED
            self._pinch_start_time = time.perf_counter()
            event = "pinch_start"
            log.debug(f"{self.name}: PINCH START dist={dist:.4f}")

        elif is_pinched and self._state == self.STATE_PINCHED:
            hold_duration = time.perf_counter() - self._pinch_start_time
            event = "pinch_hold"

        elif not is_pinched and self._state == self.STATE_PINCHED:
            self._state = self.STATE_OPEN
            hold_duration = time.perf_counter() - self._pinch_start_time
            event = "pinch_release"
            log.debug(f"{self.name}: PINCH RELEASE held={hold_duration:.3f}s")

        return {
            "distance": dist,
            "is_pinched": is_pinched,
            "event": event,
            "hold_duration": hold_duration,
        }

    def reset(self) -> None:
        self._state = self.STATE_OPEN
        self._pinch_start_time = 0.0

    @property
    def is_active(self) -> bool:
        return self._state == self.STATE_PINCHED


class DoublePinchDetector:
    """
    Detects a double-pinch (two rapid consecutive pinches).

    Wraps a PinchDetector and watches for two pinch_start events within *window_ms*.
    """

    def __init__(
        self,
        pinch_detector: PinchDetector,
        window_ms: float = 500.0,
    ) -> None:
        self._pinch = pinch_detector
        self._window_ms = window_ms
        self._last_pinch_time: float = 0.0
        self._waiting_for_second: bool = False

    def update(self, landmarks: list) -> dict:
        """
        Returns the pinch detector result plus a 'double_click' key.
        """
        result = self._pinch.update(landmarks)
        result["double_click"] = False

        if result["event"] == "pinch_start":
            now = time.perf_counter()
            elapsed_ms = (now - self._last_pinch_time) * 1000
            if self._waiting_for_second and elapsed_ms < self._window_ms:
                result["double_click"] = True
                self._waiting_for_second = False
                log.debug(f"DOUBLE CLICK detected (gap={elapsed_ms:.0f}ms)")
            else:
                self._waiting_for_second = True
                self._last_pinch_time = now

        elif result["event"] == "pinch_release":
            # If second pinch doesn't come, clear after window expires
            now = time.perf_counter()
            if self._waiting_for_second:
                if (now - self._last_pinch_time) * 1000 > self._window_ms:
                    self._waiting_for_second = False

        return result
