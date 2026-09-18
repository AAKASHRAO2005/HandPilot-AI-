"""
gestures/pinch.py — Precision pinch gesture detectors with scale-invariance and hysteresis.
"""

import time
from typing import Optional, List

from vision.landmarks import pinch_distance, normalized_pinch_distance
from utils.logger import setup_logger

log = setup_logger(__name__)


class PinchDetector:
    """
    Detects a pinch between two fingers with scale-invariance and hysteresis.

    Hysteresis:
        Enters PINCHED state when distance < threshold_on.
        Exits to OPEN state only when distance > threshold_off (where threshold_off > threshold_on).
        This eliminates rapid flickering/chatter at border distances.

    Args:
        finger_a: First finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
        finger_b: Second finger index.
        threshold: Trigger threshold (threshold_on).
        threshold_off: Release threshold (defaults to 1.35 * threshold).
        scale_invariant: If True, uses normalized hand scale distance.
        name: Human-readable name for logging.
    """

    STATE_OPEN = "open"
    STATE_PINCHED = "pinched"

    def __init__(
        self,
        finger_a: int = 0,
        finger_b: int = 1,
        threshold: float = 0.06,
        threshold_off: Optional[float] = None,
        scale_invariant: bool = False,
        name: str = "pinch",
    ) -> None:
        self.finger_a = finger_a
        self.finger_b = finger_b
        self.threshold_on = float(threshold)
        self.threshold_off = float(threshold_off) if threshold_off is not None else float(threshold * 1.35)
        self.scale_invariant = scale_invariant
        self.name = name

        self._state = self.STATE_OPEN
        self._pinch_start_time: float = 0.0
        self._last_dist: Optional[float] = None
        self._closing_speed: float = 0.0

    @property
    def threshold(self) -> float:
        return self.threshold_on

    @threshold.setter
    def threshold(self, val: float) -> None:
        self.threshold_on = float(val)
        self.threshold_off = float(val * 1.35)

    def update(self, landmarks: list) -> dict:
        """
        Process one frame of landmarks.

        Returns:
            dict with keys:
                'distance': float — current pinch distance
                'is_pinched': bool
                'event': None | 'pinch_start' | 'pinch_hold' | 'pinch_release'
                'hold_duration': float — seconds pinched
                'is_closing': bool — whether fingers are actively approaching
        """
        if self.scale_invariant:
            dist = normalized_pinch_distance(landmarks, self.finger_a, self.finger_b)
        else:
            dist = pinch_distance(landmarks, self.finger_a, self.finger_b)

        now = time.perf_counter()
        if self._last_dist is not None:
            self._closing_speed = self._last_dist - dist
        self._last_dist = dist

        event = None
        hold_duration = 0.0
        is_closing = self._closing_speed > 0.005 and dist < (self.threshold_on * 1.6)

        if self._state == self.STATE_OPEN:
            if dist < self.threshold_on:
                self._state = self.STATE_PINCHED
                self._pinch_start_time = now
                event = "pinch_start"
                log.debug(f"{self.name}: PINCH START dist={dist:.4f}")
        else:
            # STATE_PINCHED — require distance to exceed threshold_off to release
            if dist > self.threshold_off:
                self._state = self.STATE_OPEN
                hold_duration = now - self._pinch_start_time
                event = "pinch_release"
                log.debug(f"{self.name}: PINCH RELEASE held={hold_duration:.3f}s")
            else:
                hold_duration = now - self._pinch_start_time
                event = "pinch_hold"

        return {
            "distance": dist,
            "is_pinched": self._state == self.STATE_PINCHED,
            "event": event,
            "hold_duration": hold_duration,
            "is_closing": is_closing,
        }

    def reset(self) -> None:
        self._state = self.STATE_OPEN
        self._pinch_start_time = 0.0
        self._last_dist = None
        self._closing_speed = 0.0

    @property
    def is_active(self) -> bool:
        return self._state == self.STATE_PINCHED


class DoublePinchDetector:
    """
    Detects a double-pinch (two rapid consecutive pinches within window_ms).
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

    @property
    def threshold(self) -> float:
        return self._pinch.threshold

    @threshold.setter
    def threshold(self, val: float) -> None:
        self._pinch.threshold = val

    def update(self, landmarks: list) -> dict:
        result = self._pinch.update(landmarks)
        result["double_click"] = False

        if result["event"] == "pinch_start":
            now = time.perf_counter()
            elapsed_ms = (now - self._last_pinch_time) * 1000.0
            if self._waiting_for_second and elapsed_ms < self._window_ms:
                result["double_click"] = True
                self._waiting_for_second = False
                log.debug(f"DOUBLE CLICK detected (gap={elapsed_ms:.0f}ms)")
            else:
                self._waiting_for_second = True
                self._last_pinch_time = now

        elif result["event"] == "pinch_release":
            now = time.perf_counter()
            if self._waiting_for_second:
                if (now - self._last_pinch_time) * 1000.0 > self._window_ms:
                    self._waiting_for_second = False

        return result

    def reset(self) -> None:
        self._pinch.reset()
        self._waiting_for_second = False
        self._last_pinch_time = 0.0
