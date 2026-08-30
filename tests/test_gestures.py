"""
tests/test_gestures.py — Unit tests for pinch, double-pinch, swipe, and movement tracker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from gestures.pinch import PinchDetector, DoublePinchDetector
from gestures.movement import MovementTracker
from gestures.swipe import SwipeDetector


# ---------------------------------------------------------------------------
# Mock landmark helpers
# ---------------------------------------------------------------------------

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.5, 0.5), **overrides):
    """Build a 21-landmark list."""
    # Default: all landmarks at (0.5, 0.5)
    lms = [MockLandmark(0.5, 0.5) for _ in range(21)]
    # Thumb tip = index 4, index tip = index 8
    lms[4] = MockLandmark(*thumb_tip)
    lms[8] = MockLandmark(*index_tip)
    return lms


# ---------------------------------------------------------------------------
# PinchDetector tests
# ---------------------------------------------------------------------------

class TestPinchDetector:

    def test_no_pinch_when_far(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        lms = make_landmarks(thumb_tip=(0.3, 0.5), index_tip=(0.7, 0.5))
        result = det.update(lms)
        assert not result["is_pinched"]
        assert result["event"] is None

    def test_pinch_start_when_close(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        lms = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        result = det.update(lms)
        assert result["is_pinched"]
        assert result["event"] == "pinch_start"

    def test_pinch_hold_on_second_frame(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        lms = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        det.update(lms)  # pinch_start
        result = det.update(lms)  # pinch_hold
        assert result["event"] == "pinch_hold"

    def test_pinch_release(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        far = make_landmarks(thumb_tip=(0.3, 0.5), index_tip=(0.7, 0.5))
        det.update(close)  # start
        result = det.update(far)   # release
        assert result["event"] == "pinch_release"

    def test_reset_clears_state(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        det.update(close)
        det.reset()
        result = det.update(close)
        assert result["event"] == "pinch_start"  # should re-trigger


# ---------------------------------------------------------------------------
# DoublePinchDetector tests
# ---------------------------------------------------------------------------

class TestDoublePinchDetector:

    def _pinch_sequence(self, det, close_lms, far_lms):
        """One pinch: close → release."""
        det.update(close_lms)
        det.update(far_lms)

    def test_single_pinch_no_double(self):
        base_det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        det = DoublePinchDetector(base_det, window_ms=500.0)
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        far = make_landmarks(thumb_tip=(0.3, 0.5), index_tip=(0.7, 0.5))
        self._pinch_sequence(det, close, far)
        # No double click after single pinch
        result = det.update(far)
        assert not result.get("double_click", False)

    def test_double_pinch_detected(self):
        base_det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        det = DoublePinchDetector(base_det, window_ms=500.0)
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        far = make_landmarks(thumb_tip=(0.3, 0.5), index_tip=(0.7, 0.5))
        # First pinch
        self._pinch_sequence(det, close, far)
        # Second pinch (immediately)
        result = det.update(close)
        assert result.get("double_click", False)


# ---------------------------------------------------------------------------
# MovementTracker tests
# ---------------------------------------------------------------------------

class TestMovementTracker:

    def test_initial_velocity_zero(self):
        t = MovementTracker()
        vx, vy = t.update(0.5, 0.5)
        assert vx == 0.0
        assert vy == 0.0

    def test_rightward_movement_positive_vx(self):
        t = MovementTracker(history_size=2)
        t.update(0.0, 0.5)
        time.sleep(0.01)
        vx, vy = t.update(0.1, 0.5)
        assert vx > 0

    def test_upward_movement_negative_vy(self):
        t = MovementTracker(history_size=2)
        t.update(0.5, 0.5)
        time.sleep(0.01)
        vx, vy = t.update(0.5, 0.3)
        assert vy < 0

    def test_reset_clears_history(self):
        t = MovementTracker()
        t.update(0.5, 0.5)
        t.reset()
        assert t.current_pos is None
        assert t.velocity == (0.0, 0.0)


# ---------------------------------------------------------------------------
# SwipeDetector tests
# ---------------------------------------------------------------------------

class TestSwipeDetector:

    def test_no_swipe_below_threshold(self):
        det = SwipeDetector(velocity_threshold=0.05, window_frames=3)
        for _ in range(5):
            result = det.update(0.01, 0.0)
        assert result is None

    def test_rightward_swipe_detected(self):
        det = SwipeDetector(velocity_threshold=0.05, window_frames=3, cooldown_ms=0)
        result = None
        for _ in range(3):
            result = det.update(0.1, 0.0)
        assert result == SwipeDetector.SWIPE_RIGHT

    def test_leftward_swipe_detected(self):
        det = SwipeDetector(velocity_threshold=0.05, window_frames=3, cooldown_ms=0)
        result = None
        for _ in range(3):
            result = det.update(-0.1, 0.0)
        assert result == SwipeDetector.SWIPE_LEFT

    def test_downward_swipe_detected(self):
        det = SwipeDetector(velocity_threshold=0.05, window_frames=3, cooldown_ms=0)
        result = None
        for _ in range(3):
            result = det.update(0.0, 0.1)
        assert result == SwipeDetector.SWIPE_DOWN
