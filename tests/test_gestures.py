"""
tests/test_gestures.py — Unit tests for pinch, double-pinch, swipe, movement tracker, and scale invariance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from gestures.pinch import PinchDetector, DoublePinchDetector
from gestures.movement import MovementTracker
from gestures.swipe import SwipeDetector
from vision.landmarks import get_hand_scale, normalized_pinch_distance


# ---------------------------------------------------------------------------
# Mock landmark helpers
# ---------------------------------------------------------------------------

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.5, 0.5), wrist=(0.5, 0.9), middle_mcp=(0.5, 0.6), **overrides):
    """Build a 21-landmark list with reasonable anatomical anchors."""
    lms = [MockLandmark(0.5, 0.5) for _ in range(21)]
    lms[0] = MockLandmark(*wrist)        # WRIST
    lms[4] = MockLandmark(*thumb_tip)    # THUMB_TIP
    lms[5] = MockLandmark(0.4, 0.6)      # INDEX_MCP
    lms[8] = MockLandmark(*index_tip)    # INDEX_TIP
    lms[9] = MockLandmark(*middle_mcp)   # MIDDLE_MCP
    lms[17] = MockLandmark(0.6, 0.6)     # PINKY_MCP
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

    def test_pinch_hysteresis_release(self):
        """Pinch should stay active until distance exceeds threshold_off (1.35x threshold)."""
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        # 1. Trigger pinch at 0.05 (< 0.06)
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.55, 0.5))
        r1 = det.update(close)
        assert r1["is_pinched"]

        # 2. Move to intermediate distance 0.07 (between 0.06 and 0.081)
        mid = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.57, 0.5))
        r2 = det.update(mid)
        # Should STAY pinched due to hysteresis
        assert r2["is_pinched"]
        assert r2["event"] == "pinch_hold"

        # 3. Move beyond threshold_off (dist > 0.081)
        far = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.60, 0.5))
        r3 = det.update(far)
        assert not r3["is_pinched"]
        assert r3["event"] == "pinch_release"

    def test_scale_invariant_pinch(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.25, scale_invariant=True)
        # Close pinch relative to hand scale
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        r = det.update(close)
        assert r["is_pinched"]

    def test_reset_clears_state(self):
        det = PinchDetector(finger_a=0, finger_b=1, threshold=0.06)
        close = make_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.52, 0.5))
        det.update(close)
        det.reset()
        result = det.update(close)
        assert result["event"] == "pinch_start"


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
