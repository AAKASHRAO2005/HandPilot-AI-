"""
gestures/gesture_engine.py — Main gesture recognition orchestrator with click-drift stabilization.

Per-frame pipeline:
    1. Extract landmarks from HandResult.
    2. Update all gesture detectors.
    3. Apply priority rules & click-drift stabilization.
    4. Emit GestureEvent.
    5. Drive controllers.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from vision.hand_tracker import HandResult
from vision.landmarks import (
    get_fingertip,
    get_palm_center,
    count_extended_fingers,
    is_fist,
    is_open_palm,
)
from gestures.pinch import PinchDetector, DoublePinchDetector
from gestures.movement import MovementTracker
from gestures.swipe import SwipeDetector
from gestures.state_machine import GestureStateMachine, MouseState
from controller.mouse import MouseController
from controller.scroll import ScrollController
from controller.keyboard import KeyboardController
from utils.logger import setup_logger

log = setup_logger(__name__)


@dataclass
class GestureEvent:
    """Snapshot of gesture state for one frame."""
    gesture: str = "none"
    action: str = "none"
    state: str = "IDLE"
    cursor_x: int = 0
    cursor_y: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 0.0
    extended_fingers: int = 0
    pinch_distance: float = 1.0
    is_dragging: bool = False
    extra: dict = field(default_factory=dict)


class GestureEngine:
    """
    Orchestrates all gesture detectors and drives controllers.
    """

    def __init__(
        self,
        cfg: dict,
        mouse: MouseController,
        scroll: ScrollController,
        keyboard: KeyboardController,
    ) -> None:
        self._cfg = cfg
        self._mouse = mouse
        self._scroll = scroll
        self._keyboard = keyboard

        g = cfg.get("gestures", {})
        k = cfg.get("keyboard_gestures", {})
        s = cfg.get("safety", {})

        # Pinch detectors
        lc_cfg = g.get("left_click", {})
        rc_cfg = g.get("right_click", {})
        dc_cfg = g.get("double_click", {})
        drag_cfg = g.get("drag", {})

        scale_inv = lc_cfg.get("scale_invariant", False)

        self._left_pinch = PinchDetector(
            finger_a=0, finger_b=1,
            threshold=lc_cfg.get("pinch_threshold", 0.06),
            scale_invariant=scale_inv,
            name="index_pinch",
        )
        self._double_pinch = DoublePinchDetector(
            self._left_pinch,
            window_ms=dc_cfg.get("window_ms", 500.0),
        )
        self._right_pinch = PinchDetector(
            finger_a=0, finger_b=2,
            threshold=rc_cfg.get("pinch_threshold", 0.06),
            scale_invariant=scale_inv,
            name="middle_pinch",
        )

        # Movement tracker
        swipe_cfg = g.get("swipe", {})
        self._movement = MovementTracker(
            history_size=swipe_cfg.get("window_frames", 8),
        )

        # Swipe detector
        self._swipe = SwipeDetector(
            velocity_threshold=swipe_cfg.get("velocity_threshold", 0.05),
            window_frames=swipe_cfg.get("window_frames", 8),
        )

        # State machine
        self._sm = GestureStateMachine(
            drag_hold_delay_ms=drag_cfg.get("hold_delay_ms", 300.0),
        )

        # Config flags
        self._enabled = True
        self._lc_enabled = lc_cfg.get("enabled", True)
        self._rc_enabled = rc_cfg.get("enabled", True)
        self._dc_enabled = dc_cfg.get("enabled", True)
        self._drag_enabled = drag_cfg.get("enabled", True)
        self._scroll_enabled = g.get("scroll", {}).get("enabled", True)
        self._swipe_enabled = g.get("swipe", {}).get("enabled", True)
        self._scroll_fingers = g.get("scroll", {}).get("trigger_fingers", 2)

        self._lc_cooldown = lc_cfg.get("cooldown_ms", 300.0)
        self._rc_cooldown = rc_cfg.get("cooldown_ms", 400.0)
        self._drag_hold_ms = drag_cfg.get("hold_delay_ms", 300.0)
        self._confidence_threshold = s.get("confidence_threshold", 0.50)
        self._use_index_finger = cfg.get("cursor", {}).get("use_index_finger", True)
        self._click_freeze_duration = lc_cfg.get("click_freeze_s", 0.18)

        # Internal state
        self._no_hand_frames = 0
        self._auto_release_frames = s.get("auto_release_frames", 10)
        self._pinch_start_time: float = 0.0
        self._drag_started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        hand_results: list,
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> GestureEvent:
        """
        Process one frame's hand detection results and drive controllers.
        """
        if not self._enabled:
            return GestureEvent(gesture="disabled", action="control_off",
                                state="DISABLED", fps=fps, latency_ms=latency_ms)

        # No hand detected
        if not hand_results:
            self._no_hand_frames += 1
            if self._no_hand_frames >= self._auto_release_frames:
                self._handle_hand_loss()
            return GestureEvent(gesture="no_hand", action="none",
                                state=self._sm.state_name, fps=fps, latency_ms=latency_ms)

        self._no_hand_frames = 0
        hand = hand_results[0]

        if hand.confidence < self._confidence_threshold:
            return GestureEvent(gesture="low_confidence", action="none",
                                state=self._sm.state_name, fps=fps, latency_ms=latency_ms,
                                confidence=hand.confidence)

        landmarks = hand.landmarks
        extended = count_extended_fingers(landmarks)

        # Control point (index fingertip or palm center)
        if self._use_index_finger:
            cx, cy = get_fingertip(landmarks, finger=1)
        else:
            cx, cy = get_palm_center(landmarks)

        # Update movement tracker
        vx, vy = self._movement.update(cx, cy)

        # === Detect Pinches ===
        pinch_result = self._double_pinch.update(landmarks)
        pinch_dist = pinch_result["distance"]
        pinch_event = pinch_result["event"]
        is_double = pinch_result.get("double_click", False)
        pinch_held = pinch_result["hold_duration"]
        is_pinched = pinch_result["is_pinched"]
        is_closing = pinch_result.get("is_closing", False)

        right_result = self._right_pinch.update(landmarks)

        # Click-drift stabilization: if fingers are closing in for a click, freeze cursor
        if is_closing and not self._drag_started:
            self._mouse.freeze(self._click_freeze_duration)

        # === Update cursor ===
        sx, sy = self._mouse.update_cursor(cx, cy)

        gesture = "tracking"
        action = "cursor_move"

        # --- Handle double click ---
        if is_double and self._dc_enabled:
            self._mouse.freeze(0.20)
            self._mouse.double_click()
            gesture = "double_pinch"
            action = "double_click"
            self._sm.transition(MouseState.PINCH_START)
            return self._make_event(gesture, action, sx, sy, extended, pinch_dist,
                                    fps, latency_ms, hand.confidence)

        # --- Handle pinch (left click / drag) ---
        if self._lc_enabled and pinch_event == "pinch_start":
            self._sm.transition(MouseState.PINCH_START)
            self._pinch_start_time = time.perf_counter()
            self._drag_started = False
            self._mouse.freeze(self._click_freeze_duration)
            gesture = "pinch"
            action = "left_click"
            self._mouse.left_click(cooldown_ms=self._lc_cooldown)

        elif self._drag_enabled and pinch_event == "pinch_hold":
            hold_ms = pinch_held * 1000.0
            if hold_ms > self._drag_hold_ms and not self._drag_started:
                self._drag_started = True
                self._mouse.start_drag()
                self._sm.transition(MouseState.DRAG)

            if self._drag_started:
                self._sm.transition(MouseState.DRAG)
                gesture = "pinch_hold"
                action = "drag"
            else:
                gesture = "pinch"
                action = "left_click_hold"

        elif pinch_event == "pinch_release":
            if self._drag_started:
                self._mouse.end_drag()
                self._drag_started = False
            self._mouse.unfreeze()
            self._sm.transition(MouseState.TRACKING)
            gesture = "pinch_release"
            action = "release"

        # --- Handle right click ---
        elif self._rc_enabled and right_result["event"] == "pinch_start":
            self._mouse.freeze(self._click_freeze_duration)
            self._sm.transition(MouseState.RIGHT_CLICK)
            self._mouse.right_click(cooldown_ms=self._rc_cooldown)
            gesture = "middle_pinch"
            action = "right_click"

        # --- Handle scroll (2+ fingers extended, vertical movement) ---
        elif (
            not is_pinched
            and self._scroll_enabled
            and extended >= self._scroll_fingers
            and not self._sm.is_pinching
        ):
            delta_y = self._movement.delta[1]
            delta_x = self._movement.delta[0]
            scrolled = self._scroll.scroll_vertical(delta_y * 30.0)
            if scrolled:
                self._sm.transition(MouseState.SCROLL)
                gesture = "open_hand_vertical"
                action = "scroll_vertical"
            else:
                self._sm.transition(MouseState.TRACKING)

        # --- Handle swipe ---
        elif self._swipe_enabled and not is_pinched and not self._sm.is_pinching:
            swipe_dir = self._swipe.update(vx, vy)
            if swipe_dir:
                self._keyboard.trigger(swipe_dir)
                self._sm.transition(MouseState.SWIPE)
                gesture = swipe_dir
                action = f"keyboard_{swipe_dir}"
            else:
                self._swipe.update(0, 0)
                self._sm.transition(MouseState.TRACKING)

        # --- Default: cursor tracking ---
        else:
            if not self._sm.is_pinching and not self._sm.is_dragging:
                self._sm.transition(MouseState.TRACKING)

        return self._make_event(gesture, action, sx, sy, extended, pinch_dist,
                                fps, latency_ms, hand.confidence)

    # ------------------------------------------------------------------
    # Enable / Disable / Stop
    # ------------------------------------------------------------------

    def enable(self) -> None:
        self._enabled = True
        log.info("Gesture control ENABLED.")

    def disable(self) -> None:
        self._enabled = False
        self._handle_hand_loss()
        log.info("Gesture control DISABLED.")

    def toggle(self) -> bool:
        if self._enabled:
            self.disable()
        else:
            self.enable()
        return self._enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def emergency_stop(self) -> None:
        self._mouse.emergency_release()
        self._drag_started = False
        self._sm.reset()
        log.warning("EMERGENCY STOP triggered!")

    def update_cursor_config(self, **kwargs) -> None:
        self._mouse.update_config(**kwargs)

    def update_scroll_config(self, **kwargs) -> None:
        self._scroll.update_config(**kwargs)

    def _handle_hand_loss(self) -> None:
        if self._drag_started:
            log.warning("Hand lost during drag — releasing mouse button.")
            self._mouse.end_drag()
            self._drag_started = False
        self._mouse.reset_smooth()
        self._scroll.reset()
        self._movement.reset()
        self._left_pinch.reset()
        self._right_pinch.reset()
        self._swipe.reset()
        self._sm.reset()

    def _make_event(
        self,
        gesture: str,
        action: str,
        sx: int,
        sy: int,
        extended: int,
        pinch_dist: float,
        fps: float,
        latency_ms: float,
        confidence: float,
    ) -> GestureEvent:
        return GestureEvent(
            gesture=gesture,
            action=action,
            state=self._sm.state_name,
            cursor_x=sx,
            cursor_y=sy,
            fps=fps,
            latency_ms=latency_ms,
            confidence=confidence,
            extended_fingers=extended,
            pinch_distance=pinch_dist,
            is_dragging=self._drag_started,
        )
