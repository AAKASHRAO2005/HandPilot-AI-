"""
gestures/state_machine.py — Mouse interaction state machine.

States:
    IDLE          — No hand present
    TRACKING      — Hand visible, cursor moving
    PINCH_START   — Index+thumb just pinched (fires left click)
    PINCH_HOLD    — Pinch held past drag threshold (drag mode)
    DRAG          — Mouse button held, hand moving
    PINCH_RELEASE — Pinch released (ends drag or no-op)
    SCROLL        — Two fingers up, hand moving vertically
    RIGHT_CLICK   — Middle+thumb pinch (one-shot)
    SWIPE         — Fast hand movement (fires keyboard shortcut)
"""

from enum import Enum, auto
from typing import Optional
from utils.logger import setup_logger

log = setup_logger(__name__)


class MouseState(Enum):
    IDLE = auto()
    TRACKING = auto()
    PINCH_START = auto()
    PINCH_HOLD = auto()
    DRAG = auto()
    PINCH_RELEASE = auto()
    SCROLL = auto()
    RIGHT_CLICK = auto()
    SWIPE = auto()


class GestureStateMachine:
    """
    Centralised state machine that coordinates all gesture detectors and
    resolves conflicts using the priority system:

        PINCH > RIGHT_CLICK > SCROLL > SWIPE > CURSOR MOVEMENT

    Args:
        drag_hold_delay_ms: Milliseconds a pinch must be held before drag starts.
    """

    def __init__(self, drag_hold_delay_ms: float = 300.0) -> None:
        self._state = MouseState.IDLE
        self._drag_hold_delay_ms = drag_hold_delay_ms
        self._prev_state = MouseState.IDLE

    @property
    def state(self) -> MouseState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    def transition(self, new_state: MouseState) -> bool:
        """
        Attempt a state transition.

        Returns:
            True if state changed.
        """
        if new_state == self._state:
            return False
        log.debug(f"State: {self._state.name} → {new_state.name}")
        self._prev_state = self._state
        self._state = new_state
        return True

    def reset(self) -> None:
        """Reset to IDLE."""
        self.transition(MouseState.IDLE)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_idle(self) -> bool:
        return self._state == MouseState.IDLE

    @property
    def is_tracking(self) -> bool:
        return self._state in (MouseState.TRACKING, MouseState.SCROLL, MouseState.SWIPE)

    @property
    def is_pinching(self) -> bool:
        return self._state in (MouseState.PINCH_START, MouseState.PINCH_HOLD, MouseState.DRAG)

    @property
    def is_dragging(self) -> bool:
        return self._state == MouseState.DRAG

    @property
    def is_scrolling(self) -> bool:
        return self._state == MouseState.SCROLL
