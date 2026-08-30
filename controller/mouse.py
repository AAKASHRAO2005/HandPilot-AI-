"""
controller/mouse.py — Virtual cursor: camera → screen coordinate mapping + click dispatch.

Converts normalized hand landmark positions (0–1) into Windows screen coordinates,
applies smoothing and sensitivity scaling, then calls windows_input to move/click.
"""

import time
from typing import Optional, Tuple

from controller import windows_input as wi
from utils.smoothing import EMASmoother
from utils.logger import setup_logger

log = setup_logger(__name__)


class MouseController:
    """
    Translates normalized hand positions to Windows cursor movements and clicks.

    Args:
        screen_w, screen_h: Target screen resolution (auto-detected from windows_input).
        alpha: EMA smoothing factor.
        sensitivity: Multiplier applied to mapped position displacement.
        dead_zone: Pixel dead zone.
        active_region: Dict with x_min/x_max/y_min/y_max defining camera region
                       used for mapping (avoids extreme corners being unreachable).
        boundary_margin: Pixels from screen edge to clamp cursor.
        max_jump: Max pixels the cursor can move in one frame (safety limit).
        mirror: Whether camera was mirrored (affects x-axis direction).
    """

    def __init__(
        self,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
        alpha: float = 0.20,
        sensitivity: float = 1.5,
        dead_zone: float = 8.0,
        active_region: Optional[dict] = None,
        boundary_margin: int = 5,
        max_jump: int = 400,
        mirror: bool = True,
    ) -> None:
        self.screen_w = screen_w or wi.SCREEN_W
        self.screen_h = screen_h or wi.SCREEN_H
        self.sensitivity = sensitivity
        self.boundary_margin = boundary_margin
        self.max_jump = max_jump
        self.mirror = mirror

        self._smoother = EMASmoother(alpha=alpha, dead_zone=dead_zone)

        # Active camera region used for coordinate mapping
        ar = active_region or {}
        self._ax_min = ar.get("x_min", 0.10)
        self._ax_max = ar.get("x_max", 0.90)
        self._ay_min = ar.get("y_min", 0.05)
        self._ay_max = ar.get("y_max", 0.85)

        # Current cursor state
        self._cursor_x: int = self.screen_w // 2
        self._cursor_y: int = self.screen_h // 2

        # Click cooldown tracking
        self._last_left_click: float = 0.0
        self._last_right_click: float = 0.0
        self._drag_active: bool = False

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def update_cursor(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Map a normalized camera position to screen coordinates and move cursor.

        Args:
            norm_x, norm_y: Hand position in [0, 1] (normalized camera space).

        Returns:
            (screen_x, screen_y) — actual cursor position sent to Windows.
        """
        # 1. Mirror x if camera is mirrored
        if self.mirror:
            norm_x = 1.0 - norm_x

        # 2. Clamp to active region and re-normalize within it
        nx = (norm_x - self._ax_min) / max(self._ax_max - self._ax_min, 1e-6)
        ny = (norm_y - self._ay_min) / max(self._ay_max - self._ay_min, 1e-6)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # 3. Convert to raw screen pixels
        raw_x = nx * self.screen_w
        raw_y = ny * self.screen_h

        # 4. Apply smoothing (dead zone in pixel space)
        sx, sy = self._smoother.update(raw_x, raw_y)

        # 5. Clamp to screen boundaries
        m = self.boundary_margin
        sx = int(max(m, min(self.screen_w - m, sx)))
        sy = int(max(m, min(self.screen_h - m, sy)))

        # 6. Safety: cap maximum cursor jump per frame
        dx = sx - self._cursor_x
        dy = sy - self._cursor_y
        import math
        dist = math.hypot(dx, dy)
        if dist > self.max_jump:
            scale = self.max_jump / dist
            sx = int(self._cursor_x + dx * scale)
            sy = int(self._cursor_y + dy * scale)

        self._cursor_x = sx
        self._cursor_y = sy

        wi.move_mouse_absolute(sx, sy)
        return (sx, sy)

    def reset_smooth(self) -> None:
        """Reset EMA smoother (call when hand is lost)."""
        self._smoother.reset()

    @property
    def cursor_pos(self) -> Tuple[int, int]:
        return (self._cursor_x, self._cursor_y)

    # ------------------------------------------------------------------
    # Click actions
    # ------------------------------------------------------------------

    def left_click(self, cooldown_ms: float = 300.0) -> bool:
        """
        Send a left click if cooldown has elapsed.

        Returns:
            True if click was sent, False if suppressed by cooldown.
        """
        now = time.perf_counter()
        if (now - self._last_left_click) * 1000 < cooldown_ms:
            return False
        wi.left_click()
        self._last_left_click = now
        log.debug(f"LEFT CLICK @ {self._cursor_x},{self._cursor_y}")
        return True

    def right_click(self, cooldown_ms: float = 400.0) -> bool:
        """Send a right click if cooldown has elapsed."""
        now = time.perf_counter()
        if (now - self._last_right_click) * 1000 < cooldown_ms:
            return False
        wi.right_click()
        self._last_right_click = now
        log.debug(f"RIGHT CLICK @ {self._cursor_x},{self._cursor_y}")
        return True

    def double_click(self) -> None:
        """Send a double click."""
        wi.double_click()
        self._last_left_click = time.perf_counter()
        log.debug(f"DOUBLE CLICK @ {self._cursor_x},{self._cursor_y}")

    # ------------------------------------------------------------------
    # Drag
    # ------------------------------------------------------------------

    def start_drag(self) -> None:
        """Press and hold left mouse button for drag."""
        if not self._drag_active:
            wi.left_button_down()
            self._drag_active = True
            log.debug("DRAG START")

    def end_drag(self) -> None:
        """Release left mouse button to end drag."""
        if self._drag_active:
            wi.left_button_up()
            self._drag_active = False
            self._last_left_click = time.perf_counter()
            log.debug("DRAG END")

    def emergency_release(self) -> None:
        """Release all buttons and reset drag state."""
        wi.release_all_buttons()
        self._drag_active = False

    @property
    def is_dragging(self) -> bool:
        return self._drag_active

    # ------------------------------------------------------------------
    # Config update
    # ------------------------------------------------------------------

    def update_config(
        self,
        alpha: Optional[float] = None,
        sensitivity: Optional[float] = None,
        dead_zone: Optional[float] = None,
    ) -> None:
        """Live-update cursor parameters (from dashboard sliders)."""
        if alpha is not None:
            self._smoother.alpha = max(0.01, min(1.0, alpha))
        if sensitivity is not None:
            self.sensitivity = max(0.1, sensitivity)
        if dead_zone is not None:
            self._smoother.dead_zone = max(0.0, dead_zone)
