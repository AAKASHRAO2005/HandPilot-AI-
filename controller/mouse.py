"""
controller/mouse.py — Virtual cursor: camera → screen coordinate mapping + click dispatch.

Features:
    - PointOneEuroFilter: Adaptive low-pass smoothing (zero jitter at rest, zero lag at speed)
    - PointerBallistics: Dynamic non-linear acceleration
    - Click-Drift Freeze: Stabilizes cursor position when clicking to prevent fingertip drift
"""

import math
import time
from typing import Optional, Tuple

from controller import windows_input as wi
from utils.smoothing import PointOneEuroFilter, PointerBallistics, EMASmoother
from utils.logger import setup_logger

log = setup_logger(__name__)


class MouseController:
    """
    Translates normalized hand positions to Windows cursor movements and clicks.

    Args:
        screen_w, screen_h: Target screen resolution.
        filter_type: 'one_euro' (recommended) or 'ema'.
        min_cutoff: OneEuroFilter minimum cutoff frequency (Hz).
        beta: OneEuroFilter speed coefficient.
        d_cutoff: OneEuroFilter derivative cutoff frequency.
        alpha: EMA smoothing factor (if filter_type='ema').
        sensitivity: Multiplier applied to mapped position displacement.
        acceleration: If True, uses PointerBallistics non-linear acceleration.
        dead_zone: Pixel dead zone.
        active_region: Dict with x_min/x_max/y_min/y_max defining camera region.
        boundary_margin: Pixels from screen edge to clamp cursor.
        max_jump: Max pixels the cursor can move in one frame.
        mirror: Whether camera was mirrored.
    """

    def __init__(
        self,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
        filter_type: str = "one_euro",
        min_cutoff: float = 1.0,
        beta: float = 0.008,
        d_cutoff: float = 1.0,
        alpha: float = 0.20,
        sensitivity: float = 1.5,
        acceleration: bool = True,
        dead_zone: float = 8.0,
        active_region: Optional[dict] = None,
        boundary_margin: int = 5,
        max_jump: int = 400,
        mirror: bool = True,
    ) -> None:
        self.screen_w = screen_w or wi.SCREEN_W
        self.screen_h = screen_h or wi.SCREEN_H
        self.filter_type = filter_type
        self.sensitivity = sensitivity
        self.acceleration = acceleration
        self.boundary_margin = boundary_margin
        self.max_jump = max_jump
        self.mirror = mirror

        # Smoothing filters
        self._one_euro = PointOneEuroFilter(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            freq=30.0,
        )
        self._ema = EMASmoother(alpha=alpha, dead_zone=dead_zone)
        self._ballistics = PointerBallistics(base_sensitivity=sensitivity)

        # Active camera region used for coordinate mapping
        ar = active_region or {}
        self._ax_min = ar.get("x_min", 0.10)
        self._ax_max = ar.get("x_max", 0.90)
        self._ay_min = ar.get("y_min", 0.05)
        self._ay_max = ar.get("y_max", 0.85)

        # Current cursor state
        self._cursor_x: int = self.screen_w // 2
        self._cursor_y: int = self.screen_h // 2
        self._prev_raw_x: Optional[float] = None
        self._prev_raw_y: Optional[float] = None

        # Click drift freeze mechanism
        self._freeze_until: float = 0.0
        self._frozen_x: int = self._cursor_x
        self._frozen_y: int = self._cursor_y

        # Click cooldown tracking
        self._last_left_click: float = 0.0
        self._last_right_click: float = 0.0
        self._drag_active: bool = False

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def freeze(self, duration_s: float = 0.18) -> None:
        """Lock/freeze cursor position momentarily during click initiation."""
        now = time.perf_counter()
        self._freeze_until = now + duration_s
        self._frozen_x = self._cursor_x
        self._frozen_y = self._cursor_y

    def unfreeze(self) -> None:
        """Immediately release click freeze."""
        self._freeze_until = 0.0

    def update_cursor(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Map normalized camera position to screen coordinates and move Windows cursor.

        Args:
            norm_x, norm_y: Hand position in [0, 1] (normalized camera space).

        Returns:
            (screen_x, screen_y)
        """
        now = time.perf_counter()

        # 1. Mirror x if camera is mirrored
        if self.mirror:
            norm_x = 1.0 - norm_x

        # 2. Clamp to active region and re-normalize within it
        nx = (norm_x - self._ax_min) / max(self._ax_max - self._ax_min, 1e-6)
        ny = (norm_y - self._ay_min) / max(self._ay_max - self._ay_min, 1e-6)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # 3. Convert to raw screen pixels
        target_x = nx * self.screen_w
        target_y = ny * self.screen_h

        # 4. Check if cursor is frozen (click drift stabilization)
        if now < self._freeze_until and not self._drag_active:
            # Allow break-out if user deliberately makes a large movement
            raw_dx = target_x - self._frozen_x
            raw_dy = target_y - self._frozen_y
            if math.hypot(raw_dx, raw_dy) < 50.0:
                wi.move_mouse_absolute(self._frozen_x, self._frozen_y)
                return (self._frozen_x, self._frozen_y)
            else:
                self._freeze_until = 0.0

        # 5. Apply smoothing filter
        if self.filter_type == "one_euro":
            sx, sy = self._one_euro.filter(target_x, target_y, now)
        else:
            sx, sy = self._ema.update(target_x, target_y)

        # 6. Apply pointer ballistics (if enabled)
        if self.acceleration and self._prev_raw_x is not None and self._prev_raw_y is not None:
            dx = sx - self._cursor_x
            dy = sy - self._cursor_y
            adx, ady = self._ballistics.apply(dx, dy)
            sx = self._cursor_x + adx
            sy = self._cursor_y + ady

        self._prev_raw_x = target_x
        self._prev_raw_y = target_y

        # 7. Clamp to screen boundaries
        m = self.boundary_margin
        sx = int(max(m, min(self.screen_w - m, sx)))
        sy = int(max(m, min(self.screen_h - m, sy)))

        # 8. Safety cap for maximum jump
        dx = sx - self._cursor_x
        dy = sy - self._cursor_y
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
        """Reset smoother state (call when hand is lost)."""
        self._one_euro.reset()
        self._ema.reset()
        self._prev_raw_x = None
        self._prev_raw_y = None
        self._freeze_until = 0.0

    @property
    def cursor_pos(self) -> Tuple[int, int]:
        return (self._cursor_x, self._cursor_y)

    # ------------------------------------------------------------------
    # Click actions
    # ------------------------------------------------------------------

    def left_click(self, cooldown_ms: float = 300.0) -> bool:
        now = time.perf_counter()
        if (now - self._last_left_click) * 1000.0 < cooldown_ms:
            return False
        wi.left_click()
        self._last_left_click = now
        log.debug(f"LEFT CLICK @ {self._cursor_x},{self._cursor_y}")
        return True

    def right_click(self, cooldown_ms: float = 400.0) -> bool:
        now = time.perf_counter()
        if (now - self._last_right_click) * 1000.0 < cooldown_ms:
            return False
        wi.right_click()
        self._last_right_click = now
        log.debug(f"RIGHT CLICK @ {self._cursor_x},{self._cursor_y}")
        return True

    def double_click(self) -> None:
        wi.double_click()
        self._last_left_click = time.perf_counter()
        log.debug(f"DOUBLE CLICK @ {self._cursor_x},{self._cursor_y}")

    # ------------------------------------------------------------------
    # Drag
    # ------------------------------------------------------------------

    def start_drag(self) -> None:
        if not self._drag_active:
            self._freeze_until = 0.0  # unfreeze for drag
            wi.left_button_down()
            self._drag_active = True
            log.debug("DRAG START")

    def end_drag(self) -> None:
        if self._drag_active:
            wi.left_button_up()
            self._drag_active = False
            self._last_left_click = time.perf_counter()
            log.debug("DRAG END")

    def emergency_release(self) -> None:
        wi.release_all_buttons()
        self._drag_active = False
        self._freeze_until = 0.0

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
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        """Live-update cursor parameters."""
        if alpha is not None:
            self._ema.alpha = max(0.01, min(1.0, alpha))
        if sensitivity is not None:
            self.sensitivity = max(0.1, sensitivity)
            self._ballistics.base_sensitivity = self.sensitivity
        if dead_zone is not None:
            self._ema.dead_zone = max(0.0, dead_zone)
        if min_cutoff is not None or beta is not None:
            self._one_euro.update_params(min_cutoff=min_cutoff, beta=beta)
