"""
ui/dashboard.py — tkinter desktop dashboard for the Hand Gesture Controller.

Provides:
    - Live status panel (camera, YOLO, tracking, controller)
    - Current gesture and action display
    - Cursor position readout
    - FPS and latency meters
    - Sliders for sensitivity, scroll speed, and gesture threshold
    - Enable/Disable button
    - Camera selector
    - Embedded live camera preview
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, font
from typing import Callable, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
BG = "#0d0d0d"
PANEL_BG = "#141414"
CARD_BG = "#1a1a2e"
ACCENT = "#7c3aed"       # Purple
ACCENT2 = "#06d6a0"      # Teal
DANGER = "#ef233c"
WARNING = "#f77f00"
TEXT = "#e2e8f0"
TEXT_DIM = "#64748b"
GREEN = "#22c55e"
RED = "#ef4444"
YELLOW = "#eab308"
BORDER = "#2d2d4e"


def _dot(canvas: tk.Canvas, color: str) -> None:
    canvas.configure(bg=color)


class Dashboard:
    """
    Dark-themed tkinter dashboard window.

    Args:
        on_enable: Callback when Enable button pressed.
        on_disable: Callback when Disable button pressed.
        on_emergency: Callback when Emergency Stop pressed.
        on_camera_change: Callback(int) when camera index changed.
        on_sensitivity_change: Callback(float) for cursor sensitivity slider.
        on_scroll_speed_change: Callback(float) for scroll speed slider.
        on_threshold_change: Callback(float) for gesture threshold slider.
    """

    def __init__(
        self,
        on_enable: Optional[Callable] = None,
        on_disable: Optional[Callable] = None,
        on_emergency: Optional[Callable] = None,
        on_camera_change: Optional[Callable] = None,
        on_sensitivity_change: Optional[Callable] = None,
        on_scroll_speed_change: Optional[Callable] = None,
        on_threshold_change: Optional[Callable] = None,
        initial_camera: int = 0,
    ) -> None:
        self._on_enable = on_enable or (lambda: None)
        self._on_disable = on_disable or (lambda: None)
        self._on_emergency = on_emergency or (lambda: None)
        self._on_camera_change = on_camera_change or (lambda idx: None)
        self._on_sensitivity_change = on_sensitivity_change or (lambda v: None)
        self._on_scroll_speed_change = on_scroll_speed_change or (lambda v: None)
        self._on_threshold_change = on_threshold_change or (lambda v: None)

        self._root: Optional[tk.Tk] = None
        self._running = False
        self._frame_image: Optional[ImageTk.PhotoImage] = None
        self._frame_lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None

        # Shared state (updated from main thread)
        self._status: Dict[str, Any] = {
            "camera": False,
            "yolo": False,
            "tracking": False,
            "enabled": True,
            "gesture": "—",
            "action": "—",
            "state": "IDLE",
            "fps": 0.0,
            "latency": 0.0,
            "cursor_x": 0,
            "cursor_y": 0,
            "confidence": 0.0,
            "extended": 0,
            "ws_clients": 0,
        }
        self._camera_index = initial_camera

    # ------------------------------------------------------------------
    # Public API (called from main thread or other threads)
    # ------------------------------------------------------------------

    def update_status(self, **kwargs) -> None:
        """Thread-safe status update."""
        self._status.update(kwargs)

    def update_frame(self, frame: np.ndarray) -> None:
        """Push a new BGR frame for display."""
        with self._frame_lock:
            self._pending_frame = frame

    def run(self) -> None:
        """Build and run the dashboard (blocking — run in its own thread)."""
        self._build_ui()
        self._running = True
        self._schedule_refresh()
        try:
            self._root.mainloop()
        except Exception as e:
            log.error(f"Dashboard error: {e}")
        finally:
            self._running = False

    def close(self) -> None:
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = tk.Tk()
        root.title("✋ Hand Gesture Desktop Controller")
        root.configure(bg=BG)
        root.resizable(False, False)

        # Fonts
        try:
            root.tk.call("font", "create", "TitleFont", "-family", "Segoe UI", "-size", 14, "-weight", "bold")
            title_font = ("Segoe UI", 14, "bold")
            body_font = ("Segoe UI", 10)
            mono_font = ("Consolas", 10)
            big_font = ("Segoe UI", 18, "bold")
            small_font = ("Segoe UI", 8)
        except Exception:
            title_font = ("Arial", 14, "bold")
            body_font = ("Arial", 10)
            mono_font = ("Courier", 10)
            big_font = ("Arial", 18, "bold")
            small_font = ("Arial", 8)

        self._root = root
        self._fonts = {
            "title": title_font, "body": body_font, "mono": mono_font,
            "big": big_font, "small": small_font,
        }

        # === HEADER ===
        header = tk.Frame(root, bg=ACCENT, pady=8)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="✋  HAND GESTURE DESKTOP CONTROLLER",
                 fg="white", bg=ACCENT, font=title_font).pack()

        # === LEFT PANEL: Camera feed ===
        left = tk.Frame(root, bg=PANEL_BG, padx=6, pady=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)

        tk.Label(left, text="LIVE CAMERA", fg=TEXT_DIM, bg=PANEL_BG, font=small_font).pack(anchor="w")
        self._cam_label = tk.Label(left, bg="black", width=640, height=360)
        self._cam_label.pack()

        # === RIGHT PANEL: Controls ===
        right = tk.Frame(root, bg=PANEL_BG, padx=10, pady=8, width=300)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_propagate(False)

        def section(parent, label):
            f = tk.Frame(parent, bg=CARD_BG, padx=8, pady=6, relief="flat", bd=0)
            f.pack(fill="x", pady=4)
            tk.Label(f, text=label, fg=ACCENT2, bg=CARD_BG, font=small_font).pack(anchor="w")
            sep = tk.Frame(f, bg=BORDER, height=1)
            sep.pack(fill="x", pady=(2, 4))
            return f

        # -- System status --
        sf = section(right, "SYSTEM STATUS")
        self._status_dots = {}
        self._status_labels = {}
        for key, label in [("camera", "Camera"), ("yolo", "YOLO26"), ("tracking", "Tracking"), ("enabled", "Controller")]:
            row = tk.Frame(sf, bg=CARD_BG)
            row.pack(fill="x", pady=1)
            dot = tk.Label(row, text="●", fg=RED, bg=CARD_BG, font=body_font)
            dot.pack(side="left")
            tk.Label(row, text=f"  {label}", fg=TEXT, bg=CARD_BG, font=body_font).pack(side="left")
            self._status_dots[key] = dot
            status_lbl = tk.Label(row, text="Offline", fg=TEXT_DIM, bg=CARD_BG, font=small_font)
            status_lbl.pack(side="right")
            self._status_labels[key] = status_lbl

        # -- Performance --
        pf = section(right, "PERFORMANCE")
        pr = tk.Frame(pf, bg=CARD_BG)
        pr.pack(fill="x")
        self._fps_lbl = tk.Label(pr, text="FPS: —", fg=GREEN, bg=CARD_BG, font=mono_font)
        self._fps_lbl.pack(side="left")
        self._lat_lbl = tk.Label(pr, text="Latency: — ms", fg=YELLOW, bg=CARD_BG, font=mono_font)
        self._lat_lbl.pack(side="right")

        # -- Gesture --
        gf = section(right, "CURRENT GESTURE")
        self._gesture_lbl = tk.Label(gf, text="—", fg="white", bg=CARD_BG, font=big_font)
        self._gesture_lbl.pack()
        self._action_lbl = tk.Label(gf, text="Action: —", fg=ACCENT2, bg=CARD_BG, font=body_font)
        self._action_lbl.pack()
        self._state_lbl = tk.Label(gf, text="State: IDLE", fg=TEXT_DIM, bg=CARD_BG, font=small_font)
        self._state_lbl.pack()
        self._conf_lbl = tk.Label(gf, text="Confidence: —", fg=TEXT_DIM, bg=CARD_BG, font=small_font)
        self._conf_lbl.pack()

        # -- Cursor --
        cf = section(right, "CURSOR POSITION")
        self._cursor_lbl = tk.Label(cf, text="X: —   Y: —", fg=TEXT, bg=CARD_BG, font=mono_font)
        self._cursor_lbl.pack()

        # -- Controls --
        ctf = section(right, "CONTROLS")
        btn_row = tk.Frame(ctf, bg=CARD_BG)
        btn_row.pack(fill="x", pady=2)
        self._enable_btn = tk.Button(
            btn_row, text="⏵ Enable", bg=GREEN, fg="white", font=body_font,
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._on_enable_click,
        )
        self._enable_btn.pack(side="left", padx=(0, 4))
        self._disable_btn = tk.Button(
            btn_row, text="⏸ Disable", bg=WARNING, fg="white", font=body_font,
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._on_disable_click,
        )
        self._disable_btn.pack(side="left")

        self._emergency_btn = tk.Button(
            ctf, text="⛔ EMERGENCY STOP  [Ctrl+Alt+X]",
            bg=DANGER, fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, pady=6, cursor="hand2",
            command=self._on_emergency_click,
        )
        self._emergency_btn.pack(fill="x", pady=(4, 0))

        # -- Sliders --
        sf2 = section(right, "FINE TUNING")
        self._sensitivity_var = tk.DoubleVar(value=1.5)
        self._scroll_var = tk.DoubleVar(value=1.0)
        self._threshold_var = tk.DoubleVar(value=0.06)

        for label, var, from_, to, res, cb in [
            ("Cursor Sensitivity", self._sensitivity_var, 0.5, 3.0, 0.1, self._on_sensitivity),
            ("Scroll Speed", self._scroll_var, 0.2, 3.0, 0.1, self._on_scroll_speed),
            ("Pinch Threshold", self._threshold_var, 0.02, 0.15, 0.005, self._on_threshold),
        ]:
            tk.Label(sf2, text=label, fg=TEXT_DIM, bg=CARD_BG, font=small_font).pack(anchor="w")
            sl = tk.Scale(
                sf2, variable=var, from_=from_, to=to, resolution=res,
                orient="horizontal", bg=CARD_BG, fg=TEXT, troughcolor=BORDER,
                highlightthickness=0, sliderlength=15, command=cb,
            )
            sl.pack(fill="x")

        # -- Camera selector --
        camf = section(right, "CAMERA")
        cam_row = tk.Frame(camf, bg=CARD_BG)
        cam_row.pack(fill="x")
        tk.Label(cam_row, text="Index:", fg=TEXT_DIM, bg=CARD_BG, font=body_font).pack(side="left")
        self._cam_var = tk.StringVar(value=str(self._camera_index))
        cam_spin = tk.Spinbox(
            cam_row, from_=0, to=9, textvariable=self._cam_var,
            width=4, bg=CARD_BG, fg=TEXT, font=body_font,
            command=self._on_camera_select, relief="flat",
        )
        cam_spin.pack(side="left", padx=4)
        tk.Button(
            cam_row, text="Switch", bg=ACCENT, fg="white", font=small_font,
            relief="flat", padx=6, cursor="hand2",
            command=self._on_camera_select,
        ).pack(side="left")

        # -- WebSocket status --
        wsf = section(right, "BROWSER EXTENSION")
        self._ws_lbl = tk.Label(wsf, text="WebSocket: ws://127.0.0.1:8765", fg=TEXT_DIM, bg=CARD_BG, font=small_font)
        self._ws_lbl.pack(anchor="w")
        self._wsc_lbl = tk.Label(wsf, text="Connected clients: 0", fg=TEXT_DIM, bg=CARD_BG, font=small_font)
        self._wsc_lbl.pack(anchor="w")

        # Footer
        footer = tk.Frame(root, bg=BG, pady=4)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Label(footer, text="Toggle: Ctrl+Alt+G  |  Emergency Stop: Ctrl+Alt+X",
                 fg=TEXT_DIM, bg=BG, font=small_font).pack()

        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        if not self._running:
            return
        self._refresh()
        self._root.after(33, self._schedule_refresh)  # ~30 fps refresh

    def _refresh(self) -> None:
        s = self._status

        # Status dots
        for key, dot in self._status_dots.items():
            val = s.get(key, False)
            color = GREEN if val else RED
            dot.configure(fg=color)
            lbl = self._status_labels[key]
            lbl.configure(text="Active" if val else "Offline", fg=color)

        # Performance
        self._fps_lbl.configure(text=f"FPS: {s.get('fps', 0):.1f}")
        self._lat_lbl.configure(text=f"Latency: {s.get('latency', 0):.0f} ms")

        # Gesture
        gesture = s.get("gesture", "—").replace("_", " ").upper()
        self._gesture_lbl.configure(text=gesture)
        self._action_lbl.configure(text=f"Action: {s.get('action', '—')}")
        self._state_lbl.configure(text=f"State: {s.get('state', 'IDLE')}")
        self._conf_lbl.configure(text=f"Confidence: {s.get('confidence', 0):.2f}  Fingers: {s.get('extended', 0)}")

        # Cursor
        self._cursor_lbl.configure(
            text=f"X: {s.get('cursor_x', 0):>5}   Y: {s.get('cursor_y', 0):>5}"
        )

        # WebSocket
        self._wsc_lbl.configure(text=f"Connected clients: {s.get('ws_clients', 0)}")

        # Camera frame
        with self._frame_lock:
            frame = self._pending_frame
            self._pending_frame = None

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                # Fit to display size
                dw, dh = 640, 360
                scale = min(dw / w, dh / h)
                nw, nh = int(w * scale), int(h * scale)
                rgb = cv2.resize(rgb, (nw, nh))
                img = Image.fromarray(rgb)
                self._frame_image = ImageTk.PhotoImage(img)
                self._cam_label.configure(image=self._frame_image, width=dw, height=dh)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Button/slider callbacks
    # ------------------------------------------------------------------

    def _on_enable_click(self) -> None:
        self._status["enabled"] = True
        self._on_enable()

    def _on_disable_click(self) -> None:
        self._status["enabled"] = False
        self._on_disable()

    def _on_emergency_click(self) -> None:
        self._on_emergency()

    def _on_sensitivity(self, val) -> None:
        self._on_sensitivity_change(float(val))

    def _on_scroll_speed(self, val) -> None:
        self._on_scroll_speed_change(float(val))

    def _on_threshold(self, val) -> None:
        self._on_threshold_change(float(val))

    def _on_camera_select(self) -> None:
        try:
            idx = int(self._cam_var.get())
            self._on_camera_change(idx)
        except ValueError:
            pass
