"""
ui/dashboard.py — Modern Dark-Themed Desktop Dashboard with Interactive Gesture Guide.

Features:
    - Real-time gesture recognition guide showing which sign does what
    - Active sign highlighting: cards dynamically light up when the user makes the gesture
    - Embedded live camera preview with hand skeleton and overlays
    - System status indicators, performance gauges, and fine-tuning sliders
    - One-click Enable, Disable, and Emergency Stop controls
"""

import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.logger import setup_logger

log = setup_logger(__name__)


# Modern Dark Palette
BG_MAIN = "#0b0f19"         # Deep Navy / Black
PANEL_BG = "#111827"        # Dark Slate
CARD_BG = "#1e293b"         # Card Background
CARD_ACTIVE = "#0f3a40"     # Active Highlight Card Background
BORDER_COLOR = "#334155"    # Subtle Border
BORDER_ACTIVE = "#06b6d4"   # Neon Cyan Active Border
ACCENT_CYAN = "#06b6d4"     # Cyan
ACCENT_GREEN = "#10b981"    # Emerald
ACCENT_PURPLE = "#8b5cf6"   # Purple
ACCENT_RED = "#ef4444"      # Crimson
ACCENT_YELLOW = "#f59e0b"   # Amber
TEXT_PRIMARY = "#f8fafc"    # Bright White
TEXT_SECONDARY = "#94a3b8"  # Slate Muted
TEXT_MUTED = "#64748b"


class Dashboard:
    """
    Tkinter dashboard with real-time gesture feedback and interactive guide.
    """

    GESTURE_GUIDE_ITEMS = [
        {
            "id": "move",
            "keys": ["tracking", "cursor_move"],
            "icon": "✋",
            "name": "Open Palm / Index Point",
            "action": "Move Cursor",
            "detail": "Smooth 1€ filtered cursor navigation",
        },
        {
            "id": "left_click",
            "keys": ["pinch", "left_click"],
            "icon": "🤏",
            "name": "Index Pinch (Thumb + Index)",
            "action": "Left Click",
            "detail": "Quick tap to click with drift freeze",
        },
        {
            "id": "drag",
            "keys": ["pinch_hold", "drag", "left_click_hold"],
            "icon": "✊",
            "name": "Pinch & Hold",
            "action": "Drag & Drop",
            "detail": "Hold pinch > 300ms to grab and drag",
        },
        {
            "id": "double_click",
            "keys": ["double_pinch", "double_click"],
            "icon": "⚡",
            "name": "Double Index Pinch",
            "action": "Double Click",
            "detail": "Two rapid pinches within 500ms",
        },
        {
            "id": "right_click",
            "keys": ["middle_pinch", "right_click"],
            "icon": "✌️",
            "name": "Middle Pinch (Thumb + Middle)",
            "action": "Right Click",
            "detail": "Opens context / right-click menu",
        },
        {
            "id": "scroll",
            "keys": ["open_hand_vertical", "scroll_vertical"],
            "icon": "📜",
            "name": "2 Fingers Up / Down",
            "action": "Smooth Scroll",
            "detail": "Index + Middle together moves page",
        },
        {
            "id": "swipe",
            "keys": ["swipe_right", "swipe_left", "swipe_up", "swipe_down", "swipe"],
            "icon": "↔️",
            "name": "Quick Hand Swipe",
            "action": "Switch Window",
            "detail": "Swipe left/right triggers Alt+Tab",
        },
        {
            "id": "fist",
            "keys": ["fist"],
            "icon": "🔒",
            "name": "Fist (Closed Hand)",
            "action": "Pause Cursor",
            "detail": "Freezes cursor safely in place",
        },
    ]

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
        self._running = True
        self._started = False
        self._frame_image: Optional[ImageTk.PhotoImage] = None
        self._frame_lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None

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
        self._guide_cards = {}

    def update_status(self, **kwargs) -> None:
        self._status.update(kwargs)

    def update_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._pending_frame = frame

    def run(self) -> None:
        self._build_ui()
        if self._root:
            self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._running = True
        self._started = True
        self._schedule_refresh()
        try:
            self._root.mainloop()
        except Exception as e:
            log.error(f"Dashboard mainloop error: {e}")
        finally:
            self._running = False

    def close(self) -> None:
        self._running = False
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        if not self._started:
            return True
        return self._running

    @property
    def has_started(self) -> bool:
        return self._started

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = tk.Tk()
        root.title("HandPilot AI — Hand Gesture Controller")
        root.configure(bg=BG_MAIN)
        root.resizable(False, False)

        if os.path.exists("assets/app_icon.ico"):
            try:
                root.iconbitmap("assets/app_icon.ico")
            except Exception:
                pass

        f_title = ("Segoe UI", 13, "bold")
        f_head = ("Segoe UI", 11, "bold")
        f_sub = ("Segoe UI", 9)
        f_bold = ("Segoe UI", 9, "bold")
        f_mono = ("Consolas", 9)
        f_badge = ("Segoe UI", 8, "bold")

        self._root = root

        # === 1. TOP HEADER APP BAR ===
        top_bar = tk.Frame(root, bg="#0f172a", padx=16, pady=10, relief="flat")
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        title_frame = tk.Frame(top_bar, bg="#0f172a")
        title_frame.pack(side="left")
        tk.Label(title_frame, text="✋ HANDPILOT AI", fg=ACCENT_CYAN, bg="#0f172a", font=f_title).pack(side="left")
        tk.Label(title_frame, text="  |  Smooth & Accurate Motion Controller", fg=TEXT_SECONDARY, bg="#0f172a", font=f_sub).pack(side="left")

        # Top Right Status Badges
        top_right = tk.Frame(top_bar, bg="#0f172a")
        top_right.pack(side="right")

        self._pill_tracking = tk.Label(top_right, text="● TRACKING", fg=TEXT_MUTED, bg="#1e293b", font=f_badge, padx=8, pady=3)
        self._pill_tracking.pack(side="left", padx=4)

        self._pill_mode = tk.Label(top_right, text="ACTIVE", fg="#10b981", bg="#064e3b", font=f_badge, padx=8, pady=3)
        self._pill_mode.pack(side="left", padx=4)

        # === 2. MAIN CONTENT SPLIT (Left: Camera + Quick Controls, Right: Gesture Guide) ===
        content_frame = tk.Frame(root, bg=BG_MAIN, padx=12, pady=10)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # --- LEFT COLUMN (Camera + Performance + Controls) ---
        left_col = tk.Frame(content_frame, bg=BG_MAIN)
        left_col.pack(side="left", fill="both", padx=(0, 10))

        # Camera Card
        cam_card = tk.Frame(left_col, bg=CARD_BG, padx=8, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        cam_card.pack(fill="x")

        cam_header = tk.Frame(cam_card, bg=CARD_BG)
        cam_header.pack(fill="x", pady=(0, 6))
        tk.Label(cam_header, text="LIVE CAMERA FEED", fg=TEXT_PRIMARY, bg=CARD_BG, font=f_bold).pack(side="left")
        self._res_lbl = tk.Label(cam_header, text="1280x720 (Mirrored)", fg=TEXT_MUTED, bg=CARD_BG, font=f_mono)
        self._res_lbl.pack(side="right")

        self._cam_label = tk.Label(cam_card, bg="#050811", width=560, height=315)
        self._cam_label.pack()

        # Telemetry Row below camera
        tele_row = tk.Frame(cam_card, bg=CARD_BG, pady=6)
        tele_row.pack(fill="x")

        self._fps_gauge = tk.Label(tele_row, text="FPS: 30.0", fg=ACCENT_GREEN, bg=CARD_BG, font=f_mono)
        self._fps_gauge.pack(side="left", padx=(4, 12))

        self._lat_gauge = tk.Label(tele_row, text="Latency: 12ms", fg=ACCENT_YELLOW, bg=CARD_BG, font=f_mono)
        self._lat_gauge.pack(side="left", padx=12)

        self._cursor_gauge = tk.Label(tele_row, text="Cursor: (0, 0)", fg=ACCENT_CYAN, bg=CARD_BG, font=f_mono)
        self._cursor_gauge.pack(side="right", padx=4)

        # Controls & Sliders Card
        ctrl_card = tk.Frame(left_col, bg=CARD_BG, padx=10, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        ctrl_card.pack(fill="x", pady=(8, 0))

        # Action Buttons Row
        btn_row = tk.Frame(ctrl_card, bg=CARD_BG)
        btn_row.pack(fill="x", pady=(0, 8))

        self._btn_enable = tk.Button(
            btn_row, text="⏵ Enable Control", bg="#059669", fg="white", font=f_bold,
            relief="flat", padx=12, pady=4, cursor="hand2", command=self._on_enable_click
        )
        self._btn_enable.pack(side="left", padx=(0, 6))

        self._btn_disable = tk.Button(
            btn_row, text="⏸ Pause", bg="#d97706", fg="white", font=f_bold,
            relief="flat", padx=12, pady=4, cursor="hand2", command=self._on_disable_click
        )
        self._btn_disable.pack(side="left", padx=(0, 6))

        self._btn_emergency = tk.Button(
            btn_row, text="⛔ Emergency Stop", bg="#dc2626", fg="white", font=f_bold,
            relief="flat", padx=12, pady=4, cursor="hand2", command=self._on_emergency_click
        )
        self._btn_emergency.pack(side="right")

        # Sliders Row
        slider_grid = tk.Frame(ctrl_card, bg=CARD_BG)
        slider_grid.pack(fill="x")

        # Slider 1: Sensitivity
        s1 = tk.Frame(slider_grid, bg=CARD_BG)
        s1.pack(side="left", expand=True, fill="x", padx=4)
        tk.Label(s1, text="Cursor Sensitivity", fg=TEXT_SECONDARY, bg=CARD_BG, font=f_sub).pack(anchor="w")
        self._sensitivity_var = tk.DoubleVar(value=1.5)
        tk.Scale(
            s1, variable=self._sensitivity_var, from_=0.5, to=3.0, resolution=0.1,
            orient="horizontal", bg=CARD_BG, fg=TEXT_PRIMARY, troughcolor="#0f172a",
            highlightthickness=0, command=self._on_sensitivity
        ).pack(fill="x")

        # Slider 2: Scroll Speed
        s2 = tk.Frame(slider_grid, bg=CARD_BG)
        s2.pack(side="left", expand=True, fill="x", padx=4)
        tk.Label(s2, text="Scroll Speed", fg=TEXT_SECONDARY, bg=CARD_BG, font=f_sub).pack(anchor="w")
        self._scroll_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            s2, variable=self._scroll_var, from_=0.2, to=3.0, resolution=0.1,
            orient="horizontal", bg=CARD_BG, fg=TEXT_PRIMARY, troughcolor="#0f172a",
            highlightthickness=0, command=self._on_scroll_speed
        ).pack(fill="x")

        # Slider 3: Pinch Threshold
        s3 = tk.Frame(slider_grid, bg=CARD_BG)
        s3.pack(side="left", expand=True, fill="x", padx=4)
        tk.Label(s3, text="Pinch Distance", fg=TEXT_SECONDARY, bg=CARD_BG, font=f_sub).pack(anchor="w")
        self._threshold_var = tk.DoubleVar(value=0.06)
        tk.Scale(
            s3, variable=self._threshold_var, from_=0.02, to=0.15, resolution=0.005,
            orient="horizontal", bg=CARD_BG, fg=TEXT_PRIMARY, troughcolor="#0f172a",
            highlightthickness=0, command=self._on_threshold
        ).pack(fill="x")

        # --- RIGHT COLUMN (INTERACTIVE GESTURE GUIDE / CHEAT-SHEET) ---
        right_col = tk.Frame(content_frame, bg=CARD_BG, width=380, padx=12, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        right_col.pack(side="right", fill="both", expand=True)
        right_col.pack_propagate(False)

        # Guide Header
        g_header = tk.Frame(right_col, bg=CARD_BG)
        g_header.pack(fill="x", pady=(0, 6))
        tk.Label(g_header, text="📖 GESTURE GUIDE & LIVE SIGNS", fg=ACCENT_CYAN, bg=CARD_BG, font=f_head).pack(side="left")
        tk.Label(g_header, text="(Highlights active sign)", fg=TEXT_MUTED, bg=CARD_BG, font=f_sub).pack(side="right")

        # Cards for each gesture
        cards_container = tk.Frame(right_col, bg=CARD_BG)
        cards_container.pack(fill="both", expand=True)

        for item in self.GESTURE_GUIDE_ITEMS:
            card = tk.Frame(
                cards_container, bg="#0f172a", padx=10, pady=6,
                highlightthickness=1, highlightbackground=BORDER_COLOR
            )
            card.pack(fill="x", pady=3)

            # Left icon
            icon_lbl = tk.Label(card, text=item["icon"], font=("Segoe UI Emoji", 14), bg="#0f172a")
            icon_lbl.pack(side="left", padx=(0, 8))

            # Center text (Name + detail)
            info_frame = tk.Frame(card, bg="#0f172a")
            info_frame.pack(side="left", fill="both", expand=True)

            name_lbl = tk.Label(info_frame, text=item["name"], fg=TEXT_PRIMARY, bg="#0f172a", font=f_bold, anchor="w")
            name_lbl.pack(anchor="w")

            detail_lbl = tk.Label(info_frame, text=f"{item['action']} — {item['detail']}", fg=TEXT_SECONDARY, bg="#0f172a", font=f_sub, anchor="w")
            detail_lbl.pack(anchor="w")

            # Right badge (Active status pill)
            badge_lbl = tk.Label(card, text="IDLE", fg=TEXT_MUTED, bg="#1e293b", font=f_badge, padx=6, pady=2)
            badge_lbl.pack(side="right")

            self._guide_cards[item["id"]] = {
                "frame": card,
                "icon": icon_lbl,
                "info": info_frame,
                "name": name_lbl,
                "detail": detail_lbl,
                "badge": badge_lbl,
                "keys": item["keys"],
                "is_active": False,
            }

        # Footer hotkeys
        footer = tk.Frame(right_col, bg=CARD_BG, pady=6)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, text="Hotkeys: [Ctrl+Alt+G] Toggle  |  [Ctrl+Alt+X] Stop", fg=TEXT_MUTED, bg=CARD_BG, font=f_sub).pack()

    # ------------------------------------------------------------------
    # Refresh Loop
    # ------------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        if not self._running:
            return
        try:
            self._refresh()
        except Exception as e:
            log.debug(f"Dashboard refresh error: {e}")
        finally:
            if self._running and self._root:
                try:
                    self._root.after(33, self._schedule_refresh)
                except Exception:
                    pass

    def _refresh(self) -> None:
        s = self._status

        # Update gauges
        fps_val = s.get("fps", 0.0)
        lat_val = s.get("latency", 0.0)
        self._fps_gauge.configure(
            text=f"FPS: {fps_val:.1f}",
            fg=ACCENT_GREEN if fps_val >= 24 else ACCENT_YELLOW if fps_val >= 15 else ACCENT_RED
        )
        self._lat_gauge.configure(
            text=f"Latency: {lat_val:.0f}ms",
            fg=ACCENT_GREEN if lat_val < 35 else ACCENT_YELLOW if lat_val < 80 else ACCENT_RED
        )
        self._cursor_gauge.configure(
            text=f"Cursor: ({s.get('cursor_x', 0)}, {s.get('cursor_y', 0)})"
        )

        # Update top status pills
        has_tracking = s.get("tracking", False)
        is_enabled = s.get("enabled", True)

        if has_tracking:
            self._pill_tracking.configure(text="● HAND TRACKED", fg="#10b981", bg="#064e3b")
        else:
            self._pill_tracking.configure(text="○ NO HAND", fg=TEXT_MUTED, bg="#1e293b")

        if is_enabled:
            self._pill_mode.configure(text="CONTROLLER ON", fg="#10b981", bg="#064e3b")
        else:
            self._pill_mode.configure(text="CONTROLLER OFF", fg="#f59e0b", bg="#451a03")

        # === DYNAMIC GESTURE GUIDE HIGHLIGHTING ===
        curr_gesture = str(s.get("gesture", "")).lower()
        curr_action = str(s.get("action", "")).lower()

        for item_id, card_data in self._guide_cards.items():
            matches = any(k in curr_gesture or k in curr_action for k in card_data["keys"])

            if matches and has_tracking and is_enabled:
                if not card_data["is_active"]:
                    card_data["is_active"] = True
                    card_data["frame"].configure(bg="#042f2e", highlightbackground="#14b8a6", highlightthickness=2)
                    card_data["icon"].configure(bg="#042f2e")
                    card_data["info"].configure(bg="#042f2e")
                    card_data["name"].configure(bg="#042f2e", fg="#5eead4")
                    card_data["detail"].configure(bg="#042f2e", fg="#99f6e4")
                    card_data["badge"].configure(text="ACTIVE", fg="#14b8a6", bg="#134e4a")
            else:
                if card_data["is_active"]:
                    card_data["is_active"] = False
                    card_data["frame"].configure(bg="#0f172a", highlightbackground=BORDER_COLOR, highlightthickness=1)
                    card_data["icon"].configure(bg="#0f172a")
                    card_data["info"].configure(bg="#0f172a")
                    card_data["name"].configure(bg="#0f172a", fg=TEXT_PRIMARY)
                    card_data["detail"].configure(bg="#0f172a", fg=TEXT_SECONDARY)
                    card_data["badge"].configure(text="IDLE", fg=TEXT_MUTED, bg="#1e293b")

        # Update live camera preview
        with self._frame_lock:
            frame = self._pending_frame
            self._pending_frame = None

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                dw, dh = 560, 315
                scale = min(dw / w, dh / h)
                nw, nh = int(w * scale), int(h * scale)
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(rgb)
                self._frame_image = ImageTk.PhotoImage(img)
                self._cam_label.configure(image=self._frame_image, width=dw, height=dh)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Control Callbacks
    # ------------------------------------------------------------------

    def _on_enable_click(self) -> None:
        self._status["enabled"] = True
        self._on_enable()

    def _on_disable_click(self) -> None:
        self._status["enabled"] = False
        self._on_disable()

    def _on_emergency_click(self) -> None:
        self._status["enabled"] = False
        self._on_emergency()

    def _on_sensitivity(self, val) -> None:
        self._on_sensitivity_change(float(val))

    def _on_scroll_speed(self, val) -> None:
        self._on_scroll_speed_change(float(val))

    def _on_threshold(self, val) -> None:
        self._on_threshold_change(float(val))
