"""
app.py — Hand Gesture Desktop Controller — Main entry point.

Threading model (fixes Tcl_AsyncDelete crash):
    MAIN THREAD     → tkinter Dashboard (mandatory for Tcl/Tk)
    PROCESSING THREAD → camera → YOLO → TFLite → gesture → input
    CAMERA THREAD   → CameraCapture (always runs latest frame only)
    WEBSOCKET THREAD → asyncio WebSocket server
"""

import sys
import os
import threading
import time
import signal
from typing import Optional

import cv2
import yaml

from vision.camera import CameraCapture
from vision.yolo_detector import YOLODetector
from vision.hand_tracker import HandTracker
from vision.overlay import draw_hands
from gestures.gesture_engine import GestureEngine, GestureEvent
from controller.mouse import MouseController
from controller.scroll import ScrollController
from controller.keyboard import KeyboardController
from controller import windows_input as wi
from communication.websocket_server import WebSocketServer
from ui.dashboard import Dashboard
from utils.fps import FPSCounter, LatencyTracker
from utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log.info(f"Config loaded from {path}")
    return cfg


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------

class HandGestureController:
    """
    Top-level application orchestrator.

    Threading model:
        - Main thread runs tkinter (Dashboard). This is required by Tcl/Tk.
        - A dedicated 'processing' thread runs the camera→YOLO→TFLite→gesture pipeline.
        - Camera capture runs in its own thread with a size-1 queue (always latest frame).
        - WebSocket server runs in an asyncio thread.
    """

    def __init__(self) -> None:
        self._cfg: dict = {}
        self._camera: Optional[CameraCapture] = None
        self._tracker: Optional[HandTracker] = None
        self._engine: Optional[GestureEngine] = None
        self._ws: Optional[WebSocketServer] = None
        self._dashboard: Optional[Dashboard] = None
        self._hotkey_listener = None
        self._running = False
        self._proc_thread: Optional[threading.Thread] = None
        self._fps = FPSCounter(window=30)
        self._lat = LatencyTracker()
        self._last_event: Optional[GestureEvent] = None
        # Shared annotated frame for dashboard preview (written by proc thread)
        self._annotated_frame = None
        self._frame_lock = threading.Lock()

    def start(self) -> None:
        cfg = load_config()
        self._cfg = cfg

        log.info("=" * 60)
        log.info("  Hand Gesture Desktop Controller starting…")
        log.info("=" * 60)

        cam_cfg   = cfg.get("camera", {})
        yolo_cfg  = cfg.get("yolo", {})
        mp_cfg    = cfg.get("mediapipe", {})
        cur_cfg   = cfg.get("cursor", {})
        scr_cfg   = cfg.get("scroll", {})
        kb_cfg    = cfg.get("keyboard", {})
        ws_cfg    = cfg.get("websocket", {})
        dash_cfg  = cfg.get("dashboard", {})
        safety_cfg= cfg.get("safety", {})

        # --- Camera (queue_size=1 → always latest frame only) ---
        self._camera = CameraCapture(
            camera_index=cam_cfg.get("index", 0),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
            mirror=cam_cfg.get("mirror", True),
            queue_size=1,   # ← key: always discard old, keep newest
        )
        if not self._camera.start():
            log.error("Failed to open camera. Check config.yaml → camera.index")
            sys.exit(1)

        # --- YOLO ---
        yolo = YOLODetector(
            model_path=yolo_cfg.get("model_path", "hand_yolo26.pt"),
            fallback_model=yolo_cfg.get("fallback_model", "yolo26n.pt"),
            confidence=yolo_cfg.get("confidence", 0.40),
            iou=yolo_cfg.get("iou", 0.40),
            device=yolo_cfg.get("device", "auto"),
            input_size=yolo_cfg.get("input_size", 320),
            detect_every_n=yolo_cfg.get("detect_every_n", 3),
            infer_width=yolo_cfg.get("infer_width", 640),
        )

        # --- Hand Tracker (TFLite, Python 3.14 compatible) ---
        self._tracker = HandTracker(
            yolo=yolo,
            max_hands=mp_cfg.get("max_num_hands", 1),
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
            model_path=mp_cfg.get("model_path", "hand_landmarks_detector.tflite"),
        )

        # --- Controllers ---
        scroll_cfg = cfg.get("scroll", {})
        scroll_cfg_inner = scroll_cfg if isinstance(scroll_cfg, dict) else {}
        mouse = MouseController(
            screen_w=wi.SCREEN_W,
            screen_h=wi.SCREEN_H,
            alpha=cur_cfg.get("smoothing", 0.20),
            dead_zone=cur_cfg.get("dead_zone", 8),
            sensitivity=cur_cfg.get("sensitivity", 1.5),
            active_region=cur_cfg.get("active_region", {}),
            boundary_margin=cur_cfg.get("boundary_margin", 10),
            max_jump=cur_cfg.get("max_jump", 200),
            mirror=cam_cfg.get("mirror", True),
        )
        scroll = ScrollController(
            speed_multiplier=scroll_cfg_inner.get("speed_multiplier", 1.0),
            velocity_scale=scroll_cfg_inner.get("velocity_scale", 8.0),
            min_velocity=scroll_cfg_inner.get("min_velocity", 0.005),
        )
        keyboard = KeyboardController(gesture_map=kb_cfg)

        # --- Gesture Engine ---
        self._engine = GestureEngine(
            cfg=cfg,
            mouse=mouse,
            scroll=scroll,
            keyboard=keyboard,
        )

        # --- WebSocket Server ---
        if ws_cfg.get("enabled", True):
            self._ws = WebSocketServer(
                host=ws_cfg.get("host", "127.0.0.1"),
                port=ws_cfg.get("port", 8765),
            )
            self._ws.start()

        # --- Global Hotkeys ---
        self._register_hotkeys(safety_cfg)

        # --- Launch processing thread ---
        self._running = True
        self._proc_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="GestureProcessing",
        )
        self._proc_thread.start()

        # --- Dashboard runs on MAIN thread (required by Tcl/Tk) ---
        if dash_cfg.get("enabled", True):
            self._dashboard = Dashboard(
                on_enable=self._engine.enable,
                on_disable=self._engine.disable,
                on_emergency=self._engine.emergency_stop,
                on_camera_change=self._camera.switch_camera,
                on_sensitivity_change=lambda v: self._engine.update_cursor_config(alpha=v / 5),
                on_scroll_speed_change=lambda v: self._engine.update_scroll_config(speed_multiplier=v),
                on_threshold_change=self._update_pinch_threshold,
                initial_camera=cam_cfg.get("index", 0),
            )
            # This blocks until dashboard is closed (it runs tkinter mainloop)
            self._dashboard.run()
        else:
            # No dashboard — just wait for processing thread
            self._proc_thread.join()

        self._running = False
        self._shutdown()

    # ------------------------------------------------------------------
    # Processing loop (runs in background thread)
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """
        Camera → YOLO → TFLite → gesture → input.
        Runs in a dedicated thread so the main thread is free for tkinter.
        Always processes only the latest frame (queue_size=1 in camera).
        """
        show_window = self._cfg.get("dashboard", {}).get("show_camera", True)
        window_name = "Hand Gesture Controller — Camera"

        # OpenCV window must be created and shown in THIS thread
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

        log.info("System ready. Starting processing loop…")

        while self._running:
            self._lat.start()

            # 1. Always drain to get the newest frame
            frame = self._camera.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            # 2. Detect hands + landmarks
            hand_results = self._tracker.process(frame)

            # 3. Gesture → controller actions
            self._fps.tick()
            fps_val = self._fps.get()

            event = self._engine.process(
                hand_results=hand_results,
                fps=fps_val,
                latency_ms=self._lat.last_ms,
            )
            self._last_event = event
            latency = self._lat.stop()
            event.latency_ms = latency

            # 4. Draw overlay (always use live frame — never freeze dashboard)
            annotated = draw_hands(frame.copy(), hand_results, event)

            # 5. Push to dashboard (thread-safe)
            if self._dashboard:
                self._dashboard.update_frame(annotated)
                self._dashboard.update_status(
                    camera=self._camera.connected,
                    yolo=True,
                    tracking=len(hand_results) > 0,
                    enabled=self._engine.is_enabled,
                    gesture=event.gesture,
                    action=event.action,
                    state=event.state,
                    fps=fps_val,
                    latency=latency,
                    cursor_x=event.cursor_x,
                    cursor_y=event.cursor_y,
                    confidence=event.confidence,
                    extended=event.extended_fingers,
                    ws_clients=self._ws.client_count if self._ws else 0,
                )

            # 6. WebSocket broadcast
            if self._ws:
                self._ws.broadcast({
                    "type": "gesture_event",
                    "gesture": event.gesture,
                    "action": event.action,
                    "state": event.state,
                    "cursor_x": event.cursor_x,
                    "cursor_y": event.cursor_y,
                    "fps": round(fps_val, 1),
                    "latency_ms": round(latency, 1),
                    "confidence": round(event.confidence, 3),
                    "is_dragging": event.is_dragging,
                    "extended_fingers": event.extended_fingers,
                })

            # 7. Show OpenCV window
            if show_window:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:    # ESC
                    log.info("ESC — shutting down.")
                    self._running = False
                elif key == ord('g'):
                    self._engine.toggle()
                elif key == ord('x'):
                    self._engine.emergency_stop()

            # 8. Stop if dashboard was closed
            if self._dashboard and not self._dashboard.is_running:
                log.info("Dashboard closed — shutting down.")
                self._running = False

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        log.info("Shutting down…")
        wi.release_all_buttons()
        if self._camera:
            self._camera.stop()
        if self._tracker:
            self._tracker.close()
        # Stop the async YOLO background thread
        if self._tracker and hasattr(self._tracker, '_yolo') and self._tracker._yolo:
            try:
                self._tracker._yolo.stop()
            except Exception:
                pass
        if self._ws:
            self._ws.stop()
        if self._hotkey_listener:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
        log.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Hotkeys
    # ------------------------------------------------------------------

    def _register_hotkeys(self, safety_cfg: dict) -> None:
        try:
            from pynput import keyboard as pynput_kb

            hotkey_enable = safety_cfg.get("hotkey_enable", "ctrl+alt+g")
            hotkey_stop   = safety_cfg.get("hotkey_emergency", "ctrl+alt+x")

            def _parse(combo: str):
                parts = combo.lower().split("+")
                keys = set()
                for p in parts:
                    p = p.strip()
                    if p == "ctrl":
                        keys.add(pynput_kb.Key.ctrl_l)
                    elif p == "alt":
                        keys.add(pynput_kb.Key.alt_l)
                    elif p == "shift":
                        keys.add(pynput_kb.Key.shift_l)
                    elif len(p) == 1:
                        keys.add(pynput_kb.KeyCode(char=p))
                return frozenset(keys)

            _current_keys = set()
            _enable_combo = _parse(hotkey_enable)
            _stop_combo   = _parse(hotkey_stop)

            def on_press(key):
                _current_keys.add(key)
                if _current_keys >= _enable_combo:
                    self._engine.toggle()
                if _current_keys >= _stop_combo:
                    self._engine.emergency_stop()

            def on_release(key):
                _current_keys.discard(key)

            self._hotkey_listener = pynput_kb.Listener(
                on_press=on_press, on_release=on_release
            )
            self._hotkey_listener.start()
            log.info(f"Hotkeys registered: toggle={hotkey_enable}, emergency={hotkey_stop}")

        except Exception as e:
            log.warning(f"Hotkey registration failed: {e}. Use dashboard buttons instead.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_pinch_threshold(self, val: float) -> None:
        if self._engine:
            self._engine._left_pinch.threshold = val
            self._engine._right_pinch.threshold = val


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = HandGestureController()

    def _sigint_handler(sig, frame):
        log.info("Interrupt received — stopping.")
        app._running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        app.start()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt.")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        wi.release_all_buttons()
        sys.exit(1)


if __name__ == "__main__":
    main()
