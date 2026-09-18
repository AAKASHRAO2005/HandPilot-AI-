"""
vision/overlay.py — High-definition visualization of hand landmarks, bounding boxes, and HUD guide.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from vision.hand_tracker import HandResult
from vision.landmarks import FINGERTIPS, WRIST, FINGER_MCPS, THUMB_TIP, INDEX_TIP, MIDDLE_TIP
from gestures.gesture_engine import GestureEvent


# Colour palette (BGR format)
COL_BOX = (237, 58, 124)         # Neon Violet
COL_LANDMARK = (160, 214, 6)     # Neon Cyan
COL_FINGERTIP = (50, 150, 255)   # Glowing Orange
COL_PALM = (100, 255, 255)       # Soft Yellow
COL_SKELETON = (220, 120, 100)   # Light Purple
COL_TEXT = (240, 240, 240)
COL_HUD_BG = (18, 18, 28)
COL_GREEN = (94, 197, 34)
COL_RED = (68, 68, 239)
COL_YELLOW = (8, 179, 234)
COL_ACCENT = (237, 58, 124)
COL_CYAN = (230, 210, 0)

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm knuckles
]

GESTURE_SIGNS = {
    "pinch": ("PINCH (Thumb+Index)", "LEFT CLICK", COL_GREEN),
    "left_click_hold": ("PINCH HOLD", "CLICK & HOLD", COL_YELLOW),
    "drag": ("PINCH HOLD", "DRAGGING", COL_RED),
    "double_pinch": ("DOUBLE PINCH", "DOUBLE CLICK", COL_CYAN),
    "middle_pinch": ("MIDDLE PINCH", "RIGHT CLICK", COL_YELLOW),
    "open_hand_vertical": ("2 FINGERS UP/DOWN", "SCROLLING", COL_GREEN),
    "swipe_right": ("SWIPE RIGHT", "ALT + TAB", COL_CYAN),
    "swipe_left": ("SWIPE LEFT", "ALT + SHIFT + TAB", COL_CYAN),
    "swipe_up": ("SWIPE UP", "MAXIMIZE WINDOW", COL_CYAN),
    "swipe_down": ("SWIPE DOWN", "MINIMIZE WINDOW", COL_CYAN),
    "tracking": ("OPEN PALM / POINT", "MOVE CURSOR", COL_TEXT),
    "no_hand": ("NO HAND", "SEARCHING", (120, 120, 120)),
    "disabled": ("DISABLED", "HOTKEY CTRL+ALT+G", COL_RED),
}


def draw_hands(
    frame: np.ndarray,
    hands: List[HandResult],
    event: Optional[GestureEvent] = None,
) -> np.ndarray:
    """
    Draw clean, antialiased hand skeletons, labels, and gesture indicator onto frame.
    """
    h, w = frame.shape[:2]

    for hand in hands:
        lms = hand.landmarks
        bbox = hand.bbox

        # Sleek corner bounding brackets instead of harsh rectangle
        bx1, by1, bx2, by2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        line_len = min(20, (bx2 - bx1) // 4)
        # Top-left
        cv2.line(frame, (bx1, by1), (bx1 + line_len, by1), COL_BOX, 2, cv2.LINE_AA)
        cv2.line(frame, (bx1, by1), (bx1, by1 + line_len), COL_BOX, 2, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (bx2, by1), (bx2 - line_len, by1), COL_BOX, 2, cv2.LINE_AA)
        cv2.line(frame, (bx2, by1), (bx2, by1 + line_len), COL_BOX, 2, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (bx1, by2), (bx1 + line_len, by2), COL_BOX, 2, cv2.LINE_AA)
        cv2.line(frame, (bx1, by2), (bx1, by2 - line_len), COL_BOX, 2, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (bx2, by2), (bx2 - line_len, by2), COL_BOX, 2, cv2.LINE_AA)
        cv2.line(frame, (bx2, by2), (bx2, by2 - line_len), COL_BOX, 2, cv2.LINE_AA)

        # Convert normalized landmarks to pixel coords
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Skeleton connections
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], COL_SKELETON, 2, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(pts):
            if i in FINGERTIPS:
                cv2.circle(frame, pt, 6, COL_FINGERTIP, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, COL_LANDMARK, -1, cv2.LINE_AA)

        # Finger labels for key action fingers
        if len(pts) > INDEX_TIP:
            ix, iy = pts[INDEX_TIP]
            cv2.putText(frame, "Index", (ix + 8, iy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if len(pts) > THUMB_TIP:
            tx, ty = pts[THUMB_TIP]
            cv2.putText(frame, "Thumb", (tx + 8, ty - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # === HUD & Gesture Sign Guide ===
    _draw_hud(frame, event)
    _draw_active_sign_banner(frame, event)
    return frame


def _draw_hud(frame: np.ndarray, event: Optional[GestureEvent]) -> None:
    """Draw top-left status HUD."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (270, 125), COL_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (8, 8), (270, 125), (60, 60, 80), 1, cv2.LINE_AA)

    if event:
        fps_col = COL_GREEN if event.fps >= 25 else COL_YELLOW if event.fps >= 15 else COL_RED
        lat_col = COL_GREEN if event.latency_ms < 40 else COL_YELLOW if event.latency_ms < 80 else COL_RED

        lines = [
            (f"FPS: {event.fps:.1f}   |   Latency: {event.latency_ms:.0f} ms", fps_col),
            (f"Tracking: {'HAND DETECTED' if event.confidence > 0 else 'NO HAND'}", COL_GREEN if event.confidence > 0 else (140, 140, 140)),
            (f"Cursor: ({event.cursor_x}, {event.cursor_y})", COL_TEXT),
            (f"Fingers Extended: {event.extended_fingers}/5", COL_TEXT),
        ]
    else:
        lines = [("Initializing Camera & Vision…", COL_TEXT)]

    y = 28
    for text, color in lines:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 22


def _draw_active_sign_banner(frame: np.ndarray, event: Optional[GestureEvent]) -> None:
    """Draw a bottom-center active gesture guide banner directly on screen."""
    h, w = frame.shape[:2]

    raw_gesture = event.gesture if event else "no_hand"
    sign_info = GESTURE_SIGNS.get(raw_gesture, (raw_gesture.upper(), event.action.upper() if event else "—", COL_ACCENT))

    sign_name, action_desc, sign_color = sign_info

    # Draw bottom banner
    bw, bh = 420, 50
    bx = (w - bw) // 2
    by = h - bh - 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), sign_color, 2, cv2.LINE_AA)

    # Indicator light
    cv2.circle(frame, (bx + 20, by + bh // 2), 6, sign_color, -1, cv2.LINE_AA)

    text_main = f"SIGN: {sign_name}"
    text_sub = f"->  {action_desc}"
    cv2.putText(frame, text_main, (bx + 35, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_sub, (bx + 35, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sign_color, 1, cv2.LINE_AA)
