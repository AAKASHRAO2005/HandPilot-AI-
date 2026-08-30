"""
vision/overlay.py — Draws hand landmarks, bounding boxes, and HUD on camera frames.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from vision.hand_tracker import HandResult
from vision.landmarks import FINGERTIPS, WRIST, FINGER_MCPS, lm_xy
from gestures.gesture_engine import GestureEvent


# Colour palette (BGR)
COL_BOX = (124, 58, 237)        # Purple bounding box
COL_LANDMARK = (6, 214, 160)    # Teal landmark dots
COL_FINGERTIP = (255, 100, 50)  # Orange fingertips
COL_PALM = (255, 255, 100)      # Yellow palm
COL_SKELETON = (80, 80, 200)    # Blue-ish skeleton lines
COL_TEXT = (230, 230, 230)
COL_HUD_BG = (20, 20, 30)
COL_GREEN = (34, 197, 94)
COL_RED = (239, 68, 68)
COL_YELLOW = (234, 179, 8)
COL_ACCENT = (124, 58, 237)

# MediaPipe hand connection pairs (subset for clear visualisation)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),          # Palm cross connections
]


def draw_hands(
    frame: np.ndarray,
    hands: List[HandResult],
    event: Optional[GestureEvent] = None,
) -> np.ndarray:
    """
    Draw all hand annotations onto *frame*.

    Args:
        frame: BGR frame to annotate (modified in-place and returned).
        hands: List of HandResult objects.
        event: Current GestureEvent for HUD display.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    for hand in hands:
        lms = hand.landmarks
        bbox = hand.bbox

        # Bounding box
        cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                      COL_BOX, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{hand.confidence:.2f}", (bbox.x1, bbox.y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_BOX, 1, cv2.LINE_AA)

        # Convert normalized landmarks to pixel coords
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Skeleton lines
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], COL_SKELETON, 1, cv2.LINE_AA)

        # Landmark dots
        for i, pt in enumerate(pts):
            color = COL_FINGERTIP if i in FINGERTIPS else COL_LANDMARK
            radius = 5 if i in FINGERTIPS else 3
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

        # Palm center highlight
        if len(pts) > 9:
            palm_pts = [pts[idx] for idx in FINGER_MCPS if idx < len(pts)]
            if palm_pts:
                cx = int(sum(p[0] for p in palm_pts) / len(palm_pts))
                cy = int(sum(p[1] for p in palm_pts) / len(palm_pts))
                cv2.circle(frame, (cx, cy), 8, COL_PALM, 2, cv2.LINE_AA)

    # === HUD overlay ===
    _draw_hud(frame, event)
    return frame


def _draw_hud(frame: np.ndarray, event: Optional[GestureEvent]) -> None:
    """Draw the top-left HUD panel."""
    h, w = frame.shape[:2]

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 170), COL_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    lines = []
    if event:
        fps_col = COL_GREEN if event.fps >= 25 else COL_YELLOW if event.fps >= 15 else COL_RED
        lat_col = COL_GREEN if event.latency_ms < 50 else COL_YELLOW if event.latency_ms < 100 else COL_RED

        lines = [
            (f"FPS: {event.fps:.1f}", fps_col),
            (f"Latency: {event.latency_ms:.0f} ms", lat_col),
            ("", COL_TEXT),
            (f"Gesture: {event.gesture.upper()}", COL_ACCENT),
            (f"Action:  {event.action}", COL_TEXT),
            (f"State:   {event.state}", COL_TEXT),
            ("", COL_TEXT),
            (f"Cursor: ({event.cursor_x}, {event.cursor_y})", COL_TEXT),
            (f"Conf: {event.confidence:.2f}  Fingers: {event.extended_fingers}", COL_TEXT),
        ]
        if event.is_dragging:
            lines.append(("🖱  DRAGGING", COL_RED))
    else:
        lines = [("Initializing…", COL_TEXT)]

    y = 18
    for text, color in lines:
        if text:
            cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        y += 16

    # Gesture control enabled indicator (top-right)
    if event:
        enabled = event.gesture != "disabled"
        indicator_text = "GESTURE ON" if enabled else "GESTURE OFF"
        indicator_color = COL_GREEN if enabled else COL_RED
        text_size = cv2.getTextSize(indicator_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = w - text_size[0] - 12
        cv2.putText(frame, indicator_text, (tx, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, indicator_color, 2, cv2.LINE_AA)
