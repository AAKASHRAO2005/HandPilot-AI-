"""
vision/landmarks.py — MediaPipe landmark utilities.

Provides helper functions to extract meaningful positions, distances, and hand scale
from the 21-point MediaPipe hand landmark set.

MediaPipe landmark indices:
    0  = WRIST
    1  = THUMB_CMC       2  = THUMB_MCP      3  = THUMB_IP      4  = THUMB_TIP
    5  = INDEX_FINGER_MCP  6 = INDEX_FINGER_PIP  7 = INDEX_FINGER_DIP  8 = INDEX_FINGER_TIP
    9  = MIDDLE_FINGER_MCP 10= MIDDLE_FINGER_PIP 11= MIDDLE_FINGER_DIP 12= MIDDLE_FINGER_TIP
    13 = RING_FINGER_MCP  14 = RING_FINGER_PIP  15 = RING_FINGER_DIP  16 = RING_FINGER_TIP
    17 = PINKY_MCP        18 = PINKY_PIP        19 = PINKY_DIP        20 = PINKY_TIP
"""

import math
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Fingertip index for each finger (ordered: thumb, index, middle, ring, pinky)
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
# MCP (knuckle) for each finger — used for extension check
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
# PIP joint for each finger
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


# ---------------------------------------------------------------------------
# Type alias: landmark list from mediapipe (each has .x, .y, .z all in [0,1])
# ---------------------------------------------------------------------------
LandmarkList = list


def lm_xy(landmarks: LandmarkList, idx: int) -> Tuple[float, float]:
    """Return normalized (x, y) of landmark *idx*."""
    lm = landmarks[idx]
    return (lm.x, lm.y)


def lm_distance(landmarks: LandmarkList, idx_a: int, idx_b: int) -> float:
    """Euclidean distance between two landmarks (normalized coords)."""
    ax, ay = lm_xy(landmarks, idx_a)
    bx, by = lm_xy(landmarks, idx_b)
    return math.hypot(ax - bx, ay - by)


def get_fingertip(landmarks: LandmarkList, finger: int = 1) -> Tuple[float, float]:
    """
    Return normalized (x, y) of the fingertip.

    Args:
        finger: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    """
    return lm_xy(landmarks, FINGERTIPS[finger])


def get_palm_center(landmarks: LandmarkList) -> Tuple[float, float]:
    """Return normalized (x, y) of the palm center (average of MCP joints)."""
    xs = [landmarks[idx].x for idx in FINGER_MCPS]
    ys = [landmarks[idx].y for idx in FINGER_MCPS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def get_hand_scale(landmarks: LandmarkList) -> float:
    """
    Compute reference anatomical scale of the hand.
    Uses distance from WRIST to MIDDLE_MCP + palm width across knuckles.
    Guaranteed > 1e-4 to prevent division by zero.
    """
    palm_length = lm_distance(landmarks, WRIST, MIDDLE_MCP)
    palm_width = lm_distance(landmarks, INDEX_MCP, PINKY_MCP)
    scale = (palm_length * 0.7 + palm_width * 0.3)
    return max(0.01, scale)


def pinch_distance(landmarks: LandmarkList, finger_a: int, finger_b: int) -> float:
    """
    Raw Euclidean distance between two fingertips (normalized coords [0, 1]).

    finger_a, finger_b: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    """
    return lm_distance(landmarks, FINGERTIPS[finger_a], FINGERTIPS[finger_b])


def normalized_pinch_distance(
    landmarks: LandmarkList,
    finger_a: int,
    finger_b: int,
) -> float:
    """
    Scale-invariant distance between two fingertips.
    Normalizes raw distance by palm scale so pinch thresholds remain constant
    regardless of distance between hand and camera.
    A typical pinch yields <= 0.25 in normalized ratio.
    """
    raw_dist = pinch_distance(landmarks, finger_a, finger_b)
    scale = get_hand_scale(landmarks)
    return raw_dist / scale


def is_finger_extended(landmarks: LandmarkList, finger: int) -> bool:
    """
    Heuristic to check if a finger is extended (roughly straight).

    Compares fingertip distance from wrist vs PIP distance from wrist.
    This works reliably at any hand orientation/tilt, unlike simple y-axis comparison.
    """
    tip_dist = lm_distance(landmarks, WRIST, FINGERTIPS[finger])
    pip_dist = lm_distance(landmarks, WRIST, FINGER_PIPS[finger])

    if finger == 0:
        # Thumb: tip further from pinky MCP than thumb MCP
        tip_pinky = lm_distance(landmarks, PINKY_MCP, THUMB_TIP)
        mcp_pinky = lm_distance(landmarks, PINKY_MCP, THUMB_MCP)
        return tip_pinky > mcp_pinky * 1.1
    else:
        # Fingers: tip further from wrist than PIP joint
        return tip_dist > pip_dist * 1.12


def count_extended_fingers(landmarks: LandmarkList) -> int:
    """Return count of extended fingers (0–5)."""
    return sum(is_finger_extended(landmarks, i) for i in range(5))


def is_fist(landmarks: LandmarkList, threshold: int = 1) -> bool:
    """True if ≤ threshold fingers are extended."""
    return count_extended_fingers(landmarks) <= threshold


def is_open_palm(landmarks: LandmarkList, threshold: int = 4) -> bool:
    """True if ≥ threshold fingers are extended."""
    return count_extended_fingers(landmarks) >= threshold


def hand_velocity(
    prev_pos: Tuple[float, float],
    curr_pos: Tuple[float, float],
) -> Tuple[float, float]:
    """Return (vx, vy) as difference between two normalized positions."""
    return (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
