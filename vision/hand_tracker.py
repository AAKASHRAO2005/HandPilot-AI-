"""
vision/hand_tracker.py — Hand landmark detection using TFLite (ai-edge-litert).

Fully compatible with Python 3.14+. Uses the official MediaPipe hand landmark
TFLite model extracted from hand_landmarker.task.

Pipeline per frame:
    1. YOLO26 bounding box → crop + pad ROI
    2. Preprocess ROI to 224×224 float32
    3. Run hand_landmarks_detector.tflite
    4. Re-map 21 landmarks to full-frame normalised coords
    5. Return HandResult list

Model outputs (hand_landmarks_detector.tflite):
    Identity   [1, 63]  — 21 × (x, y, z) in model input space [0, 224]
    Identity_1 [1, 1]   — hand presence score [0, 1]
    Identity_2 [1, 1]   — handedness score (>0.5 = right hand)
    Identity_3 [1, 63]  — world landmarks (unused here)
"""

import os
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from vision.yolo_detector import Detection, YOLODetector
from utils.logger import setup_logger

log = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
TASK_FILE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
TASK_FILE_PATH = "hand_landmarker.task"
LANDMARK_MODEL_PATH = "hand_landmarks_detector.tflite"

INPUT_SIZE = 224  # Model expects 224×224 input


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    """Complete result for one detected hand in a frame."""
    bbox: Detection
    landmarks: list   # 21 _Landmark objects (.x .y .z normalised 0-1 in full frame)
    handedness: str = "Unknown"
    confidence: float = 1.0

    @property
    def landmark_list(self) -> list:
        return self.landmarks


class _Landmark:
    """Lightweight landmark with normalised coordinates."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Hand landmark tracker: YOLO26 detection + TFLite hand_landmarks_detector.

    Args:
        yolo: Configured YOLODetector instance.
        max_hands: Max hands to track per frame.
        model_complexity: Ignored (kept for API compat).
        min_detection_confidence: Minimum hand-presence score to accept.
        min_tracking_confidence: Ignored (kept for API compat).
        model_path: Path to hand_landmarks_detector.tflite.
    """

    def __init__(
        self,
        yolo: YOLODetector,
        max_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = LANDMARK_MODEL_PATH,
        landmark_smoothing: float = 0.50,  # EMA alpha: 0=max smooth, 1=no smooth
    ) -> None:
        self._yolo = yolo
        self._max_hands = max_hands
        self._min_conf = min_detection_confidence
        self._lm_alpha = max(0.05, min(1.0, landmark_smoothing))

        # Temporal smoothing state: previous landmark positions [21 x 2] per hand slot
        self._prev_lm: List[Optional[np.ndarray]] = [None] * max_hands
        self._frames_without_hand = 0
        self._reset_after_miss = 5  # frames with no hand before resetting smoother

        # Ensure TFLite model exists
        if not os.path.exists(model_path):
            self._ensure_model(model_path)

        # Load TFLite interpreter
        from ai_edge_litert.interpreter import Interpreter
        self._interp = Interpreter(model_path=model_path)
        self._interp.allocate_tensors()

        self._input_details = self._interp.get_input_details()
        self._output_details = self._interp.get_output_details()

        # Identify output tensor indices by shape
        self._lm_idx = None        # [1, 63] — landmarks
        self._presence_idx = None  # [1, 1]  — presence score
        self._handed_idx = None    # [1, 1]  — handedness

        for det in self._output_details:
            shape = tuple(det["shape"])
            if shape == (1, 63) and self._lm_idx is None:
                self._lm_idx = det["index"]
            elif shape == (1, 1) and self._presence_idx is None:
                self._presence_idx = det["index"]
            elif shape == (1, 1) and self._handed_idx is None:
                self._handed_idx = det["index"]

        log.info(f"TFLite hand landmark model loaded: {model_path}")
        log.info(f"  lm_idx={self._lm_idx}, presence_idx={self._presence_idx}, smooth_alpha={self._lm_alpha}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> List[HandResult]:
        """
        Process one BGR frame, return list of HandResult (may be empty).
        Applies temporal EMA smoothing on all 21 landmarks to reduce jitter.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]

        # Step 1: YOLO hand bounding boxes
        detections = self._yolo.detect(frame)
        if not detections:
            self._frames_without_hand += 1
            if self._frames_without_hand >= self._reset_after_miss:
                self._prev_lm = [None] * self._max_hands
            return []

        self._frames_without_hand = 0
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        detections = detections[: self._max_hands]

        results: List[HandResult] = []

        for slot, det in enumerate(detections):
            # Step 2: Square-crop ROI with generous padding
            pad = int(max(det.width, det.height) * 0.35)
            x1 = max(0, det.x1 - pad)
            y1 = max(0, det.y1 - pad)
            x2 = min(w, det.x2 + pad)
            y2 = min(h, det.y2 + pad)

            cw, ch = x2 - x1, y2 - y1
            side = max(cw, ch)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_h, roi_w = roi.shape[:2]

            # Step 3: Preprocess → [1, 224, 224, 3] float32
            resized = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)

            # Step 4: TFLite inference
            self._interp.set_tensor(self._input_details[0]["index"], inp)
            self._interp.invoke()

            # Step 5: Read outputs
            lm_raw   = self._interp.get_tensor(self._lm_idx)[0]           # [63]
            presence = float(self._interp.get_tensor(self._presence_idx)[0][0])

            if presence < self._min_conf:
                continue

            handedness = "Unknown"
            if self._handed_idx is not None:
                score = float(self._interp.get_tensor(self._handed_idx)[0][0])
                handedness = "Right" if score > 0.5 else "Left"

            # Step 6: Map [0, INPUT_SIZE] → normalised full-frame coords
            raw_xy = np.zeros((21, 2), dtype=np.float32)
            for i in range(21):
                mx = lm_raw[i * 3 + 0] / INPUT_SIZE
                my = lm_raw[i * 3 + 1] / INPUT_SIZE
                # Clamp to [0,1] to avoid out-of-bound landmarks
                raw_xy[i, 0] = np.clip((x1 + mx * roi_w) / w, 0.0, 1.0)
                raw_xy[i, 1] = np.clip((y1 + my * roi_h) / h, 0.0, 1.0)

            # Step 7: Temporal EMA smoothing
            # alpha=0.5: new=50% raw + 50% previous → smooth but responsive
            # Use higher alpha on first frame (no history yet)
            prev = self._prev_lm[slot]
            if prev is None:
                smoothed = raw_xy
            else:
                smoothed = self._lm_alpha * raw_xy + (1.0 - self._lm_alpha) * prev
            self._prev_lm[slot] = smoothed

            # Step 8: Build landmark objects
            wrapped = []
            for i in range(21):
                mz = float(lm_raw[i * 3 + 2]) / INPUT_SIZE
                wrapped.append(_Landmark(
                    float(smoothed[i, 0]),
                    float(smoothed[i, 1]),
                    mz,
                ))

            results.append(HandResult(
                bbox=det,
                landmarks=wrapped,
                handedness=handedness,
                confidence=min(1.0, max(0.0, presence)),
            ))

        return results

    def close(self) -> None:
        log.info("HandTracker closed.")

    # ------------------------------------------------------------------
    # Model provisioning
    # ------------------------------------------------------------------

    def _ensure_model(self, model_path: str) -> None:
        """Download and extract model if not present."""
        if not os.path.exists(TASK_FILE_PATH):
            log.info(f"Downloading hand_landmarker.task from Google Storage…")
            urllib.request.urlretrieve(TASK_FILE_URL, TASK_FILE_PATH)
            log.info(f"Downloaded {TASK_FILE_PATH}")

        log.info(f"Extracting {model_path} from {TASK_FILE_PATH}…")
        with zipfile.ZipFile(TASK_FILE_PATH, "r") as z:
            # The landmark model is always named hand_landmarks_detector.tflite inside
            z.extract("hand_landmarks_detector.tflite", ".")
        log.info(f"Extracted: {model_path}")
