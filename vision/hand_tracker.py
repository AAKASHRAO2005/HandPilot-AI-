"""
vision/hand_tracker.py — High-accuracy hand landmark detection with continuous landmark-guided ROI tracking & 1€ Filter.

Pipeline per frame:
    1. Check for active tracked hand landmarks from previous frame (high FPS continuous tracking)
    2. Incorporate latest YOLO detections when available to re-anchor / verify
    3. Smooth bounding box ROI to avoid jitter
    4. Run TFLite 21-landmark inference
    5. Apply 21-point PointOneEuroFilter for rock-solid stability
    6. Return HandResult list
"""

import os
import zipfile
import urllib.request
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision.yolo_detector import Detection, YOLODetector
from utils.smoothing import PointOneEuroFilter
from utils.logger import setup_logger

log = setup_logger(__name__)

TASK_FILE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
TASK_FILE_PATH = "hand_landmarker.task"
LANDMARK_MODEL_PATH = "hand_landmarks_detector.tflite"

INPUT_SIZE = 224


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


class _BBoxSmoother:
    """Smooths ROI bounding box to avoid crop jumping across frames."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05) -> None:
        self._tl_filter = PointOneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._br_filter = PointOneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def update(self, x1: int, y1: int, x2: int, y2: int, t: float) -> Tuple[int, int, int, int]:
        sx1, sy1 = self._tl_filter.filter(float(x1), float(y1), t)
        sx2, sy2 = self._br_filter.filter(float(x2), float(y2), t)
        return int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))

    def reset(self) -> None:
        self._tl_filter.reset()
        self._br_filter.reset()


class HandTracker:
    """
    High-accuracy hand tracker with continuous landmark-guided ROI tracking & 1€ Filter.
    """

    def __init__(
        self,
        yolo: YOLODetector,
        max_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = LANDMARK_MODEL_PATH,
        landmark_smoothing: float = 0.50,
        min_cutoff: float = 1.2,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ) -> None:
        self._yolo = yolo
        self._max_hands = max_hands
        self._min_conf = min_detection_confidence

        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff

        self._lm_filters: List[List[PointOneEuroFilter]] = [
            [PointOneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(21)]
            for _ in range(max_hands)
        ]
        self._bbox_smoothers: List[_BBoxSmoother] = [_BBoxSmoother() for _ in range(max_hands)]
        self._prev_landmarks: List[Optional[List[_Landmark]]] = [None] * max_hands
        self._frames_without_hand = 0
        self._reset_after_miss = 5

        # Ensure TFLite model exists
        if not os.path.exists(model_path):
            self._ensure_model(model_path)

        # Load TFLite interpreter
        from ai_edge_litert.interpreter import Interpreter
        self._interp = Interpreter(model_path=model_path)
        self._interp.allocate_tensors()

        self._input_details = self._interp.get_input_details()
        self._output_details = self._interp.get_output_details()

        self._lm_idx = None
        self._presence_idx = None
        self._handed_idx = None

        for det in self._output_details:
            shape = tuple(det["shape"])
            if shape == (1, 63) and self._lm_idx is None:
                self._lm_idx = det["index"]
            elif shape == (1, 1) and self._presence_idx is None:
                self._presence_idx = det["index"]
            elif shape == (1, 1) and self._handed_idx is None:
                self._handed_idx = det["index"]

        log.info(f"TFLite hand landmark model loaded: {model_path} (High-Accuracy Tracker active)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> List[HandResult]:
        """
        Process one BGR frame and return HandResult list.
        Uses continuous landmark-guided ROI tracking for maximum speed and accuracy.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        now = time.perf_counter()

        # Step 1: Candidate ROIs from YOLO or previous frame's landmarks
        yolo_dets = self._yolo.detect(frame)
        candidate_boxes: List[Detection] = []

        # If we had a tracked hand in previous frame, build candidate ROI from it
        if self._prev_landmarks[0] is not None:
            prev_lms = self._prev_landmarks[0]
            xs = [lm.x * w for lm in prev_lms]
            ys = [lm.y * h for lm in prev_lms]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_w = (max_x - min_x) * 0.25
            pad_h = (max_y - min_y) * 0.25
            candidate_boxes.append(Detection(
                x1=int(max(0, min_x - pad_w)),
                y1=int(max(0, min_y - pad_h)),
                x2=int(min(w, max_x + pad_w)),
                y2=int(min(h, max_y + pad_h)),
                confidence=0.95,
            ))

        # If no previous landmarks or YOLO found something strong, incorporate YOLO
        if yolo_dets:
            yolo_sorted = sorted(yolo_dets, key=lambda d: d.confidence, reverse=True)
            if not candidate_boxes:
                candidate_boxes = yolo_sorted[: self._max_hands]
            else:
                # If YOLO detected a hand with high confidence, use it to prevent drift
                best_yolo = yolo_sorted[0]
                if best_yolo.confidence > 0.65:
                    candidate_boxes = [best_yolo]

        if not candidate_boxes:
            self._frames_without_hand += 1
            if self._frames_without_hand >= self._reset_after_miss:
                self._reset_filters()
            return []

        results: List[HandResult] = []

        for slot, det in enumerate(candidate_boxes[: self._max_hands]):
            # Step 2: Smooth bounding box coordinates
            smooth_det_box = self._bbox_smoothers[slot].update(
                det.x1, det.y1, det.x2, det.y2, now
            )
            dx1, dy1, dx2, dy2 = smooth_det_box

            # Step 3: Square-crop ROI with generous margin
            bw, bh = max(20, dx2 - dx1), max(20, dy2 - dy1)
            pad = int(max(bw, bh) * 0.35)
            x1 = max(0, dx1 - pad)
            y1 = max(0, dy1 - pad)
            x2 = min(w, dx2 + pad)
            y2 = min(h, dy2 + pad)

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

            # Step 4: Preprocess → [1, 224, 224, 3] float32
            resized = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)

            # Step 5: TFLite inference
            self._interp.set_tensor(self._input_details[0]["index"], inp)
            self._interp.invoke()

            # Step 6: Read outputs
            lm_raw   = self._interp.get_tensor(self._lm_idx)[0]
            presence = float(self._interp.get_tensor(self._presence_idx)[0][0])

            if presence < self._min_conf:
                self._prev_landmarks[slot] = None
                continue

            self._frames_without_hand = 0

            handedness = "Unknown"
            if self._handed_idx is not None:
                score = float(self._interp.get_tensor(self._handed_idx)[0][0])
                handedness = "Right" if score > 0.5 else "Left"

            # Step 7: Map [0, INPUT_SIZE] → normalised full-frame coords & 1€ Filter
            wrapped = []
            slot_filters = self._lm_filters[slot]

            for i in range(21):
                mx = lm_raw[i * 3 + 0] / INPUT_SIZE
                my = lm_raw[i * 3 + 1] / INPUT_SIZE
                mz = float(lm_raw[i * 3 + 2]) / INPUT_SIZE

                raw_x = np.clip((x1 + mx * roi_w) / w, 0.0, 1.0)
                raw_y = np.clip((y1 + my * roi_h) / h, 0.0, 1.0)

                fx, fy = slot_filters[i].filter(raw_x, raw_y, now)
                wrapped.append(_Landmark(float(fx), float(fy), mz))

            self._prev_landmarks[slot] = wrapped

            # Build tight actual hand detection bounding box
            l_xs = [lm.x * w for lm in wrapped]
            l_ys = [lm.y * h for lm in wrapped]
            final_det = Detection(
                x1=int(max(0, min(l_xs) - 10)),
                y1=int(max(0, min(l_ys) - 10)),
                x2=int(min(w, max(l_xs) + 10)),
                y2=int(min(h, max(l_ys) + 10)),
                confidence=min(1.0, max(0.0, presence)),
            )

            results.append(HandResult(
                bbox=final_det,
                landmarks=wrapped,
                handedness=handedness,
                confidence=min(1.0, max(0.0, presence)),
            ))

        return results

    def _reset_filters(self) -> None:
        for slot in range(self._max_hands):
            for lm_f in self._lm_filters[slot]:
                lm_f.reset()
            self._bbox_smoothers[slot].reset()
            self._prev_landmarks[slot] = None

    def close(self) -> None:
        self._reset_filters()
        log.info("HandTracker closed.")

    def _ensure_model(self, model_path: str) -> None:
        if not os.path.exists(TASK_FILE_PATH):
            log.info("Downloading hand_landmarker.task from Google Storage…")
            urllib.request.urlretrieve(TASK_FILE_URL, TASK_FILE_PATH)
            log.info(f"Downloaded {TASK_FILE_PATH}")

        log.info(f"Extracting {model_path} from {TASK_FILE_PATH}…")
        with zipfile.ZipFile(TASK_FILE_PATH, "r") as z:
            z.extract("hand_landmarks_detector.tflite", ".")
        log.info(f"Extracted: {model_path}")
