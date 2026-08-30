"""
vision/yolo_detector.py — YOLO26 hand detection wrapper (async/non-blocking).

Architecture:
    A dedicated background thread runs YOLO inference continuously.
    detect() ALWAYS returns immediately — either the latest cached result
    or an empty list during the warm-up phase.
    This means the main processing loop is NEVER blocked by YOLO.

Detection pipeline:
    1. detect(frame) → enqueues frame (dropping old), returns cached detections
    2. Background YOLO thread → dequeues frame, runs inference, stores result
    3. Every detect_every_n frames a new frame is submitted to the YOLO queue
"""

import os
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import setup_logger

log = setup_logger(__name__)


@dataclass
class Detection:
    """Single hand detection result."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_rect(self) -> Tuple[int, int, int, int]:
        """(x, y, w, h) format."""
        return (self.x1, self.y1, self.width, self.height)


class YOLODetector:
    """
    Async YOLO26 hand / person detector.

    YOLO inference runs in a private background thread.
    detect() always returns immediately — never blocks the caller.

    Args:
        model_path: Path to custom hand YOLO26 .pt model.
        fallback_model: General YOLO26 model to download if custom missing.
        confidence: Detection confidence threshold.
        iou: NMS IoU threshold (low impact for YOLO26 which is NMS-free).
        device: 'auto' | 'cpu' | 'cuda'.
        input_size: YOLO internal inference image size (px).
        detect_every_n: Submit a new frame to the YOLO queue every N calls.
        infer_width: Pre-resize frame to this width before inference.
    """

    # COCO class index for person
    _PERSON_CLASSES = {0}

    def __init__(
        self,
        model_path: str = "hand_yolo26.pt",
        fallback_model: str = "yolo26n.pt",
        confidence: float = 0.40,
        iou: float = 0.40,
        device: str = "auto",
        input_size: int = 320,
        detect_every_n: int = 3,
        infer_width: int = 640,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.input_size = input_size
        self._model = None
        self._use_full_frame = False
        self._custom_model = False
        self._detect_every_n = max(1, detect_every_n)
        self._infer_width = infer_width
        self._frame_count = 0

        # Thread-safe result storage
        self._cached_detections: List[Detection] = []
        self._result_lock = threading.Lock()

        # Frame queue: maxsize=1 so YOLO always processes the newest frame
        self._frame_queue: Queue = Queue(maxsize=1)
        self._running = True

        # Resolve device
        self.device = self._resolve_device(device)
        log.info(f"YOLO device: {self.device}")

        self._load_model(model_path, fallback_model)

        # Start async YOLO inference thread
        self._yolo_thread = threading.Thread(
            target=self._yolo_loop, daemon=True, name="YOLODetector"
        )
        self._yolo_thread.start()
        log.info("YOLO async inference thread started.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Non-blocking detection.

        Submits *frame* to the YOLO background thread every detect_every_n
        calls, then immediately returns the latest cached detections.
        Never blocks the caller regardless of how long YOLO takes.
        """
        if self._use_full_frame:
            h, w = frame.shape[:2]
            return [Detection(x1=0, y1=0, x2=w, y2=h, confidence=1.0)]

        if self._model is None:
            return []

        self._frame_count += 1

        # Submit frame to YOLO thread every N frames
        if self._frame_count % self._detect_every_n == 0:
            # Drop old frame if queue is full (always process newest)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                self._frame_queue.put_nowait(frame.copy())
            except Exception:
                pass

        # Return latest cached result — never blocks
        with self._result_lock:
            return list(self._cached_detections)

    def stop(self) -> None:
        """Stop the background YOLO thread."""
        self._running = False
        # Unblock thread if waiting
        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass

    @property
    def loaded(self) -> bool:
        return self._model is not None or self._use_full_frame

    # ------------------------------------------------------------------
    # Background YOLO thread
    # ------------------------------------------------------------------

    def _yolo_loop(self) -> None:
        """
        Runs forever in a daemon thread.
        Pulls frames from the queue, runs YOLO, stores results.
        """
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except Empty:
                continue

            if frame is None:  # Sentinel: stop signal
                break

            detections = self._run_yolo(frame)

            with self._result_lock:
                self._cached_detections = detections

    def _run_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Run one YOLO inference pass. Called from background thread only."""
        h, w = frame.shape[:2]

        # Pre-downsample to infer_width for speed
        scale = self._infer_width / w
        if scale < 1.0:
            small = cv2.resize(
                frame,
                (self._infer_width, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = frame
            scale = 1.0

        try:
            gen = self._model(
                small,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.input_size,
                device=self.device,
                agnostic_nms=True,
                verbose=False,
                stream=True,
            )
            results = list(gen)
        except Exception as exc:
            log.warning(f"YOLO inference error: {exc}")
            return []

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                if self._custom_model or cls in self._PERSON_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Scale back from small-frame space to original
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)

                    # For person-class fallback: keep upper 65% (hand region)
                    if not self._custom_model and cls in self._PERSON_CLASSES:
                        bh = y2 - y1
                        y2 = min(h, y1 + int(bh * 0.65))

                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)

                    if x2 > x1 and y2 > y1:
                        detections.append(
                            Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                        )

        return detections

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, fallback_model: str) -> None:
        from ultralytics import YOLO

        if os.path.exists(model_path):
            try:
                self._model = YOLO(model_path)
                self._custom_model = True
                log.info(f"Loaded custom YOLO26 hand model: {model_path}")
                return
            except Exception as e:
                log.warning(f"Custom model load failed ({e}) — trying fallback.")

        try:
            self._model = YOLO(fallback_model)
            self._custom_model = False
            log.info(f"Loaded YOLO26 fallback model: {fallback_model}  (NMS-free, STAL small-object detection)")
            return
        except Exception as e:
            log.warning(f"Fallback model load failed ({e}) — using full-frame mode.")

        self._use_full_frame = True
        log.info("YOLO disabled — full-frame mode active.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                log.info("CUDA detected.")
                return "cuda"
        except ImportError:
            pass
        return "cpu"
