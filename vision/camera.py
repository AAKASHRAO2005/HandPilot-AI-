"""
vision/camera.py — Threaded webcam capture with reconnect support.

The CameraCapture class runs OpenCV capture in a background thread, placing
frames into a small bounded queue.  The main thread always gets the latest
frame without blocking on I/O.
"""

import threading
import time
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np

from utils.logger import setup_logger

log = setup_logger(__name__)


class CameraCapture:
    """
    Thread-safe, non-blocking webcam capture.

    Args:
        camera_index: Index passed to cv2.VideoCapture.
        width: Requested frame width.
        height: Requested frame height.
        fps: Requested capture FPS (hint to camera driver).
        mirror: If True, horizontally flip each frame.
        queue_size: Max frames in the internal queue (old frames dropped).
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
        queue_size: int = 2,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror

        self._queue: Queue = Queue(maxsize=queue_size)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Public state
        self.connected = False
        self.actual_width = 0
        self.actual_height = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        if not self._open_camera():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCapture")
        self._thread.start()
        log.info(f"Camera {self.camera_index} started ({self.actual_width}×{self.actual_height})")
        return True

    def stop(self) -> None:
        """Stop the capture thread and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_camera()
        log.info("Camera stopped.")

    def read(self) -> Optional[np.ndarray]:
        """
        Return the latest available frame, or None if no frame is ready.
        Non-blocking — caller must handle None.
        """
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def read_latest(self) -> Optional[np.ndarray]:
        """
        Drain the entire queue and return ONLY the newest frame.
        This guarantees YOLO/TFLite always runs on live data, never on a
        stale buffered frame — eliminates processing lag completely.
        """
        frame = None
        while True:
            try:
                frame = self._queue.get_nowait()
            except Empty:
                break
        return frame  # None if nothing was queued yet

    def read_blocking(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Block up to *timeout* seconds waiting for a frame."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def switch_camera(self, new_index: int) -> bool:
        """Hot-swap the camera source."""
        log.info(f"Switching to camera {new_index}")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release_camera()
        self.camera_index = new_index
        # Flush old frames
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        return self.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Try without backend hint
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            log.error(f"Cannot open camera {self.camera_index}")
            self.connected = False
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self.connected = True
        return True

    def _release_camera(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self.connected = False

    def _capture_loop(self) -> None:
        reconnect_delay = 2.0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                log.warning("Camera disconnected — attempting reconnect…")
                self.connected = False
                time.sleep(reconnect_delay)
                self._open_camera()
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                log.warning("Empty frame received.")
                time.sleep(0.01)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Drop oldest frame if queue is full (keep latest)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)
