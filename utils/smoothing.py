"""
utils/smoothing.py — Motion smoothing filters and pointer ballistics.

Includes:
    - OneEuroFilter (1-D & 2-D): Adaptive low-pass filter (Casiez et al., CHI 2012)
      eliminates high-frequency tremor at low speeds while eliminating lag at high speeds.
    - PointerBallistics: Non-linear dynamic pointer acceleration curve for precision and speed.
    - EMASmoother: Legacy Exponential Moving Average with dead-zone (for backward compatibility).
"""

import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    """Standard single-pole low-pass filter."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._s: Optional[float] = None

    def filter(self, val: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = max(0.0, min(1.0, alpha))

        if self._s is None:
            self._s = val
        else:
            self._s = self.alpha * val + (1.0 - self.alpha) * self._s
        return self._s

    def reset(self) -> None:
        self._s = None

    @property
    def last_value(self) -> Optional[float]:
        return self._s


class OneEuroFilter:
    """
    1-D 1€ Filter (One Euro Filter).

    Casiez, G., Roussel, N. and Vogel, D. (2012).
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in HCI"
    ACM CHI 2012.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = less jitter at low speed.
        beta: Speed coefficient. Higher = less lag when moving fast.
        d_cutoff: Cutoff frequency for derivative (velocity) calculation (Hz).
        freq: Default sampling frequency (Hz) if dt is not provided.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.freq = float(freq)

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None

    @staticmethod
    def _compute_alpha(rate: float, cutoff: float) -> float:
        """Compute smoothing factor alpha from sampling rate and cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 1e-4
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """Filter input value x with optional timestamp (seconds)."""
        if self._last_time is not None and timestamp is not None:
            dt = timestamp - self._last_time
            rate = 1.0 / dt if dt > 1e-5 else self.freq
        else:
            rate = self.freq

        self._last_time = timestamp

        # Estimate derivative of input (velocity)
        prev_x = self._x_filter.last_value
        dx = 0.0 if prev_x is None else (x - prev_x) * rate
        d_alpha = self._compute_alpha(rate, self.d_cutoff)
        edx = self._dx_filter.filter(dx, d_alpha)

        # Dynamic cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._compute_alpha(rate, cutoff)

        return self._x_filter.filter(x, alpha)

    def reset(self) -> None:
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None

    @property
    def last_value(self) -> Optional[float]:
        return self._x_filter.last_value


class PointOneEuroFilter:
    """
    2-D point filter combining two OneEuroFilters (X and Y) with shared timing.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        self._filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
        self._filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)

    def filter(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.perf_counter()

        fx = self._filter_x.filter(x, timestamp)
        fy = self._filter_y.filter(y, timestamp)
        return (fx, fy)

    def reset(self) -> None:
        self._filter_x.reset()
        self._filter_y.reset()

    def update_params(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        if min_cutoff is not None:
            self._filter_x.min_cutoff = float(min_cutoff)
            self._filter_y.min_cutoff = float(min_cutoff)
        if beta is not None:
            self._filter_x.beta = float(beta)
            self._filter_y.beta = float(beta)
        if d_cutoff is not None:
            self._filter_x.d_cutoff = float(d_cutoff)
            self._filter_y.d_cutoff = float(d_cutoff)

    @property
    def current(self) -> Optional[Tuple[float, float]]:
        lx = self._filter_x.last_value
        ly = self._filter_y.last_value
        if lx is None or ly is None:
            return None
        return (lx, ly)


class PointerBallistics:
    """
    Non-linear mouse pointer acceleration curve.
    Allows sub-pixel precision for slow micro-movements while scaling up for fast transitions.
    """

    def __init__(
        self,
        base_sensitivity: float = 1.5,
        acceleration_exponent: float = 1.25,
        velocity_threshold: float = 15.0,
    ) -> None:
        self.base_sensitivity = base_sensitivity
        self.acceleration_exponent = acceleration_exponent
        self.velocity_threshold = velocity_threshold

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        """Apply non-linear acceleration to displacement (dx, dy)."""
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            return (0.0, 0.0)

        # Scale factor based on speed
        if distance < self.velocity_threshold:
            scale = self.base_sensitivity
        else:
            speed_ratio = distance / self.velocity_threshold
            scale = self.base_sensitivity * (speed_ratio ** (self.acceleration_exponent - 1.0))

        # Clamp max acceleration multiplier for safety
        scale = min(scale, self.base_sensitivity * 3.5)
        return (dx * scale, dy * scale)


class EMASmoother:
    """
    Smooths a 2-D point stream using Exponential Moving Average (with dead zone).
    """

    def __init__(self, alpha: float = 0.20, dead_zone: float = 8.0) -> None:
        self.alpha = max(0.01, min(1.0, alpha))
        self.dead_zone = dead_zone
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def update(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        if self._x is None:
            self._x = raw_x
            self._y = raw_y
            return (raw_x, raw_y)

        dx = raw_x - self._x
        dy = raw_y - self._y
        if math.hypot(dx, dy) < self.dead_zone:
            return (self._x, self._y)

        self._x = self.alpha * raw_x + (1.0 - self.alpha) * self._x
        self._y = self.alpha * raw_y + (1.0 - self.alpha) * self._y
        return (self._x, self._y)

    def reset(self) -> None:
        self._x = None
        self._y = None

    @property
    def current(self) -> Optional[Tuple[float, float]]:
        if self._x is None:
            return None
        return (self._x, self._y)
