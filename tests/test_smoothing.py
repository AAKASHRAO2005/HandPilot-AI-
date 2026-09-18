"""
tests/test_smoothing.py — Unit tests for OneEuroFilter, PointOneEuroFilter, PointerBallistics, and EMASmoother.
"""

import math
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.smoothing import (
    OneEuroFilter,
    PointOneEuroFilter,
    PointerBallistics,
    EMASmoother,
    LowPassFilter,
)


class TestOneEuroFilter:

    def test_low_speed_high_filtering(self):
        """At very low speed, jitter should be heavily damped (small deviation)."""
        f = OneEuroFilter(min_cutoff=0.5, beta=0.001, freq=30.0)
        t = 0.0
        # Initialize
        f.filter(100.0, timestamp=t)

        # Simulate noisy micro-jitter around 100
        outputs = []
        for i in range(1, 30):
            t += 1.0 / 30.0
            noisy_val = 100.0 + (1.0 if i % 2 == 0 else -1.0)
            val = f.filter(noisy_val, timestamp=t)
            outputs.append(val)

        # Variance of output should be very small compared to raw noise (+-1.0)
        jitter = max(outputs[-10:]) - min(outputs[-10:])
        assert jitter < 0.25

    def test_high_speed_low_lag(self):
        """At high speed, the filter should quickly adapt and follow the fast change."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.1, freq=30.0)
        t = 0.0
        f.filter(0.0, timestamp=t)

        # Large jump
        t += 1.0 / 30.0
        val = f.filter(500.0, timestamp=t)
        # Because of high beta and large step, response should be significantly fast (> 350)
        assert val > 300.0

    def test_reset_clears_state(self):
        f = OneEuroFilter()
        f.filter(100.0, timestamp=0.0)
        f.reset()
        assert f.last_value is None
        val = f.filter(50.0, timestamp=1.0)
        assert val == pytest.approx(50.0)


class TestPointOneEuroFilter:

    def test_2d_filtering(self):
        pf = PointOneEuroFilter(min_cutoff=1.0, beta=0.01)
        fx, fy = pf.filter(10.0, 20.0, timestamp=0.0)
        assert fx == pytest.approx(10.0)
        assert fy == pytest.approx(20.0)

        # Update
        fx2, fy2 = pf.filter(10.5, 20.5, timestamp=0.033)
        assert abs(fx2 - 10.0) < 0.5
        assert abs(fy2 - 20.0) < 0.5

    def test_param_update(self):
        pf = PointOneEuroFilter(min_cutoff=1.0, beta=0.01)
        pf.update_params(min_cutoff=2.5, beta=0.05)
        assert pf._filter_x.min_cutoff == 2.5
        assert pf._filter_x.beta == 0.05


class TestPointerBallistics:

    def test_subpixel_precision_at_low_speed(self):
        ballistics = PointerBallistics(base_sensitivity=1.5, velocity_threshold=20.0)
        dx, dy = ballistics.apply(2.0, 0.0)
        # Low speed: scale = base_sensitivity (1.5)
        assert dx == pytest.approx(3.0)
        assert dy == pytest.approx(0.0)

    def test_acceleration_at_high_speed(self):
        ballistics = PointerBallistics(base_sensitivity=1.5, acceleration_exponent=1.5, velocity_threshold=10.0)
        dx, dy = ballistics.apply(50.0, 0.0)
        # High speed: scale > base_sensitivity
        effective_scale = dx / 50.0
        assert effective_scale > 1.5

    def test_zero_displacement(self):
        ballistics = PointerBallistics()
        dx, dy = ballistics.apply(0.0, 0.0)
        assert dx == 0.0
        assert dy == 0.0


class TestEMASmoother:

    def test_first_sample_no_smoothing(self):
        s = EMASmoother(alpha=0.5, dead_zone=0.0)
        x, y = s.update(100.0, 200.0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_ema_moves_toward_target(self):
        s = EMASmoother(alpha=0.5, dead_zone=0.0)
        s.update(0.0, 0.0)
        x, y = s.update(100.0, 100.0)
        assert x == pytest.approx(50.0)
        assert y == pytest.approx(50.0)

    def test_dead_zone_suppresses_small_movement(self):
        s = EMASmoother(alpha=1.0, dead_zone=20.0)
        s.update(100.0, 100.0)
        x, y = s.update(105.0, 105.0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(100.0)

    def test_dead_zone_allows_large_movement(self):
        s = EMASmoother(alpha=1.0, dead_zone=10.0)
        s.update(0.0, 0.0)
        x, y = s.update(100.0, 0.0)
        assert x == pytest.approx(100.0)

    def test_reset_reinitialises(self):
        s = EMASmoother(alpha=0.5, dead_zone=0.0)
        s.update(50.0, 50.0)
        s.reset()
        assert s.current is None
        x, y = s.update(99.0, 99.0)
        assert x == pytest.approx(99.0)

    def test_convergence(self):
        s = EMASmoother(alpha=0.3, dead_zone=0.0)
        x, y = 0.0, 0.0
        for _ in range(100):
            x, y = s.update(1000.0, 1000.0)
        assert x == pytest.approx(1000.0, abs=1.0)
        assert y == pytest.approx(1000.0, abs=1.0)
