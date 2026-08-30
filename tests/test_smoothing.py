"""
tests/test_smoothing.py — Unit tests for EMASmoother.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.smoothing import EMASmoother


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
        # After one step with alpha=0.5: 0.5*100 + 0.5*0 = 50
        assert x == pytest.approx(50.0)
        assert y == pytest.approx(50.0)

    def test_dead_zone_suppresses_small_movement(self):
        s = EMASmoother(alpha=1.0, dead_zone=20.0)
        s.update(100.0, 100.0)
        # Small move within dead zone
        x, y = s.update(105.0, 105.0)
        # Should stay at initialised position
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(100.0)

    def test_dead_zone_allows_large_movement(self):
        s = EMASmoother(alpha=1.0, dead_zone=10.0)
        s.update(0.0, 0.0)
        x, y = s.update(100.0, 0.0)
        # Distance = 100 > 10, should move
        assert x == pytest.approx(100.0)

    def test_reset_reinitialises(self):
        s = EMASmoother(alpha=0.5, dead_zone=0.0)
        s.update(50.0, 50.0)
        s.reset()
        assert s.current is None
        x, y = s.update(99.0, 99.0)
        assert x == pytest.approx(99.0)

    def test_alpha_clamped(self):
        s = EMASmoother(alpha=5.0)
        assert s.alpha == pytest.approx(1.0)
        s2 = EMASmoother(alpha=-1.0)
        assert s2.alpha == pytest.approx(0.01)

    def test_convergence(self):
        """Smoother converges to target after many frames."""
        s = EMASmoother(alpha=0.3, dead_zone=0.0)
        x, y = 0.0, 0.0
        for _ in range(100):
            x, y = s.update(1000.0, 1000.0)
        assert x == pytest.approx(1000.0, abs=1.0)
        assert y == pytest.approx(1000.0, abs=1.0)
