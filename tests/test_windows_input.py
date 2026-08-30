"""
tests/test_windows_input.py — Sanity tests for Windows input module.

NOTE: These tests do NOT send real mouse/keyboard events to avoid unintended
      side effects during test runs. They verify that the ctypes structures
      can be built without errors and that screen metrics are sensible.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


class TestWindowsInput:

    def test_screen_size_positive(self):
        from controller.windows_input import SCREEN_W, SCREEN_H
        assert SCREEN_W > 0
        assert SCREEN_H > 0

    def test_screen_size_reasonable(self):
        from controller.windows_input import SCREEN_W, SCREEN_H
        # Most screens are between 640×480 and 7680×4320 (8K)
        assert 640 <= SCREEN_W <= 7680
        assert 480 <= SCREEN_H <= 4320

    def test_make_mouse_input_no_error(self):
        from controller.windows_input import _make_mouse_input, MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE
        inp = _make_mouse_input(dx=32767, dy=32767, flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        assert inp is not None

    def test_make_key_input_no_error(self):
        from controller.windows_input import _make_key_input, VK_TAB
        inp = _make_key_input(VK_TAB)
        assert inp is not None

    def test_get_cursor_pos_returns_tuple(self):
        from controller.windows_input import get_cursor_pos
        pos = get_cursor_pos()
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        x, y = pos
        assert isinstance(x, int)
        assert isinstance(y, int)


class TestMouseController:

    def test_coordinate_mapping_center(self):
        """Center of normalized space maps near screen center."""
        from controller.mouse import MouseController
        from controller import windows_input as wi

        mc = MouseController(
            screen_w=1920,
            screen_h=1080,
            alpha=1.0,
            dead_zone=0.0,
            active_region={"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            boundary_margin=0,
            max_jump=9999,
            mirror=False,
        )
        # Mock the actual SendInput to avoid moving mouse during tests
        original_move = wi.move_mouse_absolute
        calls = []
        wi.move_mouse_absolute = lambda x, y: calls.append((x, y))
        try:
            sx, sy = mc.update_cursor(0.5, 0.5)
            assert abs(sx - 960) < 20, f"Expected ~960, got {sx}"
            assert abs(sy - 540) < 20, f"Expected ~540, got {sy}"
        finally:
            wi.move_mouse_absolute = original_move

    def test_mirror_flips_x(self):
        """With mirror=True, left half maps to right side of screen."""
        from controller.mouse import MouseController
        from controller import windows_input as wi

        mc = MouseController(
            screen_w=1920, screen_h=1080,
            alpha=1.0, dead_zone=0.0,
            active_region={"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            boundary_margin=0, max_jump=9999, mirror=True,
        )
        original_move = wi.move_mouse_absolute
        calls = []
        wi.move_mouse_absolute = lambda x, y: calls.append((x, y))
        try:
            # norm_x=0.1 with mirror → becomes 0.9 → maps to ~1728 on 1920
            sx, sy = mc.update_cursor(0.1, 0.5)
            assert sx > 960, f"Expected right-side x, got {sx}"
        finally:
            wi.move_mouse_absolute = original_move
