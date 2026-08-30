"""
controller/windows_input.py — Low-level Windows mouse and keyboard input via ctypes SendInput.

This module is the ONLY place that touches the Windows API directly.
All other controller modules call functions defined here.

References:
    https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput
    https://docs.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-input
"""

import ctypes
import ctypes.wintypes as wintypes
import time
from typing import List, Optional

from utils.logger import setup_logger

log = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Windows API constants
# ---------------------------------------------------------------------------
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# Mouse event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_ABSOLUTE = 0x8000

# Keyboard event flags
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001

# Virtual key codes — commonly used
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_ALT = 0x12  # VK_MENU
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_WIN = 0x5B  # Left Windows key

# Screen metrics
SM_CXSCREEN = 0
SM_CYSCREEN = 1

# WHEEL_DELTA — standard scroll amount per notch
WHEEL_DELTA = 120


# ---------------------------------------------------------------------------
# ctypes structures for SendInput
# ---------------------------------------------------------------------------

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_input", _INPUT_UNION),
    ]


# ---------------------------------------------------------------------------
# Windows API function references
# ---------------------------------------------------------------------------
_user32 = ctypes.windll.user32
_SendInput = _user32.SendInput
_SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
_SendInput.restype = wintypes.UINT


def _get_screen_size():
    w = _user32.GetSystemMetrics(SM_CXSCREEN)
    h = _user32.GetSystemMetrics(SM_CYSCREEN)
    return w, h


SCREEN_W, SCREEN_H = _get_screen_size()
log.info(f"Screen resolution detected: {SCREEN_W}×{SCREEN_H}")


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_mouse_input(
    dx: int = 0,
    dy: int = 0,
    mouse_data: int = 0,
    flags: int = 0,
) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp._input.mi.dx = dx
    inp._input.mi.dy = dy
    inp._input.mi.mouseData = wintypes.DWORD(mouse_data)
    inp._input.mi.dwFlags = wintypes.DWORD(flags)
    inp._input.mi.time = 0
    inp._input.mi.dwExtraInfo = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_ulong))
    return inp


def _make_key_input(vk: int, flags: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp._input.ki.wVk = wintypes.WORD(vk)
    inp._input.ki.wScan = 0
    inp._input.ki.dwFlags = wintypes.DWORD(flags)
    inp._input.ki.time = 0
    inp._input.ki.dwExtraInfo = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_ulong))
    return inp


def _send(*inputs: INPUT) -> None:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    _SendInput(n, arr, ctypes.sizeof(INPUT))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def move_mouse_absolute(x: int, y: int) -> None:
    """
    Move the Windows mouse cursor to absolute screen position (x, y).
    Uses SendInput with MOUSEEVENTF_ABSOLUTE.
    """
    # SendInput absolute coords use 65535×65535 coordinate space
    norm_x = int(x * 65535 / SCREEN_W)
    norm_y = int(y * 65535 / SCREEN_H)
    inp = _make_mouse_input(
        dx=norm_x,
        dy=norm_y,
        flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
    )
    _send(inp)


def left_button_down() -> None:
    """Press and hold the left mouse button."""
    _send(_make_mouse_input(flags=MOUSEEVENTF_LEFTDOWN))


def left_button_up() -> None:
    """Release the left mouse button."""
    _send(_make_mouse_input(flags=MOUSEEVENTF_LEFTUP))


def left_click(x: Optional[int] = None, y: Optional[int] = None) -> None:
    """Perform a single left click, optionally at position (x, y)."""
    if x is not None and y is not None:
        move_mouse_absolute(x, y)
    _send(
        _make_mouse_input(flags=MOUSEEVENTF_LEFTDOWN),
        _make_mouse_input(flags=MOUSEEVENTF_LEFTUP),
    )


def right_click(x: Optional[int] = None, y: Optional[int] = None) -> None:
    """Perform a single right click, optionally at position (x, y)."""
    if x is not None and y is not None:
        move_mouse_absolute(x, y)
    _send(
        _make_mouse_input(flags=MOUSEEVENTF_RIGHTDOWN),
        _make_mouse_input(flags=MOUSEEVENTF_RIGHTUP),
    )


def double_click(x: Optional[int] = None, y: Optional[int] = None) -> None:
    """Perform a double left click, optionally at position (x, y)."""
    if x is not None and y is not None:
        move_mouse_absolute(x, y)
    left_click()
    time.sleep(0.05)
    left_click()


def scroll_vertical(ticks: int) -> None:
    """
    Scroll the mouse wheel vertically.

    Args:
        ticks: Positive = scroll up, Negative = scroll down.
               Each unit = WHEEL_DELTA (120).
    """
    amount = int(ticks * WHEEL_DELTA)
    _send(_make_mouse_input(mouse_data=amount, flags=MOUSEEVENTF_WHEEL))


def scroll_horizontal(ticks: int) -> None:
    """
    Scroll horizontally.

    Args:
        ticks: Positive = right, Negative = left.
    """
    amount = int(ticks * WHEEL_DELTA)
    _send(_make_mouse_input(mouse_data=amount, flags=MOUSEEVENTF_HWHEEL))


def key_down(vk: int) -> None:
    """Press a key by virtual-key code."""
    _send(_make_key_input(vk))


def key_up(vk: int) -> None:
    """Release a key by virtual-key code."""
    _send(_make_key_input(vk, flags=KEYEVENTF_KEYUP))


def send_hotkey(*vk_codes: int) -> None:
    """
    Press and release a sequence of keys simultaneously (e.g. Alt+Tab).

    Args:
        vk_codes: Virtual-key codes to press in order, then release in reverse.
    """
    # Press all keys
    for vk in vk_codes:
        _send(_make_key_input(vk))
        time.sleep(0.02)
    time.sleep(0.05)
    # Release all keys in reverse
    for vk in reversed(vk_codes):
        _send(_make_key_input(vk, flags=KEYEVENTF_KEYUP))
        time.sleep(0.02)


def release_all_buttons() -> None:
    """
    Emergency release — send left and right mouse button up events.
    Safe to call at any time.
    """
    _send(
        _make_mouse_input(flags=MOUSEEVENTF_LEFTUP),
        _make_mouse_input(flags=MOUSEEVENTF_RIGHTUP),
    )
    log.info("Emergency release: all mouse buttons released.")


def get_cursor_pos() -> tuple:
    """Return current Windows cursor position as (x, y)."""
    pt = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)
