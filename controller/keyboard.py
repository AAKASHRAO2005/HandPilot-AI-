"""
controller/keyboard.py — Keyboard gesture action dispatcher.

Maps symbolic action names to Windows virtual-key sequences via SendInput.
"""

import time
from typing import Dict, List, Optional

from controller import windows_input as wi
from utils.logger import setup_logger

log = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Name → virtual key code mappings
# ---------------------------------------------------------------------------
_KEY_MAP: Dict[str, int] = {
    "ctrl":   wi.VK_CONTROL,
    "control": wi.VK_CONTROL,
    "alt":    wi.VK_ALT,
    "shift":  wi.VK_SHIFT,
    "tab":    wi.VK_TAB,
    "win":    wi.VK_WIN,
    "up":     wi.VK_UP,
    "down":   wi.VK_DOWN,
    "left":   wi.VK_LEFT,
    "right":  wi.VK_RIGHT,
    "enter":  wi.VK_RETURN,
    "esc":    wi.VK_ESCAPE,
    "space":  wi.VK_SPACE,
    # Letter keys
    **{chr(c): ord(chr(c).upper()) for c in range(ord('a'), ord('z') + 1)},
    # Number keys
    **{str(n): 0x30 + n for n in range(10)},
    # F-keys
    **{f"f{n}": 0x6F + n for n in range(1, 13)},
}


class KeyboardController:
    """
    Sends keyboard shortcuts based on gesture events.

    Args:
        gesture_map: Dict mapping gesture name → config dict with 'keys' list.
        cooldown_ms: Minimum milliseconds between the same keyboard action.
    """

    def __init__(
        self,
        gesture_map: Optional[Dict] = None,
        cooldown_ms: float = 800.0,
    ) -> None:
        self._gesture_map = gesture_map or {}
        self._cooldown_ms = cooldown_ms
        self._last_action: Dict[str, float] = {}

    def trigger(self, gesture_name: str) -> bool:
        """
        Trigger the keyboard action mapped to *gesture_name*.

        Returns:
            True if keys were sent, False if suppressed/unmapped.
        """
        cfg = self._gesture_map.get(gesture_name)
        if not cfg:
            return False
        if not cfg.get("enabled", True):
            return False

        now = time.perf_counter()
        last = self._last_action.get(gesture_name, 0.0)
        if (now - last) * 1000 < self._cooldown_ms:
            return False

        keys: List[str] = cfg.get("keys", [])
        vk_codes = [_KEY_MAP.get(k.lower()) for k in keys]
        vk_codes = [vk for vk in vk_codes if vk is not None]

        if not vk_codes:
            log.warning(f"No valid keys for gesture '{gesture_name}': {keys}")
            return False

        wi.send_hotkey(*vk_codes)
        self._last_action[gesture_name] = now
        log.info(f"KEYBOARD gesture='{gesture_name}' keys={keys}")
        return True

    def send_keys(self, keys: List[str]) -> None:
        """Send an arbitrary list of key names as a hotkey."""
        vk_codes = [_KEY_MAP.get(k.lower()) for k in keys]
        vk_codes = [vk for vk in vk_codes if vk is not None]
        if vk_codes:
            wi.send_hotkey(*vk_codes)
