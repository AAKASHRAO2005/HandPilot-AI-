/**
 * content.js — Content script for Hand Gesture Controller extension.
 *
 * Receives gesture events from background.js and applies browser-specific
 * actions (scrolling, visual overlay, etc.) in the active page context.
 */

(function () {
  "use strict";

  // Gesture overlay element
  let overlay = null;
  let overlayTimeout = null;

  // ---------------------------------------------------------------------------
  // Message handler
  // ---------------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((message) => {
    if (!message || !message.type) return;

    switch (message.type) {
      case "gesture_event":
        handleGesture(message);
        break;
      case "connection_status":
        if (message.connected) {
          showOverlay("✋ Gesture Controller Connected", "#22c55e", 2000);
        }
        break;
    }
  });

  // ---------------------------------------------------------------------------
  // Gesture handlers
  // ---------------------------------------------------------------------------

  function handleGesture(event) {
    const { gesture, action, extended_fingers } = event;

    // Show gesture overlay on significant gestures
    const showGestures = ["pinch", "middle_pinch", "double_pinch", "swipe_left", "swipe_right", "swipe_up", "swipe_down"];
    if (showGestures.includes(gesture)) {
      const label = gestureLabel(gesture, action);
      showOverlay(label, getGestureColor(gesture), 800);
    }
  }

  function gestureLabel(gesture, action) {
    const map = {
      pinch: "👆 Left Click",
      middle_pinch: "👆 Right Click",
      double_pinch: "👆👆 Double Click",
      swipe_right: "→ Swipe Right",
      swipe_left: "← Swipe Left",
      swipe_up: "↑ Swipe Up",
      swipe_down: "↓ Swipe Down",
      open_hand_vertical: "📜 Scrolling",
    };
    return map[gesture] || gesture.replace(/_/g, " ").toUpperCase();
  }

  function getGestureColor(gesture) {
    const map = {
      pinch: "#7c3aed",
      middle_pinch: "#f77f00",
      double_pinch: "#7c3aed",
      swipe_right: "#06d6a0",
      swipe_left: "#06d6a0",
      open_hand_vertical: "#3b82f6",
    };
    return map[gesture] || "#7c3aed";
  }

  // ---------------------------------------------------------------------------
  // Visual overlay
  // ---------------------------------------------------------------------------

  function showOverlay(text, color = "#7c3aed", duration = 1000) {
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "gesture-controller-overlay";
      overlay.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 2147483647;
        background: rgba(13, 13, 13, 0.92);
        border: 2px solid ${color};
        border-radius: 12px;
        padding: 10px 18px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: ${color};
        pointer-events: none;
        transition: opacity 0.2s ease, transform 0.2s ease;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
      `;
      document.body.appendChild(overlay);
    }

    overlay.style.borderColor = color;
    overlay.style.color = color;
    overlay.style.opacity = "1";
    overlay.style.transform = "translateY(0)";
    overlay.textContent = text;

    clearTimeout(overlayTimeout);
    overlayTimeout = setTimeout(() => {
      if (overlay) {
        overlay.style.opacity = "0";
        overlay.style.transform = "translateY(8px)";
      }
    }, duration);
  }
})();
