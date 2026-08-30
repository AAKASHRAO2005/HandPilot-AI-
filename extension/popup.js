/**
 * popup.js — Extension popup logic.
 * Fetches status from background service worker and displays live gesture data.
 */

const wsStatusDot = document.getElementById("ws-dot");
const wsStatusText = document.getElementById("ws-status");
const reconnectBtn = document.getElementById("reconnect-btn");
const gestureDisplay = document.getElementById("gesture-display");
const actionDisplay = document.getElementById("action-display");
const fpsVal = document.getElementById("fps-val");
const latVal = document.getElementById("lat-val");
const fingersVal = document.getElementById("fingers-val");
const confVal = document.getElementById("conf-val");

function updateStatus(status) {
  if (status.connected) {
    wsStatusDot.classList.add("connected");
    wsStatusText.textContent = "Connected";
  } else {
    wsStatusDot.classList.remove("connected");
    wsStatusText.textContent = "Disconnected — Python app running?";
  }

  const g = status.lastGesture;
  if (g) {
    gestureDisplay.textContent = g.gesture ? g.gesture.replace(/_/g, " ").toUpperCase() : "—";
    actionDisplay.textContent = g.action || "—";
    fpsVal.textContent = g.fps !== undefined ? g.fps.toFixed(1) : "—";
    latVal.textContent = g.latency_ms !== undefined ? g.latency_ms.toFixed(0) : "—";
    fingersVal.textContent = g.extended_fingers !== undefined ? g.extended_fingers : "—";
    confVal.textContent = g.confidence !== undefined ? (g.confidence * 100).toFixed(0) + "%" : "—";
  }
}

// Initial fetch
chrome.runtime.sendMessage({ type: "get_status" }, (response) => {
  if (response) updateStatus(response);
});

// Poll every second while popup is open
const pollInterval = setInterval(() => {
  chrome.runtime.sendMessage({ type: "get_status" }, (response) => {
    if (chrome.runtime.lastError) {
      clearInterval(pollInterval);
      return;
    }
    if (response) updateStatus(response);
  });
}, 1000);

reconnectBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "reconnect" });
  wsStatusText.textContent = "Reconnecting…";
});
