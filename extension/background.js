/**
 * background.js — Service worker for Hand Gesture Controller extension.
 *
 * Connects to the Python WebSocket server on ws://127.0.0.1:8765
 * and dispatches gesture events to content scripts.
 */

const WS_URL = "ws://127.0.0.1:8765";
const RECONNECT_DELAY_MS = 3000;

let ws = null;
let reconnectTimer = null;
let connected = false;
let lastGesture = null;

// ---------------------------------------------------------------------------
// WebSocket connection management
// ---------------------------------------------------------------------------

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    connected = true;
    clearTimeout(reconnectTimer);
    broadcastToTabs({ type: "connection_status", connected: true });
    chrome.storage.local.set({ wsConnected: true });
    console.log("[GestureExt] Connected to gesture controller.");
  };

  ws.onclose = () => {
    connected = false;
    broadcastToTabs({ type: "connection_status", connected: false });
    chrome.storage.local.set({ wsConnected: false });
    console.log("[GestureExt] Disconnected. Reconnecting…");
    scheduleReconnect();
  };

  ws.onerror = (err) => {
    console.warn("[GestureExt] WebSocket error:", err);
    ws.close();
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleGestureEvent(data);
    } catch (e) {
      // ignore
    }
  };
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
}

// ---------------------------------------------------------------------------
// Gesture event handling
// ---------------------------------------------------------------------------

function handleGestureEvent(data) {
  if (data.type !== "gesture_event") return;

  lastGesture = data;

  // Forward all gesture events to active tab's content script
  broadcastToActiveTabs(data);

  // Handle browser-specific actions
  const { gesture, action } = data;

  switch (gesture) {
    case "swipe_right":
      // Alt+Tab equivalent handled by Python, but can also switch browser tabs
      switchTab(1);
      break;
    case "swipe_left":
      switchTab(-1);
      break;
    default:
      break;
  }
}

// ---------------------------------------------------------------------------
// Tab management
// ---------------------------------------------------------------------------

async function switchTab(direction) {
  try {
    const tabs = await chrome.tabs.query({ currentWindow: true });
    if (!tabs.length) return;

    const activeTab = tabs.find(t => t.active);
    if (!activeTab) return;

    const currentIdx = tabs.indexOf(activeTab);
    const nextIdx = (currentIdx + direction + tabs.length) % tabs.length;
    await chrome.tabs.update(tabs[nextIdx].id, { active: true });
  } catch (e) {
    console.warn("[GestureExt] Tab switch error:", e);
  }
}

// ---------------------------------------------------------------------------
// Message broadcasting
// ---------------------------------------------------------------------------

function broadcastToTabs(message) {
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      chrome.tabs.sendMessage(tab.id, message).catch(() => {});
    }
  });
}

function broadcastToActiveTabs(message) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    for (const tab of tabs) {
      chrome.tabs.sendMessage(tab.id, message).catch(() => {});
    }
  });
}

// ---------------------------------------------------------------------------
// Popup communication
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "get_status") {
    sendResponse({
      connected,
      lastGesture,
      wsUrl: WS_URL,
    });
  } else if (message.type === "reconnect") {
    ws && ws.close();
    connect();
    sendResponse({ ok: true });
  }
  return true;
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

connect();
