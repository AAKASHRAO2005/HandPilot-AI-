# ✋ Hand Gesture Desktop Controller

> **AI-powered real-time hand gesture control for Windows** — control your entire desktop using hand movements through your webcam. No mouse required.

Built with **YOLO11** (Ultralytics) + **MediaPipe Hands** + **Windows SendInput API**.

---

## 🎥 How It Works

```
WEBCAM → OpenCV → YOLO11 (hand detection) → MediaPipe Hands (21 landmarks)
       → Gesture Engine → Windows SendInput → Any Application
```

The gesture controller operates at the **Windows OS input level**, meaning it works in:
- Google Chrome / Microsoft Edge
- Visual Studio Code
- File Explorer
- Notepad, Word, Excel
- PDF readers
- Any application that accepts mouse/keyboard input

---

## ✋ Supported Gestures

| Gesture | Action |
|---------|--------|
| Move hand | Move Windows cursor |
| Index + Thumb pinch | Left click |
| Middle + Thumb pinch | Right click |
| Two quick pinches | Double click |
| Pinch + hold + move | Drag and drop |
| 2+ fingers up, move hand up/down | Scroll vertically |
| Fast hand swipe right | Alt+Tab (next window) |
| Fast hand swipe left | Alt+Shift+Tab (prev window) |

---

## 📁 Project Structure

```
hand-desktop-controller/
├── app.py                      # Main entry point
├── config/
│   └── config.yaml             # All settings (thresholds, sensitivity, etc.)
├── vision/
│   ├── camera.py               # Threaded webcam capture
│   ├── yolo_detector.py        # YOLO11 hand detection
│   ├── hand_tracker.py         # YOLO + MediaPipe landmark fusion
│   ├── landmarks.py            # Landmark utility functions
│   └── overlay.py              # Camera feed annotations
├── gestures/
│   ├── gesture_engine.py       # Main gesture orchestrator
│   ├── pinch.py                # Pinch & double-pinch detectors
│   ├── movement.py             # Velocity & position tracker
│   ├── swipe.py                # Swipe gesture detector
│   └── state_machine.py        # Mouse interaction state machine
├── controller/
│   ├── mouse.py                # Cursor mapping + click dispatch
│   ├── keyboard.py             # Keyboard shortcut sender
│   ├── scroll.py               # Scroll wheel controller
│   └── windows_input.py        # ctypes SendInput (Windows API)
├── communication/
│   └── websocket_server.py     # WebSocket server for browser extension
├── ui/
│   └── dashboard.py            # tkinter dashboard window
├── extension/                  # Chrome/Edge browser extension (optional)
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup.html / popup.css / popup.js
├── tests/
│   ├── test_smoothing.py
│   ├── test_gestures.py
│   └── test_windows_input.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10 or 3.11
- A webcam
- (Optional) NVIDIA GPU with CUDA for faster inference

### 2. Clone / copy project
```bash
cd E:\motion_detector
```

### 3. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install PyTorch

**With CUDA (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchvision
```

### 5. Install all other dependencies
```bash
pip install -r requirements.txt
```

### 6. (Optional) Download a custom YOLO11 hand model

Place a YOLO11 hand model (`.pt` file) named `hand_yolo11.pt` in the project root.

If not present, the system automatically falls back to the standard `yolo11n.pt` model combined with MediaPipe — this works for hand tracking without requiring a custom model.

Popular community YOLO hand models:
- Search for "YOLO11 hand detection model" on Roboflow Universe or HuggingFace

---

## 🚀 Running

```bash
python app.py
```

On first run, YOLO11 will automatically download the model weights (~6 MB for `yolo11n.pt`).

### What you'll see:
1. **Camera window** opens with live hand detection overlay
2. **Dashboard window** shows system status and controls
3. Move your hand → cursor follows
4. Pinch index+thumb → left click
5. Press `ESC` in the camera window to quit

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Alt+G` | Toggle gesture control ON/OFF |
| `Ctrl+Alt+X` | Emergency stop (release all buttons) |
| `ESC` (in camera window) | Quit application |
| `G` (in camera window) | Toggle gesture control |

---

## 🌐 Browser Extension (Optional)

The extension provides visual gesture feedback inside browser tabs. Desktop control works without it.

### Install in Chrome/Edge:
1. Open `chrome://extensions/` (or `edge://extensions/`)
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `extension/` folder

The extension connects automatically to `ws://127.0.0.1:8765` when the Python app is running.

---

## ⚙️ Configuration

Edit [`config/config.yaml`](config/config.yaml) to customise:

```yaml
camera:
  index: 0        # Change if you have multiple cameras
  mirror: true    # Flip horizontally for natural feel

cursor:
  smoothing: 0.20    # Lower = smoother, Higher = more responsive
  sensitivity: 1.5   # Cursor speed multiplier
  dead_zone: 8       # Pixels of movement to ignore (reduces jitter)

gestures:
  left_click:
    pinch_threshold: 0.06   # How close fingers must be (0–1)
  scroll:
    trigger_fingers: 2       # Fingers extended to enable scroll
```

---

## 🔧 Gesture Priority System

When multiple gestures could apply simultaneously, the system resolves conflicts:

```
1. PINCH (left click / drag)    ← Highest priority
2. RIGHT CLICK
3. SCROLL
4. SWIPE (keyboard shortcuts)
5. CURSOR MOVEMENT              ← Default
```

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover:
- EMA smoother (convergence, dead zone, reset)
- Pinch detector state machine
- Double-pinch timing
- Movement velocity calculation
- Swipe direction detection
- Windows input structure construction
- Coordinate mapping (with mocked SendInput)

---

## 🛡️ Safety Features

| Feature | Description |
|---------|-------------|
| `Ctrl+Alt+X` | Immediately release all held mouse buttons |
| Auto-release on hand loss | If hand disappears during drag, mouse is released |
| Click cooldown | Prevents accidental double-clicks |
| Max cursor jump | Limits cursor speed to prevent wild movements |
| Confidence threshold | Ignores low-confidence detections |
| Enable/Disable toggle | `Ctrl+Alt+G` or dashboard button |

---

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| FPS | 25–30+ |
| Latency | < 100 ms |
| GPU | CUDA auto-detected |
| CPU fallback | Supported |

---

## 🏗️ Module Communication

```
app.py
  │
  ├── CameraCapture (vision/camera.py)
  │     └── frames → main loop
  │
  ├── HandTracker (vision/hand_tracker.py)
  │     ├── YOLODetector → bounding boxes
  │     └── MediaPipe Hands → 21 landmarks per hand
  │
  ├── GestureEngine (gestures/gesture_engine.py)
  │     ├── PinchDetector (left click / drag)
  │     ├── PinchDetector (right click)
  │     ├── DoublePinchDetector (double click)
  │     ├── MovementTracker (velocity)
  │     ├── SwipeDetector (keyboard shortcuts)
  │     └── GestureStateMachine (priority resolution)
  │
  ├── MouseController (controller/mouse.py)
  │     └── windows_input.SendInput → cursor / clicks
  │
  ├── ScrollController (controller/scroll.py)
  │     └── windows_input.scroll_vertical → scroll wheel
  │
  ├── KeyboardController (controller/keyboard.py)
  │     └── windows_input.send_hotkey → Alt+Tab etc.
  │
  ├── WebSocketServer (communication/websocket_server.py)
  │     └── ws://127.0.0.1:8765 → browser extension
  │
  └── Dashboard (ui/dashboard.py)
        └── tkinter window + OpenCV preview
```

---

## ❓ Troubleshooting

**Camera not opening:**
- Change `camera.index` in `config.yaml` (try 1, 2, etc.)
- Ensure no other application has the camera open

**Hand not detected:**
- Ensure good lighting
- Keep hand within the central 80% of the frame
- Adjust `mediapipe.min_detection_confidence` lower (e.g. 0.4)

**Cursor jitter:**
- Increase `cursor.smoothing` (e.g. 0.10)
- Increase `cursor.dead_zone` (e.g. 15)

**Clicks too sensitive / not sensitive enough:**
- Adjust `gestures.left_click.pinch_threshold` in config
- Or use the dashboard slider at runtime

**YOLO download fails:**
- Check internet connection
- Or place `yolo11n.pt` manually in the project root

---

## 📜 License

MIT License — free for personal and commercial use.
