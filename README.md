# ✋ HandPilot AI — Hand Gesture Desktop Controller

> **AI-powered real-time hand gesture desktop controller for Windows** — control your entire PC naturally using hand gestures through your webcam. No mouse or touch surface required.

Built with **YOLO (YOLO26/YOLO11)** + **MediaPipe TFLite Hand Landmarks (21 points)** + **1€ Adaptive Filter** + **Windows SendInput API**.

---

## 🎥 Architecture Overview

```
WEBCAM (30+ FPS) 
   │
   ├──► YOLO Detection (Background Thread, Async Non-Blocking)
   │        ▼
   ├──► Continuous Landmark-Guided ROI Tracking (Full 30+ FPS)
   │        ▼
   ├──► TFLite 21-Joint Inference (XNNPACK CPU / GPU)
   │        ▼
   ├──► 21-Point 1€ (One Euro) Adaptive Smoothing Filter
   │        ▼
   ├──► Gesture Recognition Engine (Pinch Hysteresis & Scale Invariance)
   │        ▼
   ├──► Click-Drift Stabilization & Pointer Ballistics
   │        ▼
   ├──► Windows SendInput API (OS-Level Cursor & Keyboard Control)
   │        ▼
   └──► Modern Tkinter Dashboard & Chrome WebSocket Extension
```

The gesture controller operates at the **Windows OS kernel input level**, meaning it controls:
- **Web Browsers**: Google Chrome, Microsoft Edge, Firefox, Brave
- **Code Editors & IDEs**: VS Code, Visual Studio, PyCharm
- **Productivity & Office**: Word, Excel, PowerPoint, Notion, PDF Readers
- **System**: File Explorer, Start Menu, Window Management (Alt+Tab), Media Players

---

## ✋ Gesture Signs Guide (Which Sign Does What)

The desktop dashboard features an **interactive live guide** that lights up in glowing cyan whenever a gesture sign is recognized:

| Sign Icon | Hand Gesture Sign | Desktop Action | How to Trigger |
| :---: | :--- | :--- | :--- |
| ✋ | **Open Palm / Pointing** | **Move Cursor** | Point or hold open hand. The cursor glides smoothly with 1€ adaptive low-pass filtering. |
| 🤏 | **Index Pinch** *(Thumb + Index)* | **Left Click** | Quickly bring thumb and index tips together. Automatic click freeze eliminates pointer drift. |
| ✊ | **Pinch & Hold** *(> 300ms)* | **Drag & Drop** | Pinch thumb + index and hold while moving hand. Releases when fingers open. |
| ⚡ | **Double Index Pinch** | **Double Click** | Tap index and thumb together twice rapidly within 500ms to open files/folders. |
| ✌️ | **Middle Pinch** *(Thumb + Middle)* | **Right Click** | Bring thumb and middle fingertip together to trigger context menus. |
| 📜 | **2 Fingers Extended** | **Smooth Scroll** | Keep index and middle fingers extended together; move hand up or down to scroll. |
| ↔️ | **Fast Hand Swipe** | **Switch Windows** | Quick horizontal hand swipe triggers `Alt+Tab` (right) or `Alt+Shift+Tab` (left). |
| 🔒 | **Fist** *(Closed Hand)* | **Pause Cursor** | Close your hand into a fist to park the cursor safely in place without accidental clicks. |

---

## 🚀 Quick Start & Deployment

### Option 1: One-Click Launch (Recommended)
Simply double-click:
```bat
run.bat
```
To launch silently in the background without keeping a console window open:
```bat
run_silent.vbs
```

---

### Option 2: Automated Installation on a New PC
Double-click:
```bat
install.bat
```
This automated script:
1. Verifies Python 3.10+
2. Creates a dedicated `.venv` virtual environment
3. Upgrades `pip` and installs all dependencies from `requirements.txt`
4. Runs automated tests to verify camera, vision, and input readiness

---

### Option 3: Manual Command-Line Launch
```bash
# 1. Activate virtual environment (if using one)
.venv\Scripts\activate

# 2. Run application
python app.py
```

---

### Option 4: Standalone Windows Executable (.exe)
You can package HandPilot AI into a standalone folder with models and config included:
```bash
python package_app.py
```
The executable will be generated at `dist/HandPilot-AI/HandPilot-AI.exe`.

---

## ⌨️ Global Hotkeys & Safety

| Shortcut | Action | Description |
| :---: | :---: | :--- |
| `Ctrl+Alt+G` | **Toggle Controller** | Enables or disables gesture mouse input without closing the app. |
| `Ctrl+Alt+X` | **Emergency Stop** | Instantly releases all mouse buttons and pauses gesture input. |
| `ESC` | **Exit App** | Closes camera and quits cleanly. |

---

## 🌐 Chrome / Edge Browser Extension (Optional)

The included browser extension connects to HandPilot AI's local WebSocket server (`ws://127.0.0.1:8765`) to provide in-browser gesture feedback and tab management.

### How to Install:
1. Open Google Chrome or Microsoft Edge.
2. Navigate to `chrome://extensions/` (or `edge://extensions/`).
3. Turn ON **Developer mode** (toggle in top-right corner).
4. Click **Load unpacked** (top-left button).
5. Select the `extension/` folder inside this repository.
6. The HandPilot AI icon will appear in your browser toolbar!

---

## ⚙️ Configuration Reference

All settings can be customized in [`config/config.yaml`](config/config.yaml):

```yaml
camera:
  index: 0              # Camera device index (0 = default webcam)
  width: 1280
  height: 720
  mirror: true          # Horizontal flip for intuitive mirror tracking
  queue_size: 2         # Low-latency queue depth

cursor:
  filter_type: "one_euro" # "one_euro" (jitter-free) or "ema"
  min_cutoff: 1.0       # Lower = less tremor when stationary
  beta: 0.008           # Higher = zero lag when moving fast
  acceleration: true    # Pointer ballistics for pixel precision
  sensitivity: 1.5      # Cursor speed multiplier
  dead_zone: 6          # Pixel threshold to eliminate hand tremor

gestures:
  left_click:
    pinch_threshold: 0.06
    click_freeze_s: 0.18 # Anti-drift click freeze duration
    scale_invariant: false
```

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest
```
```
============================= 37 passed in 0.27s ==============================
```
Tests cover:
- 1€ Filter low-speed tremor reduction and high-speed responsiveness
- 2D PointOneEuroFilter and PointerBallistics curves
- Scale-invariant pinch calculations and hysteresis release boundaries
- Movement tracker velocity calculations
- Windows input ctypes structure construction and mapping

---

## 📁 Repository Structure

```
HandPilot-AI/
├── app.py                      # Main application entry point & threading orchestrator
├── run.bat                     # Double-clickable Windows launcher
├── run_silent.vbs              # Silent background launcher
├── install.bat                 # Automated installation script
├── package_app.py              # Standalone PyInstaller builder
├── requirements.txt            # Python dependencies
├── version.py                  # Version & metadata (v1.0.0)
│
├── config/
│   └── config.yaml             # Core configuration (camera, smoothing, gestures)
├── assets/
│   ├── app_icon.ico            # Windows desktop application icon
│   └── icon256.png             # Hi-res branding icon
│
├── vision/
│   ├── camera.py               # Non-blocking threaded webcam capture
│   ├── hand_tracker.py         # Continuous landmark-guided tracker & TFLite
│   ├── yolo_detector.py        # Asynchronous YOLO hand detection
│   ├── landmarks.py            # Geometric utilities & hand scale metrics
│   └── overlay.py              # Camera HUD & gesture sign banner
│
├── gestures/
│   ├── gesture_engine.py       # Gesture orchestrator & click-drift stabilizer
│   ├── pinch.py                # Dual-threshold hysteresis pinch detectors
│   ├── movement.py             # Filtered velocity & position tracker
│   ├── swipe.py                # Fast directional swipe recognition
│   └── state_machine.py        # Interaction state machine
│
├── controller/
│   ├── mouse.py                # Cursor mapping, 1€ smoothing & ballistics
│   ├── scroll.py               # Inertial sub-tick scroll wheel accumulator
│   ├── keyboard.py             # Hotkey and shortcut dispatcher
│   └── windows_input.py        # Low-level ctypes Windows SendInput API
│
├── ui/
│   └── dashboard.py            # Dark-themed Tkinter dashboard & live gesture guide
├── extension/                  # Chrome / Edge browser extension
│   ├── manifest.json
│   ├── background.js / content.js
│   ├── popup.html / popup.css / popup.js
│   └── icons/ (16px, 48px, 128px)
└── tests/
    ├── test_smoothing.py       # 1€ filter & ballistics tests
    ├── test_gestures.py        # Pinch, scale-invariance & swipe tests
    └── test_windows_input.py   # Windows API structure tests
```
