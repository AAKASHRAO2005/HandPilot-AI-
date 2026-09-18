"""
package_app.py — PyInstaller automated build script for standalone Windows deployment.

Builds a standalone executable into dist/HandPilot-AI/ with models, config, and icons bundled.
Usage:
    python package_app.py
"""

import os
import sys
import subprocess
import shutil

def main():
    print("=" * 60)
    print("  HandPilot AI — Standalone Executable Packaging")
    print("=" * 60)

    # Check if pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("[INFO] PyInstaller not installed. Installing pyinstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Collect data files
    data_args = [
        "--add-data", "config;config",
        "--add-data", "assets;assets",
    ]

    for model_file in ["yolo26n.pt", "hand_landmarks_detector.tflite", "hand_landmarker.task"]:
        if os.path.exists(model_file):
            data_args.extend(["--add-data", f"{model_file};."])

    icon_arg = []
    if os.path.exists("assets/app_icon.ico"):
        icon_arg = ["--icon", "assets/app_icon.ico"]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "HandPilot-AI",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        *icon_arg,
        *data_args,
        "--hidden-import", "ultralytics",
        "--hidden-import", "ai_edge_litert",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "yaml",
        "--hidden-import", "websockets",
        "--hidden-import", "pynput",
        "app.py"
    ]

    print("[INFO] Running PyInstaller command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("  Build Succeeded!")
        print("  Executable folder created at: dist/HandPilot-AI/")
        print("  Main executable: dist/HandPilot-AI/HandPilot-AI.exe")
        print("=" * 60)
    else:
        print(f"\n[ERROR] Build failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
