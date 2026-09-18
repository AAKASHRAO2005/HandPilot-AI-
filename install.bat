@echo off
title HandPilot AI — Setup & Deployment Installer
cd /d "%~dp0"

echo ============================================================
echo   HandPilot AI — Automated Setup & Environment Installer
echo ============================================================
echo.

:: 1. Verify Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

python --version
echo.

:: 2. Create virtual environment (.venv)
if not exist ".venv" (
    echo [1/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment (.venv) already exists.
)

:: 3. Activate venv
call .venv\Scripts\activate.bat

:: 4. Upgrade pip and install dependencies
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo [3/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some dependencies encountered warnings. Retrying with basic requirements...
    pip install opencv-python ultralytics pywin32 pynput PyYAML websockets Pillow
)

:: 5. Run tests
echo [4/4] Verifying installation with test suite...
pytest tests/ -q
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   Setup Complete! All tests passed.
    echo   You can now launch the app anytime with run.bat
    echo ============================================================
) else (
    echo.
    echo [WARNING] Setup finished with test notices. You can still test run.bat.
)

echo.
pause
