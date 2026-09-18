@echo off
title HandPilot AI — Gesture Controller Launcher
cd /d "%~dp0"

echo ============================================================
echo   HandPilot AI — Launching Controller
echo ============================================================

:: 1. Check for virtual environment in .venv or venv
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    goto :RUN
)

if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
    goto :RUN
)

:: 2. Check system python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in system PATH.
    echo Please install Python 3.10+ or run install.bat first.
    pause
    exit /b 1
)

:RUN
echo [INFO] Starting HandPilot AI...
python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Application exited with error code %ERRORLEVEL%.
    pause
)
