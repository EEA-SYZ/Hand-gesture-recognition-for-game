@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.12 64-bit and add it to PATH.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo [2/4] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/4] Installing Python dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed. If it fails while building imutils,
    echo         install the other packages first, then copy the pure-Python imutils
    echo         package into .venv\Lib\site-packages\.
    pause
    exit /b 1
)

echo [4/4] Installing MediaPipe 0.10.21 without its optional jax dependencies ...
".venv\Scripts\python.exe" -m pip install --no-deps -r requirements-mediapipe.txt
if errorlevel 1 (
    echo [ERROR] MediaPipe installation failed.
    pause
    exit /b 1
)

echo.
echo Environment ready. Double-click run_game.bat to start.
pause
