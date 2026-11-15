@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if /I "%GAME_NO_ELEVATE%"=="1" goto run

net session >nul 2>&1
if errorlevel 1 (
    echo [INFO] Camera access requires administrator privileges on this machine.
    echo [INFO] Requesting elevation via UAC ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath cmd.exe -ArgumentList '/c','%~f0' -Verb RunAs"
    if errorlevel 1 (
        echo [ERROR] Elevation was cancelled. Right-click the file and choose "Run as administrator".
        pause
    )
    exit /b
)

:run
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found. Run setup_env.bat first.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" check_camera.py

echo.
pause
