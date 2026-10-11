@echo off
setlocal
title PixRevive Launcher
cd /d "%~dp0"

set "PYTHON=%ProgramFiles%\Python312\python.exe"

echo ============================================
echo   PixRevive - AI Photo Restoration
echo ============================================
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python 3.12 not found at "%PYTHON%"
    echo Install Python 3.12 to the default location, or edit PYTHON in this file.
    pause
    exit /b 1
)

if not exist "backend\main.py" (
    echo [ERROR] backend\main.py not found. Run this from the PixRevive folder.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [SETUP] Frontend dependencies missing, installing...
    pushd frontend
    call npm install
    popd
)

if not exist "backend\ai\weights\GFPGANv1.4.pth" (
    echo [SETUP] Model weights missing - 2.4 GB - downloading...
    "%PYTHON%" backend\ai\download_weights.py
)

echo [RUN] Starting backend on http://127.0.0.1:8000 ...
rem HD default: target the 4K-long-side output on every launch (was 2K/2048).
rem Override per run by setting PIXREVIVE_TARGET_LONG_SIDE before this file,
rem or set it to 2048 here if you want the old behavior back.
set "PIXREVIVE_TARGET_LONG_SIDE=4096"
start "PixRevive Backend" cmd /k "cd /d %~dp0backend && set PYTHONPATH=backend && set PIXREVIVE_TARGET_LONG_SIDE=%PIXREVIVE_TARGET_LONG_SIDE% && "%PYTHON%" -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo      Waiting for backend to be ready...
:wait_backend
ping -n 3 127.0.0.1 >nul
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8000/api/health -TimeoutSec 3 | Out-Null; exit 0 } catch { exit 1 }" && goto backend_ready
goto wait_backend
:backend_ready
echo      Backend is up.

echo [RUN] Starting frontend on http://localhost:5173 ...
start "PixRevive Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

ping -n 5 127.0.0.1 >nul
start "" http://localhost:5173

echo.
echo Both windows are running. Close them (or press Ctrl+C in each) to stop PixRevive.
echo API health:  http://127.0.0.1:8000/api/health
echo App:         http://localhost:5173
echo.
echo This launcher window can be closed now.
