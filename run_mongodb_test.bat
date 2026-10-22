@echo off
setlocal
title PixRevive - MongoDB Connection Test
cd /d "%~dp0"

echo ============================================
echo   PixRevive - MongoDB Test Only
echo ============================================
echo.

REM Find Python
set "PYTHON="
where python >nul 2>&1
if not errorlevel 1 set "PYTHON=python"

if not defined PYTHON (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3"
)

if not defined PYTHON (
    echo [ERROR] Python was not found. Install Python or add it to PATH.
    pause
    exit /b 1
)

if not exist "backend\mongodb_test_app.py" (
    echo [ERROR] backend\mongodb_test_app.py was not found.
    echo Create the file using the code provided in the instructions.
    pause
    exit /b 1
)

echo [INFO] Using Python:
%PYTHON% --version
echo.

echo [INFO] Installing minimal test-server dependencies if needed...
%PYTHON% -m pip install fastapi uvicorn pymongo
if errorlevel 1 (
    echo [ERROR] Could not install dependencies.
    pause
    exit /b 1
)

echo.
echo [RUN] Starting MongoDB-only test server on http://127.0.0.1:8001
echo [INFO] This does not import the AI pipeline or download model weights.
echo [INFO] Set MONGODB_URI before running this file if required.
echo.

set "PYTHONPATH=%~dp0backend"
%PYTHON% -m uvicorn mongodb_test_app:app --app-dir "%~dp0backend" --host 127.0.0.1 --port 8001

echo.
echo [INFO] MongoDB test server stopped.
pause
endlocal
