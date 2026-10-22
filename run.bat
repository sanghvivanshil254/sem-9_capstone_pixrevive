```bat
@echo off
setlocal
title PixRevive Launcher
cd /d "%~dp0"

echo ============================================
echo   PixRevive - AI Photo Restoration
echo ============================================
echo.

REM Detect Python
set "PYTHON="

if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON=%ProgramFiles%\Python312\python.exe"
)

if not defined PYTHON (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3.12"
)

if not defined PYTHON (
    echo [ERROR] Python was not found.
    echo Install Python 3.12 or add Python to PATH.
    pause
    exit /b 1
)

echo [INFO] Python command: %PYTHON%
%PYTHON% --version
echo.

REM Verify project structure
if not exist "backend\main.py" (
    echo [ERROR] backend\main.py not found.
    echo Run this file from the PixRevive root folder.
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo [ERROR] frontend\package.json not found.
    pause
    exit /b 1
)

REM Install frontend dependencies if necessary
if not exist "frontend\node_modules" (
    echo [SETUP] Installing frontend dependencies...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
    popd
)

REM Model weights are intentionally not downloaded.
echo [INFO] Skipping AI model weights download.
echo [INFO] Image restoration may not work without model weights.
echo.

REM Start backend
echo [RUN] Starting backend on http://127.0.0.1:8000 ...
set "PIXREVIVE_TARGET_LONG_SIDE=4096"

start "PixRevive Backend" cmd /k "cd /d "%~dp0backend" && set "PIXREVIVE_TARGET_LONG_SIDE=4096" && set "PYTHONPATH=%~dp0backend" && %PYTHON% -m uvicorn main:app --host 127.0.0.1 --port 8000"

REM Wait for backend health check, with timeout
echo [INFO] Waiting for backend health check...
set /a ATTEMPTS=0

:wait_backend
ping -n 3 127.0.0.1 >nul
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3 | Out-Null; exit 0 } catch { exit 1 }"

if not errorlevel 1 goto backend_ready

set /a ATTEMPTS+=1
if %ATTEMPTS% GEQ 10 (
    echo [ERROR] Backend did not become ready.
    echo Check the PixRevive Backend window for errors.
    echo MongoDB and model initialization may affect startup.
    pause
    exit /b 1
)

goto wait_backend

:backend_ready
echo [SUCCESS] Backend health endpoint responded.
echo [INFO] Health response alone does not confirm MongoDB connectivity.
echo.

REM Start frontend
echo [RUN] Starting frontend on http://localhost:5173 ...
start "PixRevive Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

ping -n 3 127.0.0.1 >nul
start "" "http://localhost:5173"

echo.
echo ============================================
echo PixRevive startup sequence completed.
echo Backend: http://127.0.0.1:8000
echo API docs: http://127.0.0.1:8000/docs
echo Health: http://127.0.0.1:8000/api/health
echo Frontend: http://localhost:5173
echo ============================================
echo.
echo Close the Backend and Frontend windows to stop them.
pause
endlocal
```