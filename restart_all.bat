@echo off
echo ============================================
echo   PixRevive — Full Server Restart
echo ============================================
echo.

echo [1/4] Killing existing processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM node.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/4] Clearing Python cache...
for /d /r "%~dp0backend" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo     Cache cleared.
echo.

echo [3/4] Starting backend (NO hot-reload, port 8000)...
start "PixRevive-Backend" cmd /k "cd /d %~dp0backend && set PYTHONPATH=%~dp0backend && "C:\Program Files\Python312\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000"
echo     Backend window opened. DDColor will load in background (~25s).
echo.

timeout /t 3 /nobreak >nul

echo [4/4] Starting frontend (Vite, port 5173)...
start "PixRevive-Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================
echo   Servers starting!
echo   Backend ready in ~15s  -> http://localhost:8000
echo   DDColor ready in ~40s  (ECCV16 works immediately)
echo   Frontend ready in ~5s  -> http://localhost:5173
echo ============================================
echo.
pause
