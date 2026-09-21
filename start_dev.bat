@echo off
set "PROJECT_DIR=%~dp0"

echo Starting PixRevive Backend (FastAPI)...
start "Backend" cmd /k "cd /d %PROJECT_DIR%backend && set PYTHONPATH=%PROJECT_DIR%backend && "C:\Program Files\Python312\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo Starting PixRevive Frontend (Vite + React)...
start "Frontend" cmd /k "cd /d %PROJECT_DIR%frontend && npm run dev"

echo.
echo Both servers are starting in separate windows!
echo Once they are ready, the frontend will be available at http://localhost:5173
echo The backend API will be available at http://localhost:8000

echo.
echo Launching Google Chrome...
ping -n 4 127.0.0.1 >nul
start chrome http://localhost:5173

