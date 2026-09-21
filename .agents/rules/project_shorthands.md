# Project Shorthand Commands — PixRevive

Whenever the user uses or refers to these shorthands, execute or respond with the corresponding action:

1. **`reload`** $\rightarrow$ **Reload Backend Server**
   - Kill any stale backend process running on port 8000.
   - Restart the backend using the system Python 3.12 interpreter:
     ```powershell
     $env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
     & "C:\Program Files\Python312\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
     ```
   - Verify health via `GET http://127.0.0.1:8000/api/health`.

2. **`map`** $\rightarrow$ **Make / Display Codebase Navigation Map**
   - Provide or inspect the rapid navigation map ([CODEBASE_MAP.md](file:///d:/Desktop/kapstone/kapstone/PixRevive/CODEBASE_MAP.md)).
   - This map indexes all pipeline stages (0 to 10b), AI models, directory layouts, routers, and test files for instant navigation without redundant search steps.
