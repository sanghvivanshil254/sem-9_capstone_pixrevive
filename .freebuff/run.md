# PixRevive — Run Doc (preview + local dev)

Two processes make up the app. Both bind the project's default ports.

## Prerequisites / artifacts

- **Frontend deps**: `node_modules` is committed-installed in `frontend/`. If missing:
  ```bash
  cd frontend && npm install
  ```
  No `.env` files exist; nothing to copy. Frontend hardcodes `http://localhost:8000`
  as the API base (see `frontend/src/components/ImageUploader.jsx`,
  `frontend/src/components/Navbar.jsx`, `frontend/src/pages/WeightsAdmin.jsx`) —
  the backend MUST be reachable on port 8000.
- **Backend deps**: system Python 3.12 with packages already installed (repo rule: NO
  venvs). Model weights live in `backend/ai/weights/` (~1.5 GB; already present).
  If weights are missing: `"C:/Program Files/Python312/python.exe" backend/ai/download_weights.py`
  then `"C:/Program Files/Python312/python.exe" backend/ai/install_ddcolor.py`.

## Run the backend (port 8000)

From `backend/` with `PYTHONPATH=backend` (repo rule: absolute interpreter path, no bare `python`):

```bash
cd backend
PYTHONPATH=backend "C:/Program Files/Python312/python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Ready when `GET http://127.0.0.1:8000/api/health` returns `{"ready": true, "models": {...}}`.
Models load in background threads (~30-60 s first start). No `--reload` flag; restart
after backend code edits. Note: uvicorn's default log config suppresses `logger.info`.

## Run the frontend (port 5173)

```bash
cd frontend && npm run dev
```

Vite serves at http://localhost:5173 (binds `localhost`/IPv6 loopback — use
`http://localhost:5173`, not `127.0.0.1`, if a direct IP fails).

## Detached start (Windows, for preview setup)

stdout and stderr must go to different files (PowerShell requirement):

```powershell
powershell -NoProfile -Command "(Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev' -WorkingDirectory 'D:\Desktop\kapstone\kapstone\PixRevive\frontend' -RedirectStandardOutput '<log>' -RedirectStandardError '<log>.err' -WindowStyle Hidden -PassThru).Id"
```

Backend equivalent: `Start-Process -FilePath 'C:\Program Files\Python312\python.exe' -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory 'D:\Desktop\kapstone\kapstone\PixRevive\backend'` with `$env:PYTHONPATH='...backend'` set first.

Alternative: `.claude/launch.json` defines both `backend` and `frontend` configurations;
`start_dev.bat` at repo root launches both in separate console windows.

## Ports

| Service | Default | Notes |
|---|---|---|
| backend | 8000 | must be 8000 (frontend hardcodes it) |
| frontend | 5173 | Vite default; free to change (API base is absolute) |

If 8000 is taken by something else, the frontend will render but restoration calls
will fail with network errors — kill the impostor or change the three frontend files.
