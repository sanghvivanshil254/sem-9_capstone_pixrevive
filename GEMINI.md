# PixRevive — Agent Rules

## Python Runtime — MANDATORY

> **This rule is ALWAYS ON. Every agent working in this project MUST follow it without exception.**

### ✅ Always use Python 3.12 (system interpreter)

```
C:\Program Files\Python312\python.exe
```

**NEVER** use `venv`, `virtualenv`, `.venv`, `conda`, `pipenv`, or any other
virtual environment mechanism. The project uses the system Python 3.12 installation
directly.

### Commands

Always prefix Python commands like this:

```powershell
# Run a script
& "C:\Program Files\Python312\python.exe" script.py

# Run a module (e.g. pytest, pip)
& "C:\Program Files\Python312\python.exe" -m pytest
& "C:\Program Files\Python312\python.exe" -m pip install <package>

# Set PYTHONPATH when running backend code
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Program Files\Python312\python.exe" -m pytest tests/
```

Or use the shorter alias if Python 3.12 is registered:

```powershell
py -3.12 script.py
```

### pytest shortcut (already on PATH for this machine)

```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/
```

### ❌ Never do this

```powershell
# FORBIDDEN — DO NOT USE
.\venv\Scripts\activate
.\venv\Scripts\python.exe
python -m venv .venv
virtualenv ...
conda activate ...
```

---

## Project Layout

```
PixRevive/
├── backend/
│   ├── ai/
│   │   ├── colorization/     # DDColor (ICCV 2023) — no ECCV16
│   │   │   └── ddcolor/      # vendored upstream inference sources (Apache-2.0)
│   │   ├── weights/          # All model weights stored HERE (project workspace)
│   │   │   └── ddcolor/      # ddcolor_modelscope.pth (~870 MB)
│   │   └── pipeline/
│   ├── routers/
│   └── requirements.txt
├── frontend/
├── tests/
├── .agents/                  # Agent customizations (this folder)
├── .claude/                  # launch.json, settings.json (runtime guard hook), skills
├── CLAUDE.md                 # Canonical agent rules — edit this one first
├── AGENTS.md                 # Same rules for AGENTS.md-convention tool-chains
└── GEMINI.md                 # This file — agent rules
```

## AI Model Stack

| Model | Purpose | Weights Location |
|-------|---------|-----------------|
| DDColor (ICCV 2023) | Colorization | `backend/ai/weights/ddcolor/` |
| GFPGAN v1.4 | Face restoration | `backend/ai/weights/` |
| RealESRGAN x4 | Super-resolution | `backend/ai/weights/` |
| LaMa | Inpainting | `backend/ai/weights/` |
| CodeFormer | Alt face restore | `backend/ai/weights/CodeFormer/` |

**No ModelScope. No venv. All weights stored in the project workspace.**

## DDColor First-Time Setup

If DDColor weights are not yet downloaded, run **once**:

```powershell
& "C:\Program Files\Python312\python.exe" backend/ai/install_ddcolor.py
```

This installs DDColor from GitHub (no ModelScope) and downloads the weights to
`backend/ai/weights/ddcolor/ddcolor_modelscope.pth`.

---

## Project Shorthand Commands

- **`reload`**: Reload/restart the backend server:
  ```powershell
  # From backend/
  $env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
  & "C:\Program Files\Python312\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
  ```
- **`map`**: Make/display the codebase navigation map ([CODEBASE_MAP.md](CODEBASE_MAP.md)) so you can navigate the architecture and codebase faster.

---

## Browser Automation — Chrome CDP (No Playwright Downloads)

Playwright CDN driver downloads fail in this environment. Never download or install Playwright binaries.
Always connect to an already-running Google Chrome instance via Chrome DevTools Protocol (CDP) on port 9222:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --remote-debugging-address=127.0.0.1 `
  --user-data-dir="$env:LOCALAPPDATA\Google\Chrome\remote-debug-profile"
```

CDP Endpoint: `http://127.0.0.1:9222`. See [.agents/rules/browser_cdp.md](.agents/rules/browser_cdp.md).


