# PixRevive — agent instructions

Applies to every agent working in this repository. The rules below are not
preferences; commands that violate them are blocked automatically by the
`PreToolUse` hook in [.claude/settings.json](.claude/settings.json) (see
[scripts/hooks/guard_python_runtime.py](scripts/hooks/guard_python_runtime.py)).

Companion copies for other tool-chains, kept in sync with this file:
[.agents/rules/python_runtime.md](.agents/rules/python_runtime.md),
[.agents/skills/pixrevive-python-runtime/SKILL.md](.agents/skills/pixrevive-python-runtime/SKILL.md),
[GEMINI.md](GEMINI.md), [AGENTS.md](AGENTS.md).

## 1. Python runtime — system Python 3.12 only

| | |
|---|---|
| Interpreter | `C:\Program Files\Python312\python.exe` |
| Pytest | `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe` |
| `PYTHONPATH` for backend code | `d:\Desktop\kapstone\kapstone\PixRevive\backend` |

### Absolute prohibitions

- **NO** `venv` / `.venv` / `virtualenv` / `pipenv`
- **NO** `conda` / `mamba`
- **NO** `.\venv\Scripts\python.exe` or any venv-relative path
- **NO** bare `python`, `python3`, `py`, `pip`, or `pytest` — PATH may resolve the
  wrong interpreter, so always use the absolute paths above
- **NO** ModelScope for model weights

The guard hook rejects all of the above. If a command is blocked but genuinely
correct, append `# runtime-guard: allow` to it rather than editing the guard.

### Command forms

This session's shell is Git Bash, so quote the interpreter path:

```bash
"C:/Program Files/Python312/python.exe" --version
```

```bash
PYTHONPATH=backend "C:/Program Files/Python312/python.exe" test_ddcolor.py
```

```bash
PYTHONPATH=backend "C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe" tests/ -v
```

```bash
"C:/Program Files/Python312/python.exe" -m pip install <package>
```

PowerShell equivalents (`& "C:\Program Files\Python312\python.exe"`, `$env:PYTHONPATH = ...`)
are in [.agents/skills/pixrevive-python-runtime/SKILL.md](.agents/skills/pixrevive-python-runtime/SKILL.md).

## 2. Model weights — project workspace only

All weights live under `backend/ai/weights/`. **Never** `~/.cache`, never a
HuggingFace hub cache, never a ModelScope cache. Downloads go through
`urllib.request` straight from GitHub or HuggingFace.

```
backend/ai/weights/
├── GFPGANv1.4.pth
├── RealESRGAN_x4plus.pth
├── realesr-general-x4v3.pth
├── big-lama.pt
├── CodeFormer/codeformer.pth
├── facelib/{detection_Resnet50_Final,parsing_parsenet}.pth
└── ddcolor/ddcolor_modelscope.pth        ~870 MB
```

`ddcolor_modelscope.pth` is just the upstream filename of the checkpoint on
HuggingFace (`piddnad/DDColor-models`) — the ModelScope service is not used.

## 3. Colorization is DDColor (ICCV 2023)

ECCV16/Caffe is retired and deleted. DDColor's inference sources are **vendored**
into `backend/ai/colorization/ddcolor/` — do not `pip install` DDColor: its
`setup.py` ships `basicsr 1.3.4.6` with CUDA C++ extensions and would overwrite
the `basicsr==1.4.2` that GFPGAN and facexlib depend on.

One-time setup (sources + checkpoint + verification):

```bash
"C:/Program Files/Python312/python.exe" backend/ai/install_ddcolor.py
```

Re-vendor after changing the pinned upstream revision with `--force`. Provenance
is recorded in `backend/ai/colorization/ddcolor/_VENDOR_INFO.md`.

## 4. Running the app

Both servers are defined in [.claude/launch.json](.claude/launch.json) — prefer the
`preview_start` tool over launching them by hand.

| Service | Port | Notes |
|---|---|---|
| backend | 8000 | `uvicorn main:app` from `backend/`; models load in background threads |
| frontend | 5173 | Vite dev server |

The backend has no `--reload`: after editing backend Python, restart it, then wait
for `GET /api/health` to report `{"ready": true, "models": {"colorizer": true}}`.
Uvicorn's default log config suppresses `logger.info`, so only WARNING and above
appear in the server log.

## 5. Progress reporting

`_emit(stage, percent, message)` in
[backend/ai/pipeline/restoration_pipeline.py](backend/ai/pipeline/restoration_pipeline.py)
clamps percentages so they never decrease. Stage ids must stay in sync with
`STAGES` in [frontend/src/components/ProgressBar.jsx](frontend/src/components/ProgressBar.jsx);
when you add or rename a stage, update both and re-run:

```bash
"C:/Program Files/Python312/python.exe" scripts/check_progress_stream.py
```

That script drives `/api/process-stream` against a real image and asserts the
events are monotonic, the stage ids are known to the UI, and the stream ends with
`complete`.

## 6. Project shorthands

- **`reload`**: Reload/restart the backend server (`uvicorn main:app` on port 8000). Kill any running backend task and start it fresh.
- **`map`**: Make/display the codebase navigation map ([CODEBASE_MAP.md](CODEBASE_MAP.md)) mapping all pipeline stages, directories, and symbols for fast agent navigation.

## 7. Browser automation — local Chrome CDP only

Playwright CDN driver downloads fail in this environment. Never download or install Playwright browser binaries.
Instead, use an already-running Google Chrome instance connected via Chrome DevTools Protocol (CDP) on port 9222:

```bash
chrome.exe --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\remote-debug-profile"
```

Endpoint: `http://127.0.0.1:9222`. See [.agents/rules/browser_cdp.md](.agents/rules/browser_cdp.md).


