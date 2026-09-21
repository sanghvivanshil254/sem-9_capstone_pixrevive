---
name: pixrevive-python-runtime
description: >-
  Correct commands for running anything Python in PixRevive — the pinned system
  Python 3.12 interpreter, pytest path, PYTHONPATH, pip installs, DDColor setup,
  and weight downloads. Use before running any script, test, server, or pip
  command in this project, and when a command was blocked by the runtime guard.
---

# PixRevive Python runtime

## The three paths

| | |
|---|---|
| Interpreter | `C:\Program Files\Python312\python.exe` |
| Pytest | `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe` |
| `PYTHONPATH` | `d:\Desktop\kapstone\kapstone\PixRevive\backend` |

No venv, no virtualenv, no conda/mamba, no pipenv, no bare `python`/`pip`/`pytest`,
no ModelScope. A `PreToolUse` Bash hook
([scripts/hooks/guard_python_runtime.py](../../../scripts/hooks/guard_python_runtime.py))
blocks violations; append `# runtime-guard: allow` to a command to override it
deliberately.

## Git Bash (this session's shell)

```bash
"C:/Program Files/Python312/python.exe" --version          # expect 3.12.x
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

```bash
"C:/Program Files/Python312/python.exe" backend/ai/install_ddcolor.py      # DDColor sources + ~870 MB checkpoint
```

```bash
"C:/Program Files/Python312/python.exe" backend/ai/download_weights.py     # all other weights
```

```bash
"C:/Program Files/Python312/python.exe" scripts/check_progress_stream.py   # SSE progress audit
```

## PowerShell

```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Program Files\Python312\python.exe" backend\ai\install_ddcolor.py
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests\ -v
```

## Servers

Use `preview_start` with the configs in [.claude/launch.json](../../launch.json)
rather than launching by hand: `backend` (port 8000) and `frontend` (port 5173).
The backend runs without `--reload`, so restart it after backend edits and wait
for `GET /api/health` → `{"ready": true, "models": {"colorizer": true}}`.

## Weights

Everything under `backend/ai/weights/` — never `~/.cache`, never a HuggingFace hub
cache, never ModelScope. `ddcolor/ddcolor_modelscope.pth` (~870 MB) is merely the
upstream filename on `huggingface.co/piddnad/DDColor-models`.

Full detail: [CLAUDE.md](../../../CLAUDE.md) and
[.agents/rules/python_runtime.md](../../../.agents/rules/python_runtime.md).
