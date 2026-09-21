---
name: pixrevive-python-runtime
description: >-
  Runbook for running Python commands in the PixRevive project.
  Use this skill to get the correct Python 3.12 interpreter paths, PYTHONPATH
  settings, test commands, and weight-download commands for this project.
  Activate when running scripts, tests, pip installs, or any shell command
  involving Python in PixRevive.
---

# PixRevive Python Runtime Runbook

## Rule #1 — Always use system Python 3.12

**Interpreter:** `C:\Program Files\Python312\python.exe`  
**Pytest:** `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe`  
**PYTHONPATH:** `d:\Desktop\kapstone\kapstone\PixRevive\backend`

Never use venv, virtualenv, conda, or any virtual environment.

---

## Common Commands

### Run any backend script

```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Program Files\Python312\python.exe" backend/ai/some_script.py
```

### Run all regression tests

```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" `
    "d:\Desktop\kapstone\kapstone\PixRevive\tests\test_restoration_regressions.py" -v
```

### Run specific test(s)

```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" `
    "d:\Desktop\kapstone\kapstone\PixRevive\tests\" -k "test_name_or_keyword" -v
```

### Install a Python package

```powershell
& "C:\Program Files\Python312\python.exe" -m pip install <package_name>
```

### Install DDColor (first time only)

```powershell
& "C:\Program Files\Python312\python.exe" backend/ai/install_ddcolor.py
```

### Download all model weights

```powershell
& "C:\Program Files\Python312\python.exe" backend/ai/download_weights.py
```

---

## Project Python Entry Points

| Script | Purpose | Command |
|--------|---------|---------|
| `backend/main.py` | Start FastAPI backend | `py312 backend/main.py` |
| `backend/ai/install_ddcolor.py` | DDColor setup (once) | `py312 backend/ai/install_ddcolor.py` |
| `backend/ai/download_weights.py` | Download all weights | `py312 backend/ai/download_weights.py` |
| `test_ddcolor.py` | DDColor smoke test | `py312 test_ddcolor.py` |
| `tests/test_restoration_regressions.py` | Full regression suite | `pytest tests/` |

Where `py312` = `& "C:\Program Files\Python312\python.exe"`

---

## Weight Storage Policy

All model weights are stored in **`backend/ai/weights/`** (project workspace).

```
backend/ai/weights/
├── GFPGANv1.4.pth
├── RealESRGAN_x4plus.pth
├── realesr-general-x4v3.pth
├── big-lama.pt
├── CodeFormer/
│   └── codeformer.pth
├── facelib/
│   ├── detection_Resnet50_Final.pth
│   └── parsing_parsenet.pth
└── ddcolor/
    └── ddcolor_modelscope.pth   ← DDColor weights (~870 MB)
```

**No weights in `~/.cache`, no ModelScope cache, no HuggingFace hub cache.**
Weights are downloaded directly from GitHub/HuggingFace using `urllib.request`.

---

## Verification Checklist

Before running Python commands in this project, verify:

1. `& "C:\Program Files\Python312\python.exe" --version` → `Python 3.12.x`
2. `$env:PYTHONPATH` is set to `d:\Desktop\kapstone\kapstone\PixRevive\backend`
3. No `venv` or virtual environment is activated
