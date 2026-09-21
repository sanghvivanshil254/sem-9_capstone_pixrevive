---
# PixRevive — Python Runtime Enforcement
# Scope: workspace (.agents/rules/) — loaded by all agents working in this project
# Trigger: always_on
---

## CRITICAL: Python Runtime Rule

All agents MUST use the **system Python 3.12** interpreter at all times in this project.

### Canonical interpreter path

```
C:\Program Files\Python312\python.exe
```

### Pytest path (already installed for this user)

```
C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe
```

### PYTHONPATH for backend tests

Always set PYTHONPATH to the backend directory when running tests or backend scripts:

```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
```

### Verification command

Before running any Python command, verify the interpreter with:

```powershell
& "C:\Program Files\Python312\python.exe" --version
# Expected: Python 3.12.x
```

### ABSOLUTE PROHIBITIONS

- **NO** `venv` / `.venv` / `virtualenv`
- **NO** `conda` / `mamba` environments  
- **NO** `pipenv`
- **NO** `.\venv\Scripts\python.exe` or any venv-relative path
- **NO** bare `python` or `python3` without the full path (PATH may resolve wrong interpreter)
- **NO** ModelScope for model weight management

### Package installation

```powershell
# Install a package to system Python 3.12
& "C:\Program Files\Python312\python.exe" -m pip install <package>
```

### Enforcement

For Claude Code these rules are enforced mechanically: a `PreToolUse` Bash hook
(`.claude/settings.json` → `scripts/hooks/guard_python_runtime.py`) rejects venv,
conda, pipenv, ModelScope and bare `python`/`pip`/`pytest` invocations. Other
tool-chains have no backstop — follow the rules by hand, and check a command with:

```powershell
& "C:\Program Files\Python312\python.exe" scripts\hooks\guard_python_runtime.py --self-test
```

The full project instruction set lives in `CLAUDE.md` (mirrored by `AGENTS.md`).
