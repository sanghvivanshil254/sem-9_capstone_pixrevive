# AGENTS.md — PixRevive

Read [CLAUDE.md](CLAUDE.md) first; it is the full instruction set and applies to
every agent regardless of tool-chain. This file exists so agents that look for
`AGENTS.md` by convention (Codex and friends) find the same rules.

The four rules that are enforced, not suggested:

1. **Python is the system 3.12 interpreter, always:**
   `C:\Program Files\Python312\python.exe`.
   Pytest: `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe`.
   Backend code needs `PYTHONPATH=backend`.
2. **No virtual environments of any kind** — no `venv`, `.venv`, `virtualenv`,
   `pipenv`, `conda`, `mamba`. No bare `python` / `pip` / `pytest` either: PATH may
   resolve the wrong interpreter.
3. **Model weights live in `backend/ai/weights/`** — never `~/.cache`, never a
   HuggingFace hub cache, never ModelScope. Downloads use `urllib.request`
   against GitHub or HuggingFace.
4. **Colorization is DDColor (ICCV 2023)**, vendored into
   `backend/ai/colorization/ddcolor/`. Never `pip install` DDColor — it would
   overwrite the `basicsr==1.4.2` that GFPGAN and facexlib need.

Rules 1–3 are enforced mechanically for Claude Code by the `PreToolUse` Bash hook
in [.claude/settings.json](.claude/settings.json), implemented in
[scripts/hooks/guard_python_runtime.py](scripts/hooks/guard_python_runtime.py).
Agents on other tool-chains have no such backstop, so follow the rules by hand.
Check your own commands with:

```bash
"C:/Program Files/Python312/python.exe" scripts/hooks/guard_python_runtime.py --self-test
```

Tool-chain-specific copies of the same runtime rules:

| File | Consumer |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Claude Code (canonical) |
| [.agents/rules/python_runtime.md](.agents/rules/python_runtime.md) | always-on workspace rule |
| [.agents/skills/pixrevive-python-runtime/SKILL.md](.agents/skills/pixrevive-python-runtime/SKILL.md) | runbook with PowerShell forms |
| [.claude/skills/pixrevive-python-runtime/SKILL.md](.claude/skills/pixrevive-python-runtime/SKILL.md) | `/pixrevive-python-runtime` in Claude Code |
| [GEMINI.md](GEMINI.md) | Gemini |

Edit CLAUDE.md first, then propagate.

---

## Project Shorthands

- **`reload`**: Reload/restart the backend server (`uvicorn main:app` on port 8000).
- **`map`**: Make/display the codebase navigation map ([CODEBASE_MAP.md](CODEBASE_MAP.md)) for rapid navigation across pipeline stages, modules, and entry points.

---

## Browser Automation & Testing

Playwright CDN driver downloads fail in this environment. Never download or install Playwright binaries.
Always use an already-running Google Chrome instance connected via Chrome DevTools Protocol (CDP) on port 9222:

```cmd
chrome.exe --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\remote-debug-profile"
```

Endpoint: `http://127.0.0.1:9222`. See [.agents/rules/browser_cdp.md](.agents/rules/browser_cdp.md).


