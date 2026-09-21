#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""guard_python_runtime.py — PreToolUse Bash guard for the PixRevive workspace.

Enforces the project's non-negotiable runtime rules (see .agents/rules/python_runtime.md
and CLAUDE.md) mechanically, so no agent can drift back to a virtualenv or a
PATH-resolved interpreter:

  * every Python invocation must use the absolute system 3.12 interpreter
  * no venv / .venv / virtualenv / pipenv / conda / mamba
  * no ModelScope for weight management

Wired up in .claude/settings.json as a PreToolUse hook on the Bash tool. Reads the
hook payload on stdin; exit 0 allows the command, exit 2 blocks it and returns the
message on stderr to the calling agent.

Escape hatch: append `# runtime-guard: allow` to a command to bypass the guard
deliberately (e.g. inspecting an existing venv directory).

Self-test:
    "C:/Program Files/Python312/python.exe" scripts/hooks/guard_python_runtime.py --self-test
"""

from __future__ import annotations

import json
import re
import sys

PY     = '"C:/Program Files/Python312/python.exe"'
PYTEST = '"C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe"'
BYPASS = "# runtime-guard: allow"

# Bare command names that must never be used: PATH may resolve a different
# interpreter than the one this project is built against.
BARE_COMMANDS = {
    "python":  f"{PY} <args>",
    "python3": f"{PY} <args>",
    "py":      f"{PY} <args>",
    "pip":     f"{PY} -m pip <args>",
    "pip3":    f"{PY} -m pip <args>",
    "pytest":  f"{PYTEST} <args>   (or {PY} -m pytest <args>)",
}

# Patterns checked against the whole command line.
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r'(?i)(?:^|[\s"\'=/\\])\.?venv[/\\]',
     "a venv-relative path"),
    (r'(?i)-m\s+(?:venv|virtualenv)\b',
     "creating a virtual environment"),
    (r'(?i)(?:^|[\s;&|(])(?:virtualenv|pipenv)\b',
     "virtualenv/pipenv"),
    (r'(?i)(?:^|[\s;&|(])(?:conda|mamba)\b',
     "conda/mamba"),
    (r'(?i)\bsource\s+\S*(?:venv|virtualenv)\S*[/\\](?:bin|Scripts)[/\\]activate',
     "activating a virtual environment"),
    (r'(?i)\bpip\b[^;&|]*\binstall\b[^;&|]*\bmodelscope\b',
     "installing ModelScope"),
    (r'(?i)(?:^|[\s;&|(])modelscope\s+(?:download|login)\b',
     "the ModelScope CLI"),
    (r'(?i)\b(?:from|import)\s+modelscope\b',
     "importing ModelScope"),
]

# Leading `VAR=value` assignments (PYTHONPATH=... python ...) sit before the
# command name and must be stripped before the first token is examined.
_ENV_PREFIX = re.compile(
    r'^\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|\'[^\']*\'|[^\s]*)\s+)*'
)
_SEGMENT_SPLIT = re.compile(r'(?:\|\||&&|[;|&\n]|\$\()')
_STRIP_PREFIXES = ("env", "time", "nohup", "exec", "sudo", "command")


def _first_token(segment: str) -> str:
    """The command name of one pipeline segment, minus env assignments/wrappers."""
    seg = _ENV_PREFIX.sub("", segment.lstrip(" \t({"))
    for _ in range(4):  # `env time python ...`
        token = seg.split(maxsplit=1)[0] if seg.strip() else ""
        if token in _STRIP_PREFIXES:
            seg = _ENV_PREFIX.sub("", seg.split(maxsplit=1)[1] if " " in seg else "")
            continue
        return token.strip('"\'')
    return ""


def check(command: str) -> str | None:
    """Return a rejection message, or None when the command is allowed."""
    if BYPASS in command:
        return None

    for pattern, what in FORBIDDEN_PATTERNS:
        if re.search(pattern, command):
            return (
                f"Blocked: this command uses {what}, which is forbidden in PixRevive.\n"
                f"  Interpreter to use instead: {PY}\n"
                f"  Pytest:                    {PYTEST}\n"
                f"  Weights live in backend/ai/weights/ — downloaded with urllib.request,\n"
                f"  never through ModelScope or a HuggingFace/pip cache.\n"
                f"  See CLAUDE.md and .agents/rules/python_runtime.md."
            )

    for segment in _SEGMENT_SPLIT.split(command):
        token = _first_token(segment)
        if token in BARE_COMMANDS:
            return (
                f"Blocked: bare `{token}` resolves through PATH, which may not be this\n"
                f"project's interpreter. PixRevive pins system Python 3.12.\n"
                f"  Use: {BARE_COMMANDS[token]}\n"
                f"  Backend imports also need: PYTHONPATH=backend\n"
                f"  See CLAUDE.md and .agents/rules/python_runtime.md."
            )
    return None


_SELF_TESTS: list[tuple[str, bool]] = [
    # (command, should_be_blocked)
    ('"C:/Program Files/Python312/python.exe" backend/ai/install_ddcolor.py', False),
    ('PYTHONPATH=backend "C:/Program Files/Python312/python.exe" -m pytest tests/', False),
    ('"C:/Program Files/Python312/python.exe" -m pip install requests', False),
    ('"C:/Program Files/Python312/python.exe" backend/ai/install_ddcolor.py --variant modelscope', False),
    ('ls backend/ai/weights/ddcolor', False),
    ('npm --prefix frontend run dev', False),
    ('cd backend && "C:/Program Files/Python312/python.exe" main.py', False),
    ('rm -rf backend/venv  # runtime-guard: allow', False),
    ('python backend/main.py', True),
    ('python3 -c "import torch"', True),
    ('pip install basicsr', True),
    ('pytest tests/ -v', True),
    ('./backend/venv/Scripts/python.exe test_ddcolor.py', True),
    ('source backend/venv/Scripts/activate', True),
    ('"C:/Program Files/Python312/python.exe" -m venv .venv', True),
    ('conda activate pixrevive', True),
    ('PYTHONPATH=backend python -m pytest tests/', True),
    ('"C:/Program Files/Python312/python.exe" -m pip install modelscope', True),
    ('ls && python --version', True),
]


def _self_test() -> int:
    failures = 0
    for command, should_block in _SELF_TESTS:
        blocked = check(command) is not None
        if blocked != should_block:
            failures += 1
            print(f"[FAIL] expected {'block' if should_block else 'allow'}: {command}")
        else:
            print(f"[ok]   {'blocked' if blocked else 'allowed'}: {command}")
    print(f"\n{len(_SELF_TESTS) - failures}/{len(_SELF_TESTS)} cases correct")
    return 1 if failures else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()

    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # Never break the tool call over a malformed payload.

    if payload.get("tool_name") != "Bash":
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    message = check(command)
    if message:
        print(message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
