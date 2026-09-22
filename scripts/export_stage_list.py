# -*- coding: utf-8 -*-
"""Export the backend stage registry to frontend/src/generated/stages.json.

Keeps ProgressBar.jsx in lockstep with what the pipeline actually emits:
run this whenever STAGE_REGISTRY changes, and commit the generated file.

Run (system Python 3.12):
    & "C:\\Program Files\\Python312\\python.exe" scripts/export_stage_list.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from ai.pipeline.stage_registry import registry_to_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "frontend", "src", "generated", "stages.json")


def main() -> int:
    payload = registry_to_json()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    n_stages = len(payload["stages"])
    n_edge = len(payload["edge_events"])
    print(f"[OK] wrote {OUT} ({n_stages} stages, {n_edge} edge events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
