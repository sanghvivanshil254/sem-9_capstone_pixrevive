# -*- coding: utf-8 -*-
"""Ad-hoc SSE check: drive /api/process-stream and audit the progress events.

Run (system Python 3.12 — never a venv):
    & "C:\\Program Files\\Python312\\python.exe" scripts/check_progress_stream.py [image]

Asserts what the progress bar depends on:
  * percentages are monotonically non-decreasing
  * every stage id is one the frontend knows about
  * the stream terminates with `complete` (or `error`)
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"),
)
from ai.pipeline.stage_registry import stage_ids  # noqa: E402

API = "http://127.0.0.1:8000/api"

# Derived from ai.pipeline.stage_registry — the same source the backend and
# the frontend (via scripts/export_stage_list.py) use. A stage id the live
# pipeline emits but the registry does not know is a contract break.
KNOWN_STAGES = set(stage_ids())

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(ROOT, "test_artifacts", "sources", "teared.jpg")


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not os.path.isfile(image):
        print(f"[ERROR] No such image: {image}")
        return 2

    health = requests.get(f"{API}/health", timeout=10).json()
    print(f"health: {json.dumps(health)}")

    t0 = time.time()
    with open(image, "rb") as fh:
        resp = requests.post(
            f"{API}/process-stream",
            files={"file": (os.path.basename(image), fh, "image/jpeg")},
            data={"restore": "true", "repair": "true", "colorize": "true",
                  "denoise": "true", "device": "auto"},
            stream=True,
            timeout=900,
        )
    resp.raise_for_status()
    print(f"headers: cache-control={resp.headers.get('cache-control')!r} "
          f"x-accel-buffering={resp.headers.get('x-accel-buffering')!r}")

    events: list[tuple[str, str, int, str]] = []
    event_type = ""
    terminal = ""
    task_id = ""

    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = json.loads(line.split(":", 1)[1].strip())

        if event_type == "progress":
            stage, pct = data["stage"], int(data["percent"])
            events.append((f"{time.time() - t0:6.1f}s", stage, pct, data["message"]))
            print(f"  {time.time() - t0:6.1f}s  {pct:3d}%  {stage:<15} {data['message']}")
        elif event_type in ("complete", "error"):
            terminal = event_type
            task_id = data.get("task_id", "")
            print(f"  {time.time() - t0:6.1f}s  [{event_type}] {data.get('message')}")
            break

    print("\n--- audit ---")
    failures: list[str] = []

    regressions = [
        (events[i - 1], events[i])
        for i in range(1, len(events))
        if events[i][2] < events[i - 1][2]
    ]
    if regressions:
        for prev, cur in regressions:
            failures.append(f"percent went backwards: {prev[1]}={prev[2]} -> {cur[1]}={cur[2]}")
    else:
        print(f"[OK] {len(events)} progress events, monotonic")

    unknown = sorted({s for _, s, _, _ in events} - KNOWN_STAGES)
    if unknown:
        failures.append(f"stage ids the frontend cannot label: {unknown}")
    else:
        print(f"[OK] all stage ids known to ProgressBar.jsx: {sorted({s for _, s, _, _ in events})}")

    if terminal != "complete":
        failures.append(f"stream ended with {terminal or 'nothing'} instead of complete")
    else:
        print("[OK] stream terminated with a complete event")

    if terminal == "complete" and task_id:
        r = requests.get(f"{API}/process/{task_id}/result", timeout=120)
        if r.ok and len(r.content) > 1000:
            print(f"[OK] result downloadable: {len(r.content) / 1024:.0f} KB PNG")
        else:
            failures.append(f"result fetch failed: {r.status_code}")

    if failures:
        print("\n".join(f"[FAIL] {f}" for f in failures))
        return 1
    print(f"\n[PASS] end-to-end in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
