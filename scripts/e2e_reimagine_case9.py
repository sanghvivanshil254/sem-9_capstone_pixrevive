"""e2e_reimagine_case9.py — H1 acceptance: real weights, real fixture.

Runs reimagine() end-to-end on case9_teared_sepia.jpg with a synthetic
hopeless-coverage Destroy mask (60% of frame), through the real chain:
resource profiler -> tier MINIMAL -> SD1.5 fp32 sequential-offload ->
evidence composite -> SFace identity gate. Saves the output + the record.

Usage:
  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" scripts/e2e_reimagine_case9.py [--steps 25]
"""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join("backend"))

from ai.restoration.reimaginer import reimagine, unload_reimagine_models  # noqa: E402
from ai.restoration import resource_profiler  # noqa: E402


def main() -> None:
    steps = 25
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])

    fixture = os.path.join("tests", "restoration_dataset", "damaged",
                           "case9_teared_sepia.jpg")
    bgr = cv2.imread(fixture)
    assert bgr is not None, f"fixture unreadable: {fixture}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Synthetic hopeless mask: destroy the central 60% of the frame
    # (the real detectors would produce something similar for this photo).
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    y0, y1 = int(h * 0.20), int(h * 0.80)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    mask[y0:y1, x0:x1] = 255

    tier, reason, meas = resource_profiler.profile()
    print(f"tier={tier.name} reason={reason} measured={meas}", flush=True)

    t0 = time.time()
    rec = reimagine(rgb, mask, task_id="e2e-case9", steps=steps)
    elapsed = time.time() - t0

    out_dir = os.path.join("test_artifacts", "reimagine_e2e")
    os.makedirs(out_dir, exist_ok=True)
    if rec.get("result") is not None:
        cv2.imwrite(os.path.join(out_dir, "case9_reimagined.png"),
                    cv2.cvtColor(rec.pop("result"), cv2.COLOR_RGB2BGR))
    rec.pop("resource", None)  # already summarized below
    print(f"elapsed: {elapsed:.1f}s", flush=True)
    print(json.dumps(rec, indent=2, default=str), flush=True)
    with open(os.path.join(out_dir, "case9_record.json"), "w") as f:
        json.dump({"elapsed_s": round(elapsed, 1), "tier": tier.name,
                   "measured": meas, "record": rec}, f, indent=2, default=str)
    unload_reimagine_models()


if __name__ == "__main__":
    main()
