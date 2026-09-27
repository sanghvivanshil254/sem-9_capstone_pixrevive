"""sweep_ip_scale.py — Phase-H3 calibration: IP_ADAPTER_SCALE vs SFace identity.

One 25-step reimagine() (rerolls=0) per candidate scale on the same case9
synthetic 60%-destroy mask as e2e_reimagine_case9.py; the recorded metric is
the SFace worst-cosine the all-faces gate measures on the composite.

Driver/child split is REQUIRED on the 7.3 GB host (measured 2026-09-27): the
fp32 SD1.5 weights never return to the OS from a live interpreter even after
unload_reimagine_models() (torch CPU allocator keeps the arenas), so the
second in-process point falls under the MINIMAL RAM floor and is refused by
the resource gate ("ram_free 1.51gb"). The driver mode spawns one fresh
python per scale; the OS reclaims the RAM between points.

Usage (backend must be STOPPED — the sweep needs the RAM floor):
  PYTHONPATH=backend python scripts/sweep_ip_scale.py                 # driver: 0.7,0.85,1.0
  PYTHONPATH=backend python scripts/sweep_ip_scale.py 0.6,1.0        # driver: custom list
  PYTHONPATH=backend python scripts/sweep_ip_scale.py --single 0.85  # child: one point
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT_DIR = os.path.join("test_artifacts", "reimagine_sweep")


def child_env(scale: float) -> dict:
    env = dict(os.environ)
    env["REIMAGINE_IP_SCALE"] = str(scale)
    env["PYTHONPATH"] = "backend"
    return env


def run_driver(scales: list[float]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for scale in scales:
        point_file = os.path.join(OUT_DIR, f"point_{scale}.json")
        print(f"\n=== spawning child for scale={scale} ===", flush=True)
        cmd = [PY, os.path.join(HERE, "sweep_ip_scale.py"), "--single", str(scale)]
        proc = subprocess.run(cmd, env=child_env(scale), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        tail = (proc.stdout or "")[-2000:]
        print(tail, flush=True)
        if proc.returncode != 0:
            print(f"child FAILED rc={proc.returncode}: {proc.stderr[-1500:]}",
                  flush=True)
            results.append({"scale": scale, "status": "child_failed",
                            "reason": (proc.stderr or "")[-300:]})
            continue
        if os.path.exists(point_file):
            with open(point_file) as f:
                results.append(json.load(f))
        else:
            results.append({"scale": scale, "status": "no_record"})
        # Reclaim check: the child must be gone before the next spawn.
        with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    print("\n=== summary ===", flush=True)
    for e in results:
        sim = (e.get("identity") or {}).get("similarity")
        print(f"scale={e.get('scale'):<5} status={e.get('status'):<18} "
              f"cosine={sim} ({e.get('elapsed_s', '?')}s)", flush=True)


def run_single(scale: float) -> None:
    import time

    import cv2
    import numpy as np

    from ai.restoration import resource_profiler
    from ai.restoration import reimaginer

    fixture = os.path.join("tests", "restoration_dataset", "damaged",
                           "case9_teared_sepia.jpg")
    bgr = cv2.imread(fixture)
    assert bgr is not None, f"fixture unreadable: {fixture}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    y0, y1 = int(h * 0.20), int(h * 0.80)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    mask[y0:y1, x0:x1] = 255

    tier, reason, meas = resource_profiler.profile()
    print(f"tier={tier.name} reason={reason} measured={meas}", flush=True)

    # Fixed seed across every scale point: scale must be the ONLY variable,
    # otherwise seed variance (measured: 0.85 -> 0.209 vs 1.0 -> 0.3596 on
    # different seeds) masquerades as a scale effect.
    t0 = time.time()
    rec = reimaginer.reimagine(rgb, mask, task_id="sweep-fixed",
                               steps=25, rerolls=0)
    elapsed = time.time() - t0
    result = rec.pop("result", None)
    os.makedirs(OUT_DIR, exist_ok=True)
    if result is not None:
        import cv2  # noqa: F811 - already imported above
        cv2.imwrite(os.path.join(OUT_DIR, f"case9_scale_{scale}.png"),
                    cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    rec.pop("resource", None)
    entry = {"scale": scale, "elapsed_s": round(elapsed, 1),
             "tier": tier.name,
             "identity_conditioned": rec.get("identity_conditioned"),
             "status": rec.get("status"),
             "identity": rec.get("identity"),
             "reason": rec.get("reason")}
    with open(os.path.join(OUT_DIR, f"point_{scale}.json"), "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(json.dumps(entry, indent=2, default=str), flush=True)
    reimaginer.unload_reimagine_models()


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--single":
        run_single(float(sys.argv[2]))
        return
    raw = sys.argv[1] if len(sys.argv) > 1 else "0.7,0.85,1.0"
    run_driver([float(s) for s in raw.split(",")])


if __name__ == "__main__":
    main()
