"""benchmark_quality.py — PixRevive restoration quality ruler (Q-6).

Drives the case images from tests/restoration_dataset/damaged/ through the
full pipeline and scores the outputs with reference-free metrics:

  - output NIQE  (naturalness, lower is better) — the primary ruler
  - SSIM vs input, downscaled to input size — structural-drift tripwire
    (a collapse here means the pipeline "fixed" the image by replacing it)
  - damage/noise/blur/color_type measured on the input, for context

Outputs (in git-ignored test_artifacts/):
  benchmark_report.json       full report for this run
  benchmark_report.md         human-readable table + diff vs previous run
  benchmark_<case>.png        restored previews for eyeballing

The markdown diff is the acceptance gate for tuning/new-stage work (4B):
every change must show parity (no NIQE regression > 2% on any case) or an
improvement. Run it before and after a change:
    "C:/Program Files/Python312/python.exe" scripts/benchmark_quality.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from ai.analysis.niqe_metric import niqe_score  # noqa: E402
from ai.analysis.quality_analyzer import QualityAnalyzer  # noqa: E402
from ai.pipeline.restoration_pipeline import RestorationPipeline  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAMAGED_DIR = os.path.join(REPO_ROOT, "tests", "restoration_dataset", "damaged")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "test_artifacts")
REPORT_JSON = os.path.join(ARTIFACTS_DIR, "benchmark_report.json")
REPORT_MD = os.path.join(ARTIFACTS_DIR, "benchmark_report.md")

REGRESSION_TOLERANCE = 0.02  # Q-6 gate: no case worse than +2% NIQE


def _ssim_vs_input(restored: np.ndarray, original: np.ndarray) -> float | None:
    """SSIM between the restored image (resized back down) and the input.

    Purely a drift detector: values near 1.0 mean structure was preserved;
    a collapse means the output no longer resembles the photograph.
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        h, w = original.shape[:2]
        down = cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)
        return round(
            float(ssim(original, down, channel_axis=-1, data_range=255)), 4
        )
    except Exception:
        return None


def run_benchmark(case_paths: list[str], pipeline: RestorationPipeline | None = None):
    """Restore each case and score it. Returns the report dict."""
    owned = pipeline is None
    if owned:
        pipeline = RestorationPipeline()
        pipeline._models_ready.wait(timeout=300.0)
        pipeline._colorizer_ready.wait(timeout=300.0)

    cases: dict[str, dict] = {}
    try:
        for path in case_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            bgr = cv2.imread(path)
            if bgr is None:
                cases[name] = {"error": "unreadable image"}
                continue
            original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            in_meta = QualityAnalyzer.analyze(original)
            restored = pipeline.restore(
                image_input=original.copy(),
                enable_repair=True,
                enable_denoise=True,
                enable_colorization=True,
                enable_face_restore=True,
                enable_sr=True,
            )

            cases[name] = {
                "input": {
                    "damage_percentage": round(in_meta["damage_percentage"], 3),
                    "dark_spot_percentage": round(in_meta["dark_spot_percentage"], 3),
                    "noise_score": round(in_meta["noise_score"], 3),
                    "blur_score": round(in_meta["blur_score"], 1),
                    "color_type": in_meta["color_type"],
                    "niqe": in_meta.get("niqe_score"),
                },
                "output": {
                    "niqe": niqe_score(restored),
                    "ssim_vs_input": _ssim_vs_input(restored, original),
                    "size": [int(restored.shape[1]), int(restored.shape[0])],
                },
            }
            cv2.imwrite(
                os.path.join(ARTIFACTS_DIR, f"benchmark_{name}.png"),
                cv2.cvtColor(restored, cv2.COLOR_RGB2BGR),
            )
            print(f"  [done] {name}: niqe={cases[name]['output']['niqe']} "
                  f"ssim={cases[name]['output']['ssim_vs_input']}")
    finally:
        if owned:
            pass  # pipeline processes are one-shot for this script; let it GC

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": cases,
    }


def _load_previous() -> dict | None:
    try:
        with open(REPORT_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_report(report: dict) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    previous = _load_previous()

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# PixRevive benchmark report",
        "",
        f"Run: {report['timestamp']}",
    ]
    if previous:
        lines.append(f"Previous run: {previous.get('timestamp', '?')}")
    lines += [
        "",
        "| Case | in damage% | in NIQE | out NIQE | Δ NIQE vs prev | SSIM vs input |",
        "|---|---|---|---|---|---|",
    ]

    regressions: list[str] = []
    for name, case in report["cases"].items():
        if "error" in case:
            lines.append(f"| {name} | ERROR: {case['error']} | | | | |")
            continue
        out_niqe = case["output"]["niqe"]
        delta_str = ""
        if previous:
            prev = previous.get("cases", {}).get(name, {}).get("output", {})
            prev_niqe = prev.get("niqe")
            if prev_niqe is not None and out_niqe is not None:
                delta = out_niqe - prev_niqe
                marker = "🟢" if delta <= 0 else ("🟡" if delta / prev_niqe <= REGRESSION_TOLERANCE else "🔴")
                delta_str = f"{marker} {delta:+.2f}"
                if delta > 0 and delta / prev_niqe > REGRESSION_TOLERANCE:
                    regressions.append(name)
        in_niqe = case["input"].get("niqe")
        lines.append(
            f"| {name} | {case['input']['damage_percentage']:.2f} "
            f"| {in_niqe if in_niqe is not None else '—'} "
            f"| {out_niqe if out_niqe is not None else '—'} "
            f"| {delta_str} "
            f"| {case['output']['ssim_vs_input']} |"
        )

    lines.append("")
    if previous and report["cases"]:
        if regressions:
            lines.append(
                f"**VERDICT: REGRESSION** — NIQE worsened >{REGRESSION_TOLERANCE:.0%} "
                f"on: {', '.join(regressions)}"
            )
        else:
            lines.append(
                f"**VERDICT: PASS** — no case regressed beyond "
                f"{REGRESSION_TOLERANCE:.0%} NIQE."
            )
    else:
        lines.append("First recorded run — baseline established (no diff).")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] wrote {REPORT_JSON}")
    print(f"[OK] wrote {REPORT_MD}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        case_paths = [
            os.path.join(DAMAGED_DIR, a if a.endswith((".jpg", ".png", ".jpeg")) else f"{a}.jpg")
            for a in args
        ]
    else:
        case_paths = sorted(
            os.path.join(DAMAGED_DIR, f)
            for f in os.listdir(DAMAGED_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    case_paths = [p for p in case_paths if os.path.exists(p)]
    if not case_paths:
        print("No case images found.")
        sys.exit(1)

    print(f"Benchmarking {len(case_paths)} case(s) through the full pipeline...")
    report = run_benchmark(case_paths)
    write_report(report)


if __name__ == "__main__":
    main()
