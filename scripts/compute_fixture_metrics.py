"""compute_fixture_metrics.py — Phase 0 Step 6.

Computes the audit-mapped metrics for arbitrary fixture-output PNGs, without
running the pipeline (pure post-hoc, CPU-only, deterministic). Use after
run_fixture_matrix.py has produced an output dir.

Usage:
  python scripts/compute_fixture_metrics.py --dir debug_output/matrix_baseline/teared_on
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FIXTURE_ROIS = {
    "teared": {
        "skin": (225, 255, 523, 591),
        "face_flake": (383, 336, 445, 398),
        "bg_right_blob": (16, 787, 274, 921),
        "bg_left_blob": (430, 792, 569, 911),
        "neck_collar": (569, 383, 688, 518),
        "shirt": (688, 207, 771, 373),
        "iris_left": (316, 440, 352, 507),
        "iris_right": (321, 600, 352, 668),
        "lip": (484, 487, 523, 621),
        "cheek_ring": (472, 420, 540, 690),
        "flat_bg": (0, 0, 80, 80),
    },
}


def norm_roi(img: np.ndarray, frac: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    y0, x0, y1, x1 = frac
    return img[int(y0 / 1000 * h):int(y1 / 1000 * h), int(x0 / 1000 * w):int(x1 / 1000 * w)]


def _lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[:, :, 0], lab[:, :, 1] - 128, lab[:, :, 2] - 128


def flake_blobs(img, thresh=22.0, min_area=100):
    L, _, _ = _lab(img)
    local = cv2.GaussianBlur(L, (31, 31), 0)
    cand = ((L > local + thresh) * 255).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area))


def colorfulness(img):
    rg = img[:, :, 0].astype(np.float32) - img[:, :, 1].astype(np.float32)
    yb = 0.5 * (img[:, :, 0] + img[:, :, 1]).astype(np.float32) - img[:, :, 2].astype(np.float32)
    return float(0.30 * (np.sqrt(rg.var()) + np.sqrt(yb.var())) + 0.30 * (np.abs(rg).std() + np.abs(yb).std()))


def compute(img, inp, rois):
    L, A, B = _lab(img)
    m = {}
    if rois.get("skin"):
        a, b = _lab(norm_roi(img, rois["skin"]))
        m["skin_a"], m["skin_b"] = round(float(a.mean()), 2), round(float(b.mean()), 2)
    if rois.get("bg_right_blob"):
        _, b = _lab(norm_roi(img, rois["bg_right_blob"]))
        m["bg_b"] = round(float(b.mean()), 2)
    if rois.get("iris_left") and rois.get("iris_right"):
        la, lb = _lab(norm_roi(img, rois["iris_left"]))
        ra, rb = _lab(norm_roi(img, rois["iris_right"]))
        m["iris_ab_sep"] = round(float(np.hypot(la.mean() - ra.mean(), lb.mean() - rb.mean())), 2)
    if rois.get("lip") and rois.get("cheek_ring"):
        lpa, _ = _lab(norm_roi(img, rois["lip"]))
        cka, _ = _lab(norm_roi(img, rois["cheek_ring"]))
        m["lip_cheek_a"] = round(float(cka.mean() - lpa.mean()), 2)
    if rois.get("flat_bg"):
        g = cv2.cvtColor(norm_roi(img, rois["flat_bg"]), cv2.COLOR_RGB2GRAY)
        med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
        m["flat_sigma_mad"] = round(1.4826 * float(np.median(np.abs(g.astype(np.float32) - med))), 2)
    chroma = np.sqrt(A ** 2 + B ** 2)
    m["near_neutral_share"] = round(float(np.mean(chroma < 6)), 4)
    m["colorfulness"] = round(colorfulness(img), 2)
    m["flake_blob_count_all"] = flake_blobs(img)
    for name in ("face_flake", "bg_right_blob", "bg_left_blob", "neck_collar", "shirt"):
        if rois.get(name):
            m[f"flake_blobs_{name}"] = flake_blobs(norm_roi(img, rois[name]))
    if inp is not None and rois.get("skin"):
        def lv(x):
            r = norm_roi(x, rois["skin"])
            g = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return float(cv2.Laplacian(g, cv2.CV_32F).var())
        li, lo = lv(inp), lv(img)
        m["lapvar_retention"] = round(lo / max(li, 1e-6), 4) if li > 0 else None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fixture", default="teared")
    args = ap.parse_args()
    d = Path(args.dir)
    final = d / "final.png"
    inp = d / "input.png"
    if not final.exists():
        print(json.dumps({"error": f"no final.png in {d}"})); return
    img = cv2.cvtColor(cv2.imread(str(final), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    inp_img = None
    if inp.exists():
        inp_img = cv2.cvtColor(cv2.imread(str(inp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rois = FIXTURE_ROIS.get(args.fixture, {"flat_bg": (0, 0, 80, 80)})
    m = compute(img, inp_img, rois)
    m["fixture"], m["dir"] = args.fixture, str(d)
    out = d / f"metrics_recomputed_{args.fixture}.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=None))


if __name__ == "__main__":
    main()
