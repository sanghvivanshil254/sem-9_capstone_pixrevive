"""Phase 0 harness — run_fixture_matrix.py
Step 4 (A/B on one fixture) + Step 7 (matrix) in one tool.

Environment picks pipeline flag set:
  PIXREVIVE_AB_MODE=on   -> leave all flags at defaults
  PIXREVIVE_AB_MODE=off  -> force every behavioural flag to legacy/OFF
No pipeline-code changes; env only. Snapshots produced:
  pipeline_snap_<mode>/          per-fixture stage dumps
  pipeline_snap_<mode>/metrics_<fix>.json  audit-mapped metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys_path_backend = REPO / "backend"

import sys  # noqa: E402
sys.path.insert(0, str(sys_path_backend))

FIXTURES = {
    "teared":       "teared.jpg",
    "image1":       "image1.jpeg",
    "image3":       "image3.jpg.jpeg",
    "bw":           "bw.jpg",
    "lena_figure":  "SigShrink-denoising-of-the-Lena-image-corrupted-by-speckle-noise-The-SWT-with-four.png",
}

SNAP_NAMES = [
    "01_after_analyze", "02b_after_damage_cluster_smooth",
    "03_after_inpaint", "05_after_denoise",
    "06a_raw_ddcolor", "06_after_colorize",
    "07_after_face_restore", "09_after_super_res",
    "10_after_sharpen", "10b_after_projector_grade",
]

# Audit's 0-1000-normalized ROIs on teared.jpg pair
ROIS_TEARED = {
    "skin":           (225, 255, 523, 591),
    "face_flake":     (383, 336, 445, 398),
    "bg_right_blob":  (16, 787, 274, 921),
    "bg_left_blob":   (430, 792, 569, 911),
    "neck_collar":    (569, 383, 688, 518),
    "shirt":          (688, 207, 771, 373),
    "iris_left":      (316, 440, 352, 507),
    "iris_right":     (321, 600, 352, 668),
    "lip":            (484, 487, 523, 621),
    "cheek_ring":     (472, 420, 540, 690),  # ring around lip, skin only
    "flat_bg":        (0, 0, 80, 80),
}

OFF_FLAGS = {
    "PIXREVIVE_DDCOLOR_LONLY": "0",
    "PIXREVIVE_MS_FLAKES": "0",
    "PIXREVIVE_DUST_COLONIES": "0",
    "PIXREVIVE_RAGGED_BORDER": "0",
    "PIXREVIVE_NLM_CAP": "0",
    "PIXREVIVE_GRAIN": "0",
    "PIXREVIVE_FACE_NECK_FIX": "0",
    "PIXREVIVE_CHROMA_SNAP_RAW": "1",   # legacy raw-L guide
    "PIXREVIVE_CHROMA_VALIDATOR": "0",
    "PIXREVIVE_CHROMA_DISCIPLINE": "0",
    "PIXREVIVE_GRAY_SKIP": "0",
    "PIXREVIVE_CLIP_GUARD": "0",
    "PIXREVIVE_HAIR_GUARD": "0",
    "PIXREVIVE_HF_GUARD": "0",
    "PIXREVIVE_STRUCTURE_GUARD": "0",
    "PIXREVIVE_SPECK_SWEEP": "0",
    "PIXREVIVE_POST_SR_CAST": "0",
}


def set_mode(mode: str) -> None:
    for flag, val in OFF_FLAGS.items():
        if mode == "on":
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = val


def _lab(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0


def norm_roi(img: np.ndarray, frac: tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    y0, x0, y1, x1 = frac
    return img[int(y0 / 1000 * h):int(y1 / 1000 * h),
               int(x0 / 1000 * w):int(x1 / 1000 * w)]


def flake_blobs(img: np.ndarray, thresh: float = 22.0, min_area: int = 100) -> int:
    l_chan, _, _ = _lab(img)
    local = cv2.GaussianBlur(l_chan, (31, 31), 0)
    cand = ((l_chan > local + thresh) * 255).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area))


def colorfulness(img: np.ndarray) -> float:
    rg = img[:, :, 0].astype(np.float32) - img[:, :, 1].astype(np.float32)
    yb = 0.5 * (img[:, :, 0] + img[:, :, 1]).astype(np.float32) - img[:, :, 2].astype(np.float32)
    def stats(x):
        return float(x.std()), float(np.sqrt(x.var() + (x.mean() ** 2)) * 0.3)
    return float(0.30 * (np.sqrt(rg.var()) + np.sqrt(yb.var())) + 0.30 * (np.abs(rg).std() + np.abs(yb).std()))


def metrics(img: np.ndarray, inp: np.ndarray, rois: dict) -> dict:
    out: dict = {}
    L, A, B = _lab(img)
    def lab_roi(frac):
        roi = norm_roi(img, frac)
        l, a, b = _lab(roi)
        return a.mean(), b.mean()
    if "skin" in rois:
        sa, sb = lab_roi(rois["skin"])
        out["skin_a"] = round(float(sa), 2)
        out["skin_b"] = round(float(sb), 2)
    if "bg_right_blob" in rois:
        _, bb = lab_roi(rois["bg_right_blob"])
        out["bg_b"] = round(float(bb), 2)
    if "iris_left" in rois and "iris_right" in rois:
        la, lb = lab_roi(rois["iris_left"])
        ra, rb = lab_roi(rois["iris_right"])
        out["iris_ab_sep"] = round(float(np.hypot(la - ra, lb - rb)), 2)
    if "lip" in rois and "cheek_ring" in rois:
        lpa, _ = lab_roi(rois["lip"])
        cka, _ = lab_roi(rois["cheek_ring"])
        out["lip_cheek_a"] = round(float(cka - lpa), 2)  # positive => lip redder
    flat = norm_roi(img, rois.get("flat_bg", (0, 0, 80, 80)))
    Lf = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
    mad = float(np.median(np.abs(Lf.astype(np.float32) - cv2.medianBlur(Lf.astype(np.uint8), 3).astype(np.float32))))
    out["flat_sigma_mad"] = round(1.4826 * mad, 2)
    chroma = np.sqrt(A ** 2 + B ** 2)
    out["near_neutral_share"] = round(float(np.mean(chroma < 6)), 4)
    out["colorfulness"] = round(colorfulness(img), 2)
    out["flake_blob_count_all"] = flake_blobs(img)
    for name in ("bg_right_blob", "bg_left_blob", "neck_collar", "shirt", "face_flake"):
        if name in rois:
            out[f"flake_blobs_{name}"] = flake_blobs(norm_roi(img, rois[name]))
    L_in, _, _ = _lab(inp)
    face = rois.get("skin", (225, 255, 523, 591))
    def lapvar(x, frac):
        r = norm_roi(x, frac)
        g = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())
    li, lo = lapvar(inp, face), lapvar(img, face)
    out["lapvar_retention"] = round(lo / max(li, 1e-6), 4) if li > 0 else None
    return out


def run_fixture(name: str, path: str, outdir: Path, mode: str) -> dict:
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    inp = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t0 = time.time()
    try:
        pipeline = RestorationPipeline()
        result = pipeline.restore(inp, on_progress=lambda *a, **k: None)
    except Exception as exc:  # harness must never crash the report
        return {"fixture": name, "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
    dt = time.time() - t0
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "final.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / "input.png"), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
    slug_dir = REPO / "backend" / "debug_output"
    for snap in SNAP_NAMES:
        for f in slug_dir.glob(f"{snap}_*.png"):
            shutil.copy(f, outdir / f.name)
    m = metrics(result, inp, ROIS_TEARED if name == "teared" else {"flat_bg": (0, 0, 80, 80)})
    m["fixture"], m["mode"], m["seconds"] = name, mode, round(dt, 1)
    with open(outdir / f"metrics_{name}.json", "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    print(json.dumps(m, indent=None))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="teared")
    ap.add_argument("--mode", default=os.environ.get("PIXREVIVE_AB_MODE", "on"),
                    choices=["on", "off"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    set_mode(args.mode)
    outbase = Path(args.outdir or (REPO / "debug_output" / f"matrix_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"))
    for name in args.fixtures.split(","):
        path = REPO / "test demos" / FIXTURES[name]
        outdir = outbase / f"{name}_{args.mode}"
        run_fixture(name, path, outdir, args.mode)


if __name__ == "__main__":
    main()
