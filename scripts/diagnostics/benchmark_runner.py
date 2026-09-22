import os
import sys
import time
import cv2
import numpy as np

# Ensure backend is on PYTHONPATH (script lives in scripts/diagnostics/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from ai.pipeline.restoration_pipeline import RestorationPipeline

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "restoration_dataset", "damaged")
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "test_artifacts", "iteration_benchmark")
os.makedirs(OUTPUT_BASE, exist_ok=True)

BENCHMARK_CASES = {
    "case1_noisy_lena": {
        "name": "Noisy Lenna",
        "path": os.path.join(DATASET_DIR, "case1_noisy_lena.jpg"),
        "key_checks": ["Noise removal without loss of hat texture", "Facial feature crispness", "No false inpainting of hat feathers"],
    },
    "case4_diagonal_tear": {
        "name": "Teared Portrait",
        "path": os.path.join(DATASET_DIR, "case4_diagonal_tear_portrait.jpg"),
        "key_checks": ["Clean tear line filling", "Hair continuity across tear", "Face texture consistency"],
    },
    "case5_spiderweb_creases": {
        "name": "Woman Smiling with Spiderweb Cracks",
        "path": os.path.join(DATASET_DIR, "case5_spiderweb_creases_portrait.jpg"),
        "key_checks": ["Removal of network of cracks over smile & cheek", "Preserve teeth & lips without distortion", "Skin tone naturalness"],
    },
    "case3_sepia_flaked": {
        "name": "Sepia Flaked Portrait",
        "path": os.path.join(DATASET_DIR, "case3_sepia_flaked_portrait.jpg"),
        "key_checks": ["Flake inpainting without smearing", "Sepia to natural color conversion", "Eye/mouth detail preservation"],
    },
    "case6_severe_3d_tear": {
        "name": "Severe 3D Torn Photo",
        "path": os.path.join(DATASET_DIR, "case6_severe_3d_torn_photo.png"),
        "key_checks": ["3D ragged tear boundary inpainting", "Paper crease texture elimination", "High-frequency detail"],
    }
}

def analyze_image_quality(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean, std = cv2.meanStdDev(gray)
    h, w = img_bgr.shape[:2]
    return {
        "dimensions": f"{w}x{h}",
        "laplacian_variance": round(float(lap_var), 2),
        "brightness_mean": round(float(mean[0][0]), 2),
        "contrast_std": round(float(std[0][0]), 2),
    }

def make_side_by_side(input_bgr, restored_bgr, title):
    h_r, w_r = restored_bgr.shape[:2]
    h_in, w_in = input_bgr.shape[:2]
    scale = h_r / h_in
    w_in_scaled = int(round(w_in * scale))
    resized_in = cv2.resize(input_bgr, (w_in_scaled, h_r), interpolation=cv2.INTER_AREA)

    divider = np.full((h_r, 8, 3), 50, dtype=np.uint8)
    combined = np.hstack([resized_in, divider, restored_bgr])

    # Header banner
    banner_h = 50
    banner = np.full((banner_h, combined.shape[1], 3), 25, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(banner, f"Original: {title}", (20, 33), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(banner, "PixRevive Restored", (w_in_scaled + 28, 33), font, 0.8, (50, 220, 100), 2, cv2.LINE_AA)

    return np.vstack([banner, combined])

def run_case(case_key, pipe=None):
    case = BENCHMARK_CASES.get(case_key)
    if not case:
        print(f"Unknown case: {case_key}")
        return None

    print(f"\n{'='*70}")
    print(f"RUNNING: {case['name']} ({case_key})")
    print(f"Path: {case['path']}")
    print(f"Key checks: {case['key_checks']}")
    print(f"{'='*70}")

    if not os.path.exists(case["path"]):
        print(f"ERROR: Input file does not exist: {case['path']}")
        return None

    in_bgr = cv2.imread(case["path"])
    in_rgb = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB)
    in_metrics = analyze_image_quality(in_bgr)
    print(f"Input Stats: {in_metrics}")

    if pipe is None:
        print("Initializing RestorationPipeline...")
        pipe = RestorationPipeline()

    t0 = time.time()
    restored_rgb = pipe.restore(
        image_input=in_rgb,
        enable_repair=True,
        enable_denoise=True,
        enable_colorization=True,
        enable_face_restore=True,
        enable_sr=True,
    )
    elapsed = time.time() - t0
    print(f"Restoration completed in {elapsed:.2f}s")

    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
    out_metrics = analyze_image_quality(restored_bgr)
    print(f"Output Stats: {out_metrics}")

    # Save outputs
    out_dir = os.path.join(OUTPUT_BASE, "latest")
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{case_key}_restored.png")
    cv2.imwrite(out_file, restored_bgr)

    comp_file = os.path.join(out_dir, f"{case_key}_comparison.png")
    comparison = make_side_by_side(in_bgr, restored_bgr, case["name"])
    cv2.imwrite(comp_file, comparison)

    # Save processing log
    log_file = os.path.join(out_dir, f"{case_key}_log.txt")
    if hasattr(pipe, "last_context") and pipe.last_context:
        with open(log_file, "w", encoding="utf-8") as f:
            for l in pipe.last_context.processing_log:
                f.write(l + "\n")
        print(f"Saved log: {log_file}")

    print(f"Saved restored: {out_file}")
    print(f"Saved comparison: {comp_file}")

    return {
        "case_key": case_key,
        "name": case["name"],
        "elapsed_s": round(elapsed, 2),
        "in_metrics": in_metrics,
        "out_metrics": out_metrics,
        "restored_path": out_file,
        "comp_path": comp_file,
    }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "case1_noisy_lena"
    pipe = RestorationPipeline()
    if target == "all":
        for k in BENCHMARK_CASES:
            run_case(k, pipe)
    else:
        run_case(target, pipe)
