import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "backend"))
import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.restoration_pipeline import RestorationPipeline
from ai.preprocessing.speck_suppressor import SpeckSuppressor

p = os.path.join(_ROOT, "tests", "restoration_dataset", "damaged")
cases = [
    "case1_noisy_lena.jpg",
    "case2_dead_tree_landscape.jpg",
    "case3_sepia_flaked_portrait.jpg",
    "case4_diagonal_tear_portrait.jpg",
    "case5_spiderweb_creases_portrait.jpg",
    "case6_severe_3d_torn_photo.png"
]

def estimate_sigma(gray: np.ndarray) -> float:
    med = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    mad = float(np.median(np.abs(gray.astype(np.float32) - med)))
    return 1.4826 * mad

print("=" * 100)
print("FULL 6-CASE DATASET COMPREHENSIVE SANITY SWEEP (PRODUCTION PIPELINE)")
print("=" * 100)

results = []
pipe = RestorationPipeline()

for fname in cases:
    fpath = os.path.join(p, fname)
    bgr = cv2.imread(fpath)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Test SpeckSuppressor standalone measurement & action
    ctx_speck = PipelineContext(rgb.copy())
    acted = SpeckSuppressor.suppress(ctx_speck)
    metrics = ctx_speck.image_metadata.get("speck_field", {})
    
    # 2. Pipeline sequence with proper metadata initialization
    ctx_pipe = PipelineContext(rgb.copy())
    ctx_pipe.pipeline_flags["enable_sr"] = True
    pipe._stage_0_despeckle(ctx_pipe, None)
    pipe._stage_1_analyze(ctx_pipe, None)
    faces = pipe._stage_4_face_detect(ctx_pipe, None)
    pipe._stage_7_face_restoration(ctx_pipe, None)
    deferred = ctx_pipe.pipeline_flags.get("face_restore_deferred_to_post_sr", False)

    # Damage detection & refinement via production Stage 2
    refined_mask = pipe._stage_2_fracture_masking(ctx_pipe, None)
    dmg_cov = float(np.mean(refined_mask > 0) * 100)
    
    # 3. Test Stage 5 adaptive denoising
    sig_before = estimate_sigma(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    pipe._stage_5_spatially_gated_denoise(ctx_pipe, faces, None)
    sig_after = estimate_sigma(cv2.cvtColor(ctx_pipe.working_image, cv2.COLOR_RGB2GRAY))
    
    denoise_log = [l for l in ctx_pipe.processing_log if "Stage 5 Denoise: Spatially-gated adaptive NLM" in l]
    h_params = denoise_log[0].split("(")[-1].split(")")[0] if denoise_log else "default"
    
    res = {
        "case": fname,
        "speck_acted": acted,
        "rho": metrics.get("spatial_autocorr", 0.0),
        "edge_coloc": metrics.get("edge_colocation", 0.0),
        "dmg_cov": dmg_cov,
        "faces": len(faces),
        "deferred": deferred,
        "sig_before": sig_before,
        "sig_after": sig_after,
        "h_params": h_params
    }
    results.append(res)
    
    print(f"\n[{fname}]")
    print(f"  SpeckSuppressor Acted : {acted} (rho={res['rho']:.3f}, edge_coloc={res['edge_coloc']:.1%})")
    print(f"  Damage Mask Coverage  : {dmg_cov:.2f}%")
    print(f"  Faces Detected        : {len(faces)} (Deferred to Post-SR: {deferred})")
    print(f"  Stage 5 Denoise       : sigma {sig_before:.2f} -> {sig_after:.2f} | {h_params}")

print("\n" + "=" * 115)
print("FINAL PIPELINE SANITY SWEEP SUMMARY TABLE")
print("=" * 115)
print(f"{'Case':<35} {'Speck':<7} {'Rho':<6} {'DamageCov':<10} {'Faces':<6} {'Deferred':<9} {'Sigma B/A':<14} {'Denoise Params'}")
print("-" * 115)
for r in results:
    sig_str = f"{r['sig_before']:.1f} -> {r['sig_after']:.1f}"
    dmg_str = f"{r['dmg_cov']:.2f}%"
    print(f"{r['case']:<35} {str(r['speck_acted']):<7} {r['rho']:<6.3f} {dmg_str:<10} {r['faces']:<6} {str(r['deferred']):<9} {sig_str:<14} {r['h_params']}")
print("=" * 115)

