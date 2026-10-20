"""Regression test suite for Fix 1: Structure Discriminator and Overprocessing Gates.

Tests:
1. Pristine controls (8 fixtures) do not route to DAMAGED_* (needs_damage_repair == False).
2. Damaged fixtures route to DAMAGED_* (needs_damage_repair == True) or appropriate restoration.
3. Hair strands crop: protected from inpaint mask by LineStructureAnalyzer in MaskRefiner.
4. case11_minor_creases: robust_sigma mottle rescue does not classify crease web as grain,
   preserving mask coverage in [0.5%, 25.0%].
"""

import os
import sys
import pytest
import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
backend_path = os.path.join(REPO_ROOT, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.pipeline_router import PipelineRouter
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.analysis.face_detector import FaceDetector
from ai.analysis.line_structure import LineStructureAnalyzer
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.mask_refiner import MaskRefiner
from tests.test_control_fixtures import _clean_photo
import skimage.data


PRISTINE_DIR = os.path.join(REPO_ROOT, "tests", "restoration_dataset", "pristine")
DAMAGED_DIR = os.path.join(REPO_ROOT, "tests", "restoration_dataset", "damaged")


def _load_img(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Fixture not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _analyze_and_route(img_rgb: np.ndarray):
    qa = QualityAnalyzer()
    meta = qa.analyze(img_rgb)
    fd = FaceDetector()
    try:
        faces = fd.detect(img_rgb)
    except Exception:
        faces = []
    meta["faces"] = faces
    ctx = PipelineContext(img_rgb, meta)
    router = PipelineRouter()
    route = router.route(ctx)
    route_name = route.get("pipeline_name", "UNKNOWN")
    return ctx, route, route_name, meta


@pytest.mark.parametrize("name,loader", [
    ("control_1_clean_synthetic_portrait", lambda: _clean_photo()),
    ("control_2_pristine_mono_camera", lambda: cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)),
    ("control_3_pristine_color_astronaut", lambda: skimage.data.astronaut()),
    ("control_4_pristine_color_chelsea", lambda: skimage.data.chelsea()),
    ("hair_strands_crop", lambda: _load_img(os.path.join(PRISTINE_DIR, "hair_strands_crop.png"))),
    ("real_hair_portrait", lambda: _load_img(os.path.join(PRISTINE_DIR, "real_hair_portrait.png"))),
    ("old_undamaged_scan", lambda: _load_img(os.path.join(PRISTINE_DIR, "old_undamaged_scan.png"))),
    ("high_contrast_texture", lambda: _load_img(os.path.join(PRISTINE_DIR, "high_contrast_texture.png"))),
])
def test_pristine_controls_skip_damage_repair(name, loader):
    """All 8 pristine fixtures must NOT be routed to DAMAGED_* or have damage repair enabled."""
    img = loader()
    ctx, route, route_name, meta = _analyze_and_route(img)
    needs_repair = bool(route.get("run_damage_repair", False))
    assert not needs_repair, f"Pristine control {name} routed to damage repair: {route_name} (meta: {meta})"
    assert not route_name.startswith("damaged_"), f"Pristine control {name} assigned damaged route: {route_name}"


@pytest.mark.parametrize("damaged_file", [
    "case3_sepia_flaked_portrait.jpg",
    "case4_diagonal_tear_portrait.jpg",
    "case5_spiderweb_creases_portrait.jpg",
    "case6_severe_3d_torn_photo.png",
    "case8_speckle_lena_margins.png",
    "case9_teared_sepia.jpg",
    "case10_spiderweb_lips.jpg",
    "case11_minor_creases.jpg",
    "case12_foxing_spots.png",
])
def test_genuine_damaged_fixtures_keep_damage_repair(damaged_file):
    """Genuine damaged images must be routed to damage repair."""
    path = os.path.join(DAMAGED_DIR, damaged_file)
    if not os.path.exists(path):
        pytest.skip(f"Fixture {damaged_file} not found")
    img = _load_img(path)
    ctx, route, route_name, meta = _analyze_and_route(img)
    needs_repair = bool(route.get("run_damage_repair", False))
    assert needs_repair, f"Damaged fixture {damaged_file} failed to route to damage repair: {route_name}"


def test_hair_strands_crop_inpaint_mask_protected():
    """Fine hair strands must be protected from LaMa inpaint mask refinement."""
    path = os.path.join(PRISTINE_DIR, "hair_strands_crop.png")
    img = _load_img(path)
    ctx, route, route_name, meta = _analyze_and_route(img)

    det = EnhancedDamageDetector(use_defect_segmenter=False)
    crack_mask = det.detect_thin_crack_network(img, ctx)
    scratch_mask = det.detect_scratches(img, ctx)
    combined = cv2.bitwise_or(crack_mask, scratch_mask)

    refiner = MaskRefiner()
    refined = refiner.refine(combined, ctx)
    cov = float(np.sum(refined > 0) / refined.size)
    assert cov < 0.005, f"Hair strands crop has excessive inpaint mask coverage: {cov*100:.3f}%"


def test_case11_minor_creases_robust_sigma_and_coverage():
    """case11 crease web must not be mistaken for grain, preserving crack mask in [0.5%, 25.0%]."""
    path = os.path.join(DAMAGED_DIR, "case11_minor_creases.jpg")
    img = _load_img(path)

    # 1. Verify robust_sigma in content_region stayed low (not grain)
    from ai.analysis.content_region import robust_sigma
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sigma = robust_sigma(gray)
    assert sigma <= 3.5, f"case11 sigma blew up to {sigma:.2f} due to crease misclassification"

    # 2. Verify damage mask coverage in [0.5%, 25.0%] through full Stage 2
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    pipe = RestorationPipeline(device="cpu")
    ctx = PipelineContext(img.copy())
    pipe._stage_1_analyze(ctx, None)
    mask = pipe._stage_2_fracture_masking(ctx, None)
    cov_pct = 100.0 * float(np.count_nonzero(mask)) / float(mask.size)
    assert 0.5 <= cov_pct <= 25.0, f"case11 mask coverage {cov_pct:.3f}% outside target band [0.5%, 25.0%]"


def test_sepia_hue_uniformity_chelsea_and_real_hair_are_color():
    """Multi-hue color photos with warm palettes must not be misclassified as sepia."""
    qa = QualityAnalyzer()
    
    # 1. Chelsea (cat photo with warm fur, green eyes, distinct hues)
    chelsea = skimage.data.chelsea()
    meta_ch = qa.analyze(chelsea)
    assert meta_ch.get("color_type") == "color", f"Chelsea falsely classified as: {meta_ch.get('color_type')}"
    assert meta_ch.get("chrom_hue_std", 0.0) > 8.0, f"Chelsea hue std too low: {meta_ch.get('chrom_hue_std')}"

    # 2. Real hair portrait (brunette portrait with skin and lips)
    hair_p = os.path.join(PRISTINE_DIR, "real_hair_portrait.png")
    if os.path.exists(hair_p):
        meta_hp = qa.analyze(_load_img(hair_p))
        assert meta_hp.get("color_type") == "color", f"Real hair falsely classified as: {meta_hp.get('color_type')}"
        assert meta_hp.get("chrom_hue_std", 0.0) > 8.0, f"Real hair hue std too low: {meta_hp.get('chrom_hue_std')}"

    # 3. Genuine sepia fixtures must remain sepia
    for s_file in ["case3_sepia_flaked_portrait.jpg", "case9_teared_sepia.jpg"]:
        p = os.path.join(DAMAGED_DIR, s_file)
        if os.path.exists(p):
            meta_s = qa.analyze(_load_img(p))
            assert meta_s.get("color_type") == "sepia", f"{s_file} should be sepia, got: {meta_s.get('color_type')}"
            assert meta_s.get("chrom_hue_std", 0.0) <= 8.0, f"{s_file} hue std too high: {meta_s.get('chrom_hue_std')}"


def test_option_3_unblocks_case9_colorization():
    """Option 3 must keep case9_teared_sepia colorized instead of rolling back to grayscale."""
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    path = os.path.join(DAMAGED_DIR, "case9_teared_sepia.jpg")
    img = _load_img(path)
    qa = QualityAnalyzer()
    meta = qa.analyze(img)
    pipe = RestorationPipeline(device="cuda")
    ctx = PipelineContext(img.copy(), meta)
    pipe._stage_6_colorization(ctx, None)
    assert ctx.image_metadata.get("colorization_applied", False), "Colorization was not applied to case9"
    assert not ctx.image_metadata.get("colorization_rolled_back", False), "Colorization was rolled back on case9"


def test_denoise_floor_gate_bypasses_negligible_noise():
    """Pristine image with sigma <= 2.5 and noise_score <= 2.5 must bypass Stage 5 NLM completely."""
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    pipe = RestorationPipeline(device="cuda")
    clean = np.full((100, 100, 3), 128, dtype=np.uint8)
    qa = QualityAnalyzer()
    meta = qa.analyze(clean)
    ctx = PipelineContext(clean.copy(), meta)
    pre = ctx.working_image.copy()

    pipe._stage_5_spatially_gated_denoise(ctx, [], None)
    assert any("Bypassed by noise floor gate" in l for l in ctx.processing_log), "Floor gate bypass log missing"
    assert not ctx.image_metadata.get("denoise_applied", True), "denoise_applied should be False when bypassed"
    assert np.array_equal(pre, ctx.working_image), "Canvas should be bitwise identical when bypassed"


def test_denoise_floor_gate_scales_down_h_on_camera(monkeypatch):
    """Pristine camera image must scale h down proportionally to sigma (1.48) and keep HF >= 0.95."""
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    pipe = RestorationPipeline(device="cuda")
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    qa = QualityAnalyzer()
    meta = qa.analyze(cam)
    ctx = PipelineContext(cam.copy(), meta)
    pre = ctx.working_image.copy()

    pipe._stage_5_spatially_gated_denoise(ctx, [], None)
    logs = [l for l in ctx.processing_log if "h_face=" in l]
    assert logs, "Stage 5 log missing"
    import re
    h_bg = float(re.search(r"h_bg=([\d.]+)", logs[-1]).group(1))
    assert h_bg <= 2.5, f"Expected scaled h_bg <= 2.5, got {h_bg}"

    # Calculate HF retention
    g0 = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(ctx.working_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    v0 = float(np.var(cv2.Laplacian(g0, cv2.CV_32F)))
    v1 = float(np.var(cv2.Laplacian(g1, cv2.CV_32F)))
    hf = v1 / max(v0, 1e-6)
    assert hf >= 0.95, f"Expected HF retention >= 0.95, got {hf:.3f}"


def test_denoise_floor_gate_flag_zero_restores_legacy(monkeypatch):
    """PIXREVIVE_DENOISE_FLOOR_GATE=0 must restore legacy clamped floors (h_bg >= 6.5, h_mid >= 7.5)."""
    monkeypatch.setenv("PIXREVIVE_DENOISE_FLOOR_GATE", "0")
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    pipe = RestorationPipeline(device="cuda")
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    qa = QualityAnalyzer()
    meta = qa.analyze(cam)
    ctx = PipelineContext(cam.copy(), meta)

    pipe._stage_5_spatially_gated_denoise(ctx, [], None)
    logs = [l for l in ctx.processing_log if "h_face=" in l]
    assert logs, "Stage 5 log missing"
    import re
    h_bg = float(re.search(r"h_bg=([\d.]+)", logs[-1]).group(1))
    assert h_bg >= 6.5, f"Legacy mode should clamp h_bg >= 6.5, got {h_bg}"
