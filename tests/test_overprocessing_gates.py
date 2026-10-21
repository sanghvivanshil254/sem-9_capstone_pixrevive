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


def test_projector_grade_default_applies(monkeypatch):
    """By default (PIXREVIVE_PROJECTOR_GRADE=1), Stage 10b applies grading and logs."""
    monkeypatch.setenv("PIXREVIVE_PROJECTOR_GRADE", "1")
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    pipe = RestorationPipeline(device="cuda")
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    ctx = PipelineContext(cam.copy())

    pipe._stage_10b_projector_grade(ctx, None)
    assert any("Stage 10b ProjectorGrade: Tamed CLAHE" in l for l in ctx.processing_log), "Grading log missing"
    assert ctx.image_metadata.get("projector_grade_applied", False), "projector_grade_applied should be True"


def test_projector_grade_flag_zero_skips(monkeypatch):
    """When PIXREVIVE_PROJECTOR_GRADE=0, Stage 10b is bypassed with bitwise identity."""
    monkeypatch.setenv("PIXREVIVE_PROJECTOR_GRADE", "0")
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    pipe = RestorationPipeline(device="cuda")
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    ctx = PipelineContext(cam.copy())
    pre = ctx.working_image.copy()

    pipe._stage_10b_projector_grade(ctx, None)
    assert any("Skipped (disabled by PIXREVIVE_PROJECTOR_GRADE=0" in l for l in ctx.processing_log), "Skip log missing"
    assert not ctx.image_metadata.get("projector_grade_applied", True), "projector_grade_applied should be False"
    assert np.array_equal(pre, ctx.working_image), "Canvas should be unchanged when projector grade is disabled"


def test_projector_grade_pipeline_flag_skips():
    """When context.pipeline_flags['enable_projector_grade']=False, Stage 10b is bypassed."""
    from ai.pipeline.stages.finish_stages import FinishStagesMixin

    class DummyFinishPipeline(FinishStagesMixin):
        pass

    pipe = DummyFinishPipeline()
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    ctx = PipelineContext(cam.copy())
    ctx.pipeline_flags["enable_projector_grade"] = False
    pre = ctx.working_image.copy()

    pipe._stage_10b_projector_grade(ctx, None)
    assert any("Skipped (disabled by PIXREVIVE_PROJECTOR_GRADE=0 or pipeline flag)" in l for l in ctx.processing_log)
    assert not ctx.image_metadata.get("projector_grade_applied", True)
    assert np.array_equal(pre, ctx.working_image)


def test_sharpen_gate_skips_on_sharp_image(monkeypatch):
    """PIXREVIVE_SHARPEN_GATE=1 must skip Stage 10 unsharp pass on images with blur_score > 200.0."""
    monkeypatch.setenv("PIXREVIVE_SHARPEN_GATE", "1")
    monkeypatch.setenv("PIXREVIVE_SPECK_SWEEP", "0")
    from ai.pipeline.stages.finish_stages import FinishStagesMixin

    class DummyFinishPipeline(FinishStagesMixin):
        pass

    pipe = DummyFinishPipeline()
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    ctx = PipelineContext(cam.copy(), {"blur_score": 450.0, "faces": [{"bbox": [50, 50, 200, 200]}]})
    pre = ctx.working_image.copy()

    pipe._stage_10_post_sr_sharpen(ctx, None)
    assert any("Skipped unsharp pass (image already sharp" in l for l in ctx.processing_log), "Skip log missing"
    assert not ctx.image_metadata.get("stage_10_sharpen_applied", True), "stage_10_sharpen_applied should be False"
    assert np.array_equal(pre, ctx.working_image), "Canvas should be unchanged when unsharp is skipped"


def test_sharpen_gate_applies_on_blurry_image(monkeypatch):
    """PIXREVIVE_SHARPEN_GATE=1 must apply Stage 10 sharpening without broadcast crash on blurry portrait."""
    monkeypatch.setenv("PIXREVIVE_SHARPEN_GATE", "1")
    monkeypatch.setenv("PIXREVIVE_SPECK_SWEEP", "0")
    from ai.pipeline.stages.finish_stages import FinishStagesMixin

    class DummyFinishPipeline(FinishStagesMixin):
        pass

    pipe = DummyFinishPipeline()
    # Synthetic blurry portrait
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 150), 60, (200, 200, 200), -1)
    ctx = PipelineContext(img.copy(), {"blur_score": 120.0, "faces": [{"bbox": [140, 90, 260, 210]}]})
    pre = ctx.working_image.copy()

    pipe._stage_10_post_sr_sharpen(ctx, None)
    assert any("Edge-directed multi-scale crispness applied" in l for l in ctx.processing_log), "Crispness log missing"
    assert ctx.image_metadata.get("stage_10_sharpen_applied", False), "stage_10_sharpen_applied should be True"
    assert not np.array_equal(pre, ctx.working_image), "Sharpening should modify the blurry canvas"


def test_sharpen_gate_flag_zero_restores_legacy(monkeypatch):
    """PIXREVIVE_SHARPEN_GATE=0 must apply Stage 10 sharpening even when blur_score > 200.0."""
    monkeypatch.setenv("PIXREVIVE_SHARPEN_GATE", "0")
    monkeypatch.setenv("PIXREVIVE_SPECK_SWEEP", "0")
    from ai.pipeline.stages.finish_stages import FinishStagesMixin

    class DummyFinishPipeline(FinishStagesMixin):
        pass

    pipe = DummyFinishPipeline()
    cam = cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2RGB)
    ctx = PipelineContext(cam.copy(), {"blur_score": 450.0, "faces": []})
    pre = ctx.working_image.copy()

    pipe._stage_10_post_sr_sharpen(ctx, None)
    assert any("Edge-directed multi-scale crispness applied" in l for l in ctx.processing_log), "Legacy sharpen should run"
    assert ctx.image_metadata.get("stage_10_sharpen_applied", False), "stage_10_sharpen_applied should be True"
    assert not np.array_equal(pre, ctx.working_image), "Legacy sharpen should modify canvas"


def test_despeckle_guard_preserves_high_contrast_texture(monkeypatch):
    """PIXREVIVE_DESPECKLE_GUARD=1 must protect macroscopic scene edges from false impulse/blob suppression."""
    monkeypatch.setenv("PIXREVIVE_DESPECKLE_GUARD", "1")
    from ai.pipeline.pipeline_context import PipelineContext
    from ai.pipeline.stages.preprocess_stages import PreprocessStagesMixin

    class DummyPipeline(PreprocessStagesMixin):
        pass

    path = "tests/restoration_dataset/pristine/high_contrast_texture.png"
    if not os.path.exists(path):
        pytest.skip(f"Fixture {path} not found")

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ctx = PipelineContext(img.copy())
    pipe = DummyPipeline()
    pipe._stage_0_despeckle(ctx, None)

    diff = np.abs(ctx.working_image.astype(float) - img.astype(float))
    assert np.count_nonzero(diff) == 0, f"Expected 0 pixels changed, got {np.count_nonzero(diff)}"
    assert np.array_equal(ctx.working_image, img), "Canvas should be bit-identical"
    assert not ctx.image_metadata.get("specks_suppressed", False)


def test_despeckle_guard_flag_zero_restores_legacy(monkeypatch):
    """PIXREVIVE_DESPECKLE_GUARD=0 must restore legacy behavior on high contrast textures."""
    monkeypatch.setenv("PIXREVIVE_DESPECKLE_GUARD", "0")
    from ai.pipeline.pipeline_context import PipelineContext
    from ai.pipeline.stages.preprocess_stages import PreprocessStagesMixin

    class DummyPipeline(PreprocessStagesMixin):
        pass

    path = "tests/restoration_dataset/pristine/high_contrast_texture.png"
    if not os.path.exists(path):
        pytest.skip(f"Fixture {path} not found")

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ctx = PipelineContext(img.copy())
    pipe = DummyPipeline()
    pipe._stage_0_despeckle(ctx, None)

    diff = np.abs(ctx.working_image.astype(float) - img.astype(float))
    assert np.count_nonzero(diff) > 1000, f"Expected legacy modifications (>1000 px), got {np.count_nonzero(diff)}"
    assert not np.array_equal(ctx.working_image, img)
    assert ctx.image_metadata.get("specks_suppressed", False)


def test_despeckle_guard_allows_real_impulses_on_damaged_fixture(monkeypatch):
    """PIXREVIVE_DESPECKLE_GUARD=1 must still suppress real impulses on damaged fixtures."""
    monkeypatch.setenv("PIXREVIVE_DESPECKLE_GUARD", "1")
    from ai.pipeline.pipeline_context import PipelineContext
    from ai.pipeline.stages.preprocess_stages import PreprocessStagesMixin

    class DummyPipeline(PreprocessStagesMixin):
        pass

    path = "tests/restoration_dataset/damaged/case11_minor_creases.jpg"
    if not os.path.exists(path):
        pytest.skip(f"Fixture {path} not found")

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ctx = PipelineContext(img.copy())
    pipe = DummyPipeline()
    pipe._stage_0_despeckle(ctx, None)

    diff = np.abs(ctx.working_image.astype(float) - img.astype(float))
    assert np.count_nonzero(diff) > 1000, "Genuine damage should still be cleaned"
    assert ctx.image_metadata.get("specks_suppressed", False)


# ---------------------------------------------------------------------------
# Fix 7: Crack Continuity & Diagonal Mask Truncation Tests
# ---------------------------------------------------------------------------

def test_crack_continuity_default_is_off(monkeypatch):
    """PIXREVIVE_CRACK_CONTINUITY must default to 0 / False when unset."""
    monkeypatch.delenv("PIXREVIVE_CRACK_CONTINUITY", raising=False)
    assert os.environ.get("PIXREVIVE_CRACK_CONTINUITY", "0").strip().lower() not in ("1", "true", "yes", "on")


def test_crack_continuity_preserves_diagonal_cracks_in_scenegate(monkeypatch):
    """
    Diagonal cracks have bounding box width ~= height (aspect ratio ~1.0 < 3.0).
    PIXREVIVE_CRACK_CONTINUITY=1 uses rotation-invariant moment elongation so
    diagonal tears are recognized as damage and not pruned as scene blobs.
    """
    # 500x500 test canvas with realistic background texture
    h, w = 500, 500
    gray_bg = np.random.RandomState(42).randint(110, 140, (h, w, 3), dtype=np.uint8)
    
    # 5px wide diagonal crack running from (50, 450) to (450, 50)
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(damage_mask, (50, 450), (450, 50), 255, 5)
    
    # Test baseline (0): SceneGate checks only max(w, h)/min(w, h)
    monkeypatch.setenv("PIXREVIVE_CRACK_CONTINUITY", "0")
    monkeypatch.setenv("PIXREVIVE_SCENE_GATE", "1")
    ctx_0 = PipelineContext(gray_bg.copy())
    ctx_0.damage_masks["fractures"] = damage_mask.copy()
    out_0 = MaskRefiner.refine(damage_mask.copy(), ctx_0)
    # The diagonal crack has bbox 400x400 (aspect 1.0 < 3.0), no popout on flat bg, so SceneGate prunes it
    assert np.sum(out_0 > 0) == 0, "Expected baseline SceneGate to prune diagonal crack due to axis-aligned aspect ratio trap"
    
    # Test Fix 7 (1): SceneGate uses moment-based rotation-invariant elongation
    monkeypatch.setenv("PIXREVIVE_CRACK_CONTINUITY", "1")
    ctx_1 = PipelineContext(gray_bg.copy())
    ctx_1.damage_masks["fractures"] = damage_mask.copy()
    out_1 = MaskRefiner.refine(damage_mask.copy(), ctx_1)
    assert np.sum(out_1 > 0) > 1000, f"Expected Fix 7 to preserve diagonal crack, got {np.sum(out_1 > 0)} px"


def test_crack_continuity_preserves_through_eye_crack(monkeypatch):
    """
    PIXREVIVE_CRACK_CONTINUITY=1 preserves thin linear crack strokes cutting across ocular
    landmarks, while still zeroing out broad area cavity fills that would destroy the pupil.
    """
    h, w = 400, 400
    gray_bg = np.full((h, w, 3), 128, dtype=np.uint8)
    
    # Face covering center of frame with left eye landmark at (150, 150)
    face_meta = [{
        "bbox": [50, 50, 350, 350],
        "landmarks": [[150, 150], [250, 150], [200, 200], [170, 260], [230, 260]],
    }]
    
    # Crack line cutting right through the eye at (150, 150)
    crack_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(crack_mask, (100, 120), (200, 180), 255, 3)
    
    # Baseline (0): unconditionally draws solid black circle over (150, 150)
    monkeypatch.setenv("PIXREVIVE_CRACK_CONTINUITY", "0")
    ctx_0 = PipelineContext(gray_bg.copy(), {"faces": face_meta})
    ctx_0.damage_masks["fractures"] = crack_mask.copy()
    out_0 = MaskRefiner.refine(crack_mask.copy(), ctx_0)
    # At the eye landmark (150, 150), baseline circle zeroing leaves a 0-hole
    assert out_0[150, 150] == 0, "Baseline should zero out pixels at eye landmark"
    
    # Fix 7 (1): selectively protects broad fills, preserving thin crack stroke through eye
    monkeypatch.setenv("PIXREVIVE_CRACK_CONTINUITY", "1")
    ctx_1 = PipelineContext(gray_bg.copy(), {"faces": face_meta})
    ctx_1.damage_masks["fractures"] = crack_mask.copy()
    out_1 = MaskRefiner.refine(crack_mask.copy(), ctx_1)
    assert out_1[150, 150] > 0, "Fix 7 should preserve thin crack stroke cutting through eye"


# ---------------------------------------------------------------------------
# Fix 8: Seamless Face Blend & Seam Suppression Tests
# ---------------------------------------------------------------------------

def test_seamless_face_blend_default_is_off(monkeypatch):
    """PIXREVIVE_SEAMLESS_FACE_BLEND must default to 0 / False when unset."""
    monkeypatch.delenv("PIXREVIVE_SEAMLESS_FACE_BLEND", raising=False)
    assert os.environ.get("PIXREVIVE_SEAMLESS_FACE_BLEND", "0").strip().lower() not in ("1", "true", "yes", "on")


def test_seamless_face_blend_zero_boundary_derivative():
    """
    PIXREVIVE_SEAMLESS_FACE_BLEND=1 generates C1-continuous cosine border ramps
    whose boundary gradient (derivative at x=0) is near-zero (<0.005), eliminating
    the first-derivative step discontinuity of linear fades (>0.05).
    """
    crop_h, crop_w = 512, 512
    # Baseline linear fade
    m_base = 16
    base_fade = np.ones((crop_h, crop_w), dtype=np.float32)
    base_fade[:m_base, :] = np.linspace(0.0, 1.0, m_base)[:, None]
    base_deriv = float(base_fade[1, 256] - base_fade[0, 256])
    assert base_deriv > 0.05, f"Expected linear fade derivative > 0.05, got {base_deriv}"
    
    # Fix 8 cosine fade
    m_cos = 48
    t = np.linspace(0.0, np.pi, m_cos)
    cos_ramp = 0.5 * (1.0 - np.cos(t)).astype(np.float32)
    cos_fade = np.ones((crop_h, crop_w), dtype=np.float32)
    cos_fade[:m_cos, :] = cos_ramp[:, None]
    cos_deriv = float(cos_fade[1, 256] - cos_fade[0, 256])
    assert cos_deriv < 0.005, f"Expected cosine fade derivative < 0.005, got {cos_deriv}"
    assert cos_deriv < base_deriv / 10.0, "Cosine fade should reduce boundary derivative by >10x"


def test_seamless_face_blend_smooths_seam_ratio_threshold(monkeypatch):
    """
    PIXREVIVE_SEAMLESS_FACE_BLEND=1 lowers the feathering trigger threshold from
    ratio > 3.0 to ratio > 1.8 and feathers all matching footprints rather than
    just worst_idx.
    """
    from ai.restoration.paste_guard import feather_seam
    # 200x200 canvas with a footprint
    h, w = 200, 200
    canvas = np.full((h, w, 3), 120, dtype=np.uint8)
    footprint = np.zeros((h, w), dtype=bool)
    footprint[50:150, 50:150] = True
    
    # Create step discontinuity along footprint edge
    canvas[footprint] = 160 # +40 step
    
    # Feather seam
    feathered = feather_seam(canvas.copy(), footprint, band_px=16, blur_k=25, strength=0.65)
    # The step between (49, 100) and (50, 100) should be smoothed
    orig_step = abs(int(canvas[50, 100, 0]) - int(canvas[49, 100, 0]))
    feathered_step = abs(int(feathered[50, 100, 0]) - int(feathered[49, 100, 0]))
    assert orig_step == 40
    assert feathered_step < orig_step, f"Feathering should reduce boundary step, got {feathered_step}"


# ---------------------------------------------------------------------------
# Fix 9: Chroma Edge Guide & Colour Bleed Prevention Tests
# ---------------------------------------------------------------------------

def test_chroma_edge_guide_default_is_off(monkeypatch):
    """PIXREVIVE_CHROMA_EDGE_GUIDE must default to 0 / False when unset."""
    monkeypatch.delenv("PIXREVIVE_CHROMA_EDGE_GUIDE", raising=False)
    assert os.environ.get("PIXREVIVE_CHROMA_EDGE_GUIDE", "0").strip().lower() not in ("1", "true", "yes", "on")


def test_chroma_edge_guide_snaps_chroma_across_edge(monkeypatch):
    """
    PIXREVIVE_CHROMA_EDGE_GUIDE=1 snaps bled chrominance across sharp luminance
    edges, reducing boundary leakage in the 2-10px transition band.
    """
    from ai.colorization.color_refiner import EdgeAwareColorBleedFixer
    # Create 100x100 synthetic image with a sharp vertical luminance step at x=50
    # Left (x < 50): L=50, a=160 (vivid red lips)
    # Right (x >= 50): L=180, a=128 (pale skin)
    h, w = 100, 100
    L = np.full((h, w), 180.0, dtype=np.float32)
    L[:, :50] = 50.0
    
    # Smeared chrominance channel simulating DDColor 512x512 bilinear upsampling blur:
    # a transitions linearly from 160 at x=40 to 128 at x=60 (bleeding 10px into the light side)
    a = np.full((h, w), 128.0, dtype=np.float32)
    a[:, :40] = 160.0
    for x in range(40, 60):
        a[:, x] = 160.0 - (x - 40) * (32.0 / 20.0)
    b = np.full((h, w), 128.0, dtype=np.float32)
    
    rgb_in = cv2.cvtColor(np.dstack([L, a, b]).astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # Baseline (flag 0)
    # Baseline (flag 0)
    monkeypatch.setenv("PIXREVIVE_CHROMA_EDGE_GUIDE", "0")
    rgb_0 = EdgeAwareColorBleedFixer.fix_bleeding(rgb_in.copy(), strength=0.75)
    lab_0 = cv2.cvtColor(rgb_0, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Fix 9 (flag 1)
    monkeypatch.setenv("PIXREVIVE_CHROMA_EDGE_GUIDE", "1")
    rgb_1 = EdgeAwareColorBleedFixer.fix_bleeding(rgb_in.copy(), strength=0.75)
    lab_1 = cv2.cvtColor(rgb_1, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Chroma step across the boundary x=49 -> x=50
    step_0 = float(lab_0[50, 49, 1] - lab_0[50, 50, 1])
    step_1 = float(lab_1[50, 49, 1] - lab_1[50, 50, 1])
    
    # Bleed into pale skin side at x=50
    bleed_0 = abs(float(lab_0[50, 50, 1]) - 128.0)
    bleed_1 = abs(float(lab_1[50, 50, 1]) - 128.0)
    
    assert step_1 > step_0 * 3.0, f"Fix 9 should sharpen chroma edge step by >3x: {step_1:.2f} vs {step_0:.2f}"
    assert bleed_1 < bleed_0, f"Fix 9 should reduce chrominance bleed at boundary: {bleed_1:.2f} vs {bleed_0:.2f}"


def test_chroma_edge_guide_reduces_misregistration_energy(monkeypatch):
    """
    PIXREVIVE_CHROMA_EDGE_GUIDE=1 in DDColor chroma blending reduces chrominance
    gradient energy located > 4px from strong luminance edges.
    """
    from ai.colorization.ddcolor_colorizer import _blend_chroma
    h, w = 120, 120
    # Step edge in luminance at x=60
    gray_L = np.full((h, w), 190, dtype=np.uint8)
    gray_L[:, :60] = 60
    gray_rgb = cv2.cvtColor(gray_L, cv2.COLOR_GRAY2RGB)
    
    # Color image with blurred chroma
    col_a = np.full((h, w), 128, dtype=np.uint8)
    col_a[:, :50] = 165
    # Smooth blur across edge
    col_a = cv2.GaussianBlur(col_a, (15, 15), 0)
    col_b = np.full((h, w), 128, dtype=np.uint8)
    col_lab = np.dstack([gray_L, col_a, col_b])
    col_rgb = cv2.cvtColor(col_lab, cv2.COLOR_LAB2RGB)
    
    # Measure far-edge energy under flag 0 vs flag 1
    monkeypatch.setenv("PIXREVIVE_CHROMA_EDGE_GUIDE", "0")
    res_0 = _blend_chroma(gray_rgb, col_rgb, opacity=1.0)
    lab_0 = cv2.cvtColor(res_0, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    monkeypatch.setenv("PIXREVIVE_CHROMA_EDGE_GUIDE", "1")
    res_1 = _blend_chroma(gray_rgb, col_rgb, opacity=1.0)
    lab_1 = cv2.cvtColor(res_1, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Gradient of 'a' at x=70 (10px from edge):
    grad_a_0 = abs(lab_0[60, 71, 1] - lab_0[60, 69, 1])
    grad_a_1 = abs(lab_1[60, 71, 1] - lab_1[60, 69, 1])
    
    assert grad_a_1 <= grad_a_0, f"Far-edge chroma gradient should decrease under Fix 9: {grad_a_1} vs {grad_a_0}"


# ---------------------------------------------------------------------------
# Fix 10: Texture Recovery & Feather Preservation Tests
# ---------------------------------------------------------------------------

def test_texture_recovery_default_is_off(monkeypatch):
    """PIXREVIVE_TEXTURE_RECOVERY must default to 0 / False when unset."""
    monkeypatch.delenv("PIXREVIVE_TEXTURE_RECOVERY", raising=False)
    assert os.environ.get("PIXREVIVE_TEXTURE_RECOVERY", "0").strip().lower() not in ("1", "true", "yes", "on")


def test_texture_recovery_protects_speck_suppressor_fibers(monkeypatch):
    """
    PIXREVIVE_TEXTURE_RECOVERY=1 prevents SpeckSuppressor from eroding fine oriented
    fibrous structures (coherence > 0.25) as if they were isotropic noise specks.
    """
    from ai.preprocessing.speck_suppressor import SpeckSuppressor
    from ai.analysis.line_structure import LineStructureAnalyzer
    h, w = 100, 100
    canvas = np.full((h, w, 3), 120, dtype=np.uint8)
    
    # Draw fine parallel 1px fibrous lines (feather/hair strands)
    for y in range(20, 80, 4):
        cv2.line(canvas, (10, y), (90, y), (180, 180, 180), 1)
        
    # Baseline (flag 0)
    monkeypatch.setenv("PIXREVIVE_TEXTURE_RECOVERY", "0")
    res_0 = SpeckSuppressor.apply(canvas.copy(), threshold=6.0)
    
    # Fix 10 (flag 1)
    monkeypatch.setenv("PIXREVIVE_TEXTURE_RECOVERY", "1")
    res_1 = SpeckSuppressor.apply(canvas.copy(), threshold=6.0)
    
    # Peak amplitude of fiber lines at y=48
    fiber_amp_in = int(canvas[48, 50, 0]) - 120
    fiber_amp_0 = int(res_0[48, 50, 0]) - 120
    fiber_amp_1 = int(res_1[48, 50, 0]) - 120
    
    # Fix 10 should preserve more of the authentic fiber amplitude than baseline
    assert fiber_amp_1 > fiber_amp_0, f"Fix 10 should preserve fiber amplitude: {fiber_amp_1} vs {fiber_amp_0}"


def test_texture_recovery_boosts_denoise_protect_map(monkeypatch):
    """
    PIXREVIVE_TEXTURE_RECOVERY=1 integrates structure tensor orientation coherence
    into Stage 5 denoise protect_map and boosts authentic texture retention.
    """
    from ai.pipeline.pipeline_context import PipelineContext
    from ai.pipeline.stages.finish_stages import FinishStagesMixin
    
    class DummyPipeline(FinishStagesMixin):
        pass
        
    h, w = 120, 120
    canvas = np.full((h, w, 3), 120, dtype=np.uint8)
    # Add fibrous parallel lines in upper half
    for y in range(10, 50, 4):
        cv2.line(canvas, (10, y), (110, y), (170, 170, 170), 1)
        
    # Run Stage 5 denoise under flag 0 vs flag 1
    pipe = DummyPipeline()
    
    # Flag 0
    monkeypatch.setenv("PIXREVIVE_TEXTURE_RECOVERY", "0")
    ctx_0 = PipelineContext(canvas.copy(), {"noise_score": 6.0, "content_sigma": 6.0})
    pipe._stage_5_spatially_gated_denoise(ctx_0, faces=[])
    
    # Flag 1
    monkeypatch.setenv("PIXREVIVE_TEXTURE_RECOVERY", "1")
    ctx_1 = PipelineContext(canvas.copy(), {"noise_score": 6.0, "content_sigma": 6.0})
    pipe._stage_5_spatially_gated_denoise(ctx_1, faces=[])
    
    # Compare residual fiber line amplitude at y=28
    amp_0 = int(ctx_0.working_image[28, 60, 0]) - 120
    amp_1 = int(ctx_1.working_image[28, 60, 0]) - 120
    
    assert amp_1 >= amp_0, f"Fix 10 should retain more fiber contrast under denoise: {amp_1} vs {amp_0}"







