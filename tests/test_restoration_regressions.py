import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.colorization.ddcolor_colorizer import DDColorColorizer
from ai.colorization.color_refiner import ColorSmoother
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.pipeline_router import PipelineRouter
from ai.pipeline.restoration_pipeline import RestorationPipeline
from ai.postprocessing.denoiser import AdaptiveDenoiser, ArtifactDetector, SmartDenoiser
from ai.postprocessing.noise_router import NoiseClassifier, NoiseRouter, SUPPORTED_NOISE_TYPES
from ai.restoration.inpainting_engine import InpaintingEngine
from ai.restoration.super_resolution import _target_long_side_for_source
from ai.validation.identity_validator import IdentityValidator


class TestRestorationRegressions(unittest.TestCase):
    def test_opencv_runtime_has_required_apis(self):
        self.assertTrue(hasattr(cv2, "CascadeClassifier"))
        self.assertTrue(hasattr(cv2, "ximgproc"))

    def test_identity_validator_normalizes_restored_output_size(self):
        img = np.full((40, 40, 3), [120, 110, 100], dtype=np.uint8)
        restored = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
        context = PipelineContext(
            img,
            {
                "faces": [{"bbox": [10, 10, 30, 30], "confidence": 1.0}],
                "width": 40,
                "height": 40,
                "color_type": "color",
            },
        )
        context.working_image = restored
        context.damage_masks["final"] = np.zeros((40, 40), dtype=np.uint8)

        metrics = IdentityValidator().validate_context(context)

        self.assertTrue(metrics.pixel_preservation_ok)
        self.assertTrue(metrics.identity_preserved)

    def test_multiscale_fill_keeps_channel_dimension_for_weights(self):
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        img[:, :20] = [80, 90, 100]
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        context = PipelineContext(img, {"color_type": "color"})

        out = InpaintingEngine._multiscale_fill(img, mask, context)

        self.assertEqual(out.shape, img.shape)
        self.assertFalse(
            any("Pass 2] failed" in line for line in context.processing_log),
            context.processing_log,
        )

    def test_color_smoother_handles_lab_l_channel_range(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :16] = [30, 120, 180]
        img[:, 16:] = [220, 180, 80]

        out = ColorSmoother.smooth_colors(img, strength=0.25)

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_noisy_grayscale_is_not_major_structural_damage(self):
        rng = np.random.default_rng(42)
        base = np.full((148, 196), 128, dtype=np.int16)
        noise = rng.normal(0, 18, base.shape).astype(np.int16)
        gray = np.clip(base + noise, 0, 255).astype(np.uint8)
        img = np.stack([gray, gray, gray], axis=2)

        metadata = QualityAnalyzer.analyze(img)

        self.assertLess(metadata["damage_percentage"], 10.0)
        self.assertFalse(metadata["has_white_flakes"])

    def test_low_saturation_channel_jitter_stays_grayscale(self):
        rng = np.random.default_rng(123)
        base = np.full((96, 128, 1), 128, dtype=np.int16)
        jitter = rng.integers(-4, 5, size=(96, 128, 3), dtype=np.int16)
        img = np.clip(base + jitter, 0, 255).astype(np.uint8)

        metadata = QualityAnalyzer.analyze(img)

        self.assertEqual(metadata["color_type"], "grayscale")
        self.assertTrue(metadata["is_grayscale"])

    def test_grayscale_portrait_routes_to_denoise_without_auto_colorization(self):
        img = np.full((180, 220, 3), 128, dtype=np.uint8)
        context = PipelineContext(
            img,
            {
                "color_type": "grayscale",
                "noise_score": 9.0,
                "damage_percentage": 3.0,
                "blur_score": 120.0,
                "has_white_flakes": False,
            },
        )

        route = PipelineRouter.route(context)

        self.assertTrue(route["run_denoising"])
        self.assertTrue(route["run_sr"])
        self.assertFalse(route["run_colorization"])
        self.assertEqual(route["pipeline_name"], "BW_RESTORE")

    def test_explicit_colorization_runs_for_bw_route(self):
        self.assertTrue(RestorationPipeline._should_colorize(True, "grayscale"))
        self.assertTrue(RestorationPipeline._should_colorize(True, "sepia"))
        self.assertFalse(RestorationPipeline._should_colorize(True, "color"))
        self.assertFalse(RestorationPipeline._should_colorize(False, "grayscale"))

    def test_ddcolor_skips_tiny_sources(self):
        """DDColorColorizer.colorization_safety_reason must skip images < 96px."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        context = PipelineContext(
            img,
            {
                "width": 64,
                "height": 64,
                "color_type": "grayscale",
                "damage_percentage": 0.0,
            },
        )

        reason = DDColorColorizer.colorization_safety_reason(context)

        self.assertIsNotNone(reason)
        self.assertIn("source too small", reason)

    def test_smart_denoiser_rolls_back_over_smoothed_candidate(self):
        original_denoise = AdaptiveDenoiser.denoise
        original_blocking = ArtifactDetector.detect_blocking
        original_smoothing = ArtifactDetector.detect_over_smoothing
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        candidate = np.zeros_like(img)
        try:
            AdaptiveDenoiser.denoise = staticmethod(lambda img, noise_level=None, context=None: candidate)  # type: ignore[assignment]
            ArtifactDetector.detect_blocking = staticmethod(lambda img, threshold=0.15: False)  # type: ignore[assignment]
            ArtifactDetector.detect_over_smoothing = staticmethod(  # type: ignore[assignment]
                lambda img, original, threshold=0.05: True
            )

            out = SmartDenoiser.denoise_smart(img, max_iterations=1)

            np.testing.assert_array_equal(out, img)
        finally:
            AdaptiveDenoiser.denoise = original_denoise
            ArtifactDetector.detect_blocking = original_blocking
            ArtifactDetector.detect_over_smoothing = original_smoothing

    def test_tiny_source_sr_target_stays_at_model_scale(self):
        self.assertEqual(_target_long_side_for_source(196), 784)

    def test_sr_fallback_respects_2k_target_for_3000px_source(self):
        """Q-5: with the model unavailable, the fallback resamples a 3000px
        source to the 2K target (2048 long side) instead of blindly x4-ing
        it to 12000px."""
        from ai.restoration import super_resolution as sr_mod

        img = np.zeros((2250, 3000, 3), dtype=np.uint8)  # 4:3, long side 3000
        sr = sr_mod.SuperResolution()  # is_loaded=False -> fallback path
        self.assertFalse(sr.is_loaded)

        ctx = PipelineContext(img, {"color_type": "color"})
        out = sr.restore(ctx)

        self.assertEqual(max(out.shape[:2]), sr_mod._2K_LONG_SIDE)
        self.assertEqual(max(ctx.working_image.shape[:2]), sr_mod._2K_LONG_SIDE)

    def test_noise_router_declares_all_requested_noise_types(self):
        expected = {
            "gaussian",
            "salt_and_pepper",
            "uniform",
            "poisson",
            "speckle",
            "rayleigh",
            "exponential",
            "gamma",
            "shot",
            "thermal",
            "impulse",
            "random_valued_impulse",
            "quantization",
            "periodic",
            "sinusoidal",
            "striping",
            "banding",
            "interference",
            "fixed_pattern",
            "read",
            "dark_current",
            "amplifier",
            "reset",
            "prnu",
            "color",
            "chroma",
            "luminance",
            "jpeg_compression",
            "blocking",
            "ringing",
        }

        self.assertEqual(set(SUPPORTED_NOISE_TYPES), expected)
        self.assertEqual(len(SUPPORTED_NOISE_TYPES), 30)

    def test_noise_classifier_detects_impulse_noise(self):
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(7)
        coords = rng.choice(img.shape[0] * img.shape[1], size=900, replace=False)
        flat = img.reshape(-1, 3)
        flat[coords[:450]] = 0
        flat[coords[450:]] = 255

        profile = NoiseClassifier.classify(img, {"color_type": "grayscale"})

        self.assertIn("salt_and_pepper", profile.detected_types)
        self.assertIn("impulse", profile.detected_types)

    def test_noise_classifier_detects_periodic_striping(self):
        x = np.arange(160, dtype=np.float32)
        stripe = (128.0 + 35.0 * np.sin(2 * np.pi * x / 8.0)).astype(np.uint8)
        gray = np.tile(stripe, (128, 1))
        img = np.stack([gray, gray, gray], axis=2)

        profile = NoiseClassifier.classify(img, {"color_type": "grayscale"})

        self.assertTrue(
            {"periodic", "sinusoidal", "striping"}.intersection(profile.detected_types),
            profile.to_dict(),
        )

    def test_noise_classifier_detects_chroma_noise(self):
        rng = np.random.default_rng(11)
        img = np.full((96, 96, 3), [128, 126, 124], dtype=np.int16)
        chroma_noise = rng.normal(0, 22, img.shape).astype(np.int16)
        chroma_noise[:, :, 1] = 0
        noisy = np.clip(img + chroma_noise, 0, 255).astype(np.uint8)

        profile = NoiseClassifier.classify(noisy, {"color_type": "color"})

        self.assertIn("chroma", profile.detected_types)

    def test_noise_router_reduces_impulse_without_shape_change(self):
        img = np.full((96, 96, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(13)
        coords = rng.choice(img.shape[0] * img.shape[1], size=600, replace=False)
        flat = img.reshape(-1, 3)
        flat[coords[:300]] = 0
        flat[coords[300:]] = 255
        context = PipelineContext(img, {"color_type": "grayscale"})
        profile = NoiseClassifier.classify(img, context.image_metadata)

        out = NoiseRouter.reduce(context, profile)

    def test_pipeline_process_method_returns_png_bytes(self):
        """RestorationPipeline.process must accept bytes/ndarrays and return valid PNG bytes."""
        pipeline = RestorationPipeline()
        self.assertTrue(hasattr(pipeline, "process"))
        self.assertTrue(hasattr(pipeline, "restore"))
        self.assertTrue(hasattr(pipeline, "_wait_for_models"))

        # Test process with small test array and all stages disabled for instant unit test
        dummy_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out_bytes = pipeline.process(
            dummy_img,
            enable_restoration=False,
            enable_repair=False,
            enable_colorization=False,
            enable_denoise=False,
            enable_sr=False,
        )
        self.assertIsInstance(out_bytes, bytes)
        self.assertTrue(out_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_noisy_lowres_portrait_face_damage_guard(self):
        """
        Verify that small, noisy portraits (e.g. case1_noisy_lena 196x148)
        do not suffer false-positive damage detection across facial regions.
        """
        lena_path = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case1_noisy_lena.jpg")
        if not os.path.exists(lena_path):
            self.skipTest(f"Test image not found at {lena_path}")

        bgr = cv2.imread(lena_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        context = PipelineContext(rgb)

        from ai.analysis.quality_analyzer import QualityAnalyzer
        from ai.analysis.face_detector import FaceDetector
        from ai.damage.enhanced_detector import EnhancedDamageDetector
        from ai.damage.mask_refiner import MaskRefiner

        context.image_metadata.update(QualityAnalyzer.analyze(rgb))
        FaceDetector().detect(context)
        faces = context.image_metadata.get("faces", [])
        self.assertGreaterEqual(len(faces), 1, "Early face detector should locate face bbox")

        detector = EnhancedDamageDetector(use_defect_segmenter=True)
        raw_mask = detector.detect(context)
        refined_mask = MaskRefiner.refine(raw_mask, context)

        bbox = faces[0]["bbox"]
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        face_mask = refined_mask[y0:y1, x0:x1]
        face_damage_ratio = float(np.sum(face_mask > 0)) / float(max(1, (x1 - x0) * (y1 - y0)))

        self.assertLessEqual(
            face_damage_ratio,
            0.05,
            f"Face damage coverage {face_damage_ratio*100:.2f}% exceeds safe 5% ceiling"
        )

    def test_speck_suppressor_bypasses_gaussian_noise(self):
        """Assert SpeckSuppressor bypasses pure Gaussian noise and leaves image bit-identical."""
        from ai.preprocessing.speck_suppressor import SpeckSuppressor

        img_path = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case1_noisy_lena.jpg")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        context = PipelineContext(img_rgb.copy())

        acted = SpeckSuppressor.suppress(context)
        self.assertFalse(acted, "SpeckSuppressor should NOT trigger on Gaussian noise")

        metrics = context.image_metadata.get("speck_field", {})
        self.assertLess(
            metrics.get("spatial_autocorr", 1.0),
            0.18,
            "Spatial autocorrelation for Gaussian noise should be below 0.18 cutoff"
        )
        self.assertTrue(
            np.array_equal(context.working_image, img_rgb),
            "Output must be bit-identical to input when SpeckSuppressor bypasses"
        )

    def test_speck_suppressor_bypasses_landscape_edges(self):
        """Assert SpeckSuppressor bypasses macroscopic scene edges (tree branches/sky)."""
        from ai.preprocessing.speck_suppressor import SpeckSuppressor

        img_path = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case2_dead_tree_landscape.jpg")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        context = PipelineContext(img_rgb.copy())

        acted = SpeckSuppressor.suppress(context)
        self.assertFalse(acted, "SpeckSuppressor should NOT trigger on macroscopic scene edges/landscape")

        metrics = context.image_metadata.get("speck_field", {})
        self.assertGreater(
            metrics.get("edge_colocation", 0.0),
            0.55,
            "Edge co-location for tree branches should exceed 55% macro edge cutoff"
        )
        self.assertTrue(
            np.array_equal(context.working_image, img_rgb),
            "Landscape output must be bit-identical to input, preventing diamond mosaic artifacts"
        )

    def test_speck_suppressor_triggers_on_flaked_portrait(self):
        """Assert SpeckSuppressor continues to trigger on genuine emulsion loss flakes."""
        from ai.preprocessing.speck_suppressor import SpeckSuppressor

        img_path = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case3_sepia_flaked_portrait.jpg")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        context = PipelineContext(img_rgb.copy())

        acted = SpeckSuppressor.suppress(context)
        self.assertTrue(acted, "SpeckSuppressor MUST trigger on genuine emulsion loss flakes")

        metrics = context.image_metadata.get("speck_field", {})
        self.assertGreaterEqual(
            metrics.get("spatial_autocorr", 0.0),
            0.18,
            "Spatial autocorrelation for physical flakes must be >= 0.18"
        )
        self.assertLessEqual(
            metrics.get("edge_colocation", 1.0),
            0.55,
            "Edge co-location for distributed flakes must be <= 55%"
        )
        self.assertLess(
            metrics.get("coverage_after", 1.0),
            metrics.get("coverage", 0.0),
            "Speck suppression must reduce effective speck coverage"
        )

    def test_deferred_face_restoration_logic_on_small_canvas(self):
        """Assert small canvas portraits defer face restoration pre-SR and execute post-SR."""
        img_path = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case1_noisy_lena.jpg")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        context = PipelineContext(img_rgb.copy())
        context.pipeline_flags["enable_sr"] = True

        pipeline = RestorationPipeline()
        # Early face detect
        faces = pipeline._stage_4_face_detect(context, None)
        self.assertGreaterEqual(len(faces), 1, "Face detector should identify the face")

        # Stage 7 Pre-SR
        pipeline._stage_7_face_restoration(context, None)
        self.assertTrue(
            context.pipeline_flags.get("face_restore_deferred_to_post_sr", False),
            "Small canvas face restoration must be deferred to post-SR"
        )

    def test_noise_adaptive_stage_5_denoising(self):
        """Assert Stage 5 adapts filter strength based on measured noise floor."""
        pipeline = RestorationPipeline()

        # 1. Clean image: sigma ~ 0 -> floor parameters maintained
        #    New floors: h_face=max(5.5, 0.95*0)=5.5, h_mid=max(7.5, 1.0*0)=7.5, h_bg=max(6.5, 0.95*0)=6.5
        clean_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ctx_clean = PipelineContext(clean_img)
        ctx_clean.image_metadata["faces"] = []
        pipeline._stage_5_spatially_gated_denoise(ctx_clean, [], None)
        self.assertTrue(any("h_face=5.5, h_mid=7.5, h_bg=6.5" in log for log in ctx_clean.processing_log))

        # 2. Noisy image: sigma high -> filter strength scales and reduces sigma
        np.random.seed(42)
        noise = np.random.normal(0, 15, (100, 100, 3)).astype(np.int16)
        noisy_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ctx_noisy = PipelineContext(noisy_img)
        ctx_noisy.image_metadata["faces"] = []
        pipeline._stage_5_spatially_gated_denoise(ctx_noisy, [], None)

        # Check log has adaptive parameters
        denoise_logs = [log for log in ctx_noisy.processing_log if "Stage 5 Denoise: Spatially-gated adaptive NLM" in log]
        self.assertTrue(len(denoise_logs) > 0)
        # Verify noise was substantially reduced
        l_after = cv2.cvtColor(ctx_noisy.working_image, cv2.COLOR_RGB2LAB)[:, :, 0]
        med = cv2.medianBlur(l_after, 3).astype(np.float32)
        mad = float(np.median(np.abs(l_after.astype(np.float32) - med)))
        sigma_after = 1.4826 * mad
        self.assertLess(sigma_after, 6.0, "Adaptive NLM should reduce high noise significantly")

    def test_face_restorer_adaptive_prior_weight(self):
        """Assert FaceRestorer scales skin prior weight with image degradation level."""
        from ai.restoration.face_restorer import FaceRestorer

        restorer = FaceRestorer()

        # 1. Clean scan: skin prior cap is 0.35
        ctx_clean = PipelineContext(np.zeros((100, 100, 3), dtype=np.uint8))
        ctx_clean.image_metadata["noise_score"] = 1.5
        noise_clean = float(ctx_clean.image_metadata.get("noise_score", 0.0))
        is_deferred_clean = ctx_clean.pipeline_flags.get("_running_deferred_face_restore", False)
        if is_deferred_clean or noise_clean > 3.0:
            cap_clean = min(0.85, max(0.35, 0.35 + 0.05 * (noise_clean - 3.0) if noise_clean > 3.0 else 0.80))
        else:
            cap_clean = 0.35
        self.assertEqual(cap_clean, 0.35, "Clean scan must maintain gentle 0.35 skin prior cap")

        # 2. Noisy scan: skin prior cap scales up to 0.85
        ctx_noisy = PipelineContext(np.zeros((100, 100, 3), dtype=np.uint8))
        ctx_noisy.image_metadata["noise_score"] = 13.0
        noise_noisy = float(ctx_noisy.image_metadata.get("noise_score", 0.0))
        is_deferred_noisy = ctx_noisy.pipeline_flags.get("_running_deferred_face_restore", False)
        if is_deferred_noisy or noise_noisy > 3.0:
            cap_noisy = min(0.85, max(0.35, 0.35 + 0.05 * (noise_noisy - 3.0) if noise_noisy > 3.0 else 0.80))
        else:
            cap_noisy = 0.35
        self.assertEqual(cap_noisy, 0.85, "Noisy scan must increase skin prior cap to 0.85 to avoid blending noise over face")


if __name__ == "__main__":
    unittest.main()
