"""
face_restorer.py — GFPGAN face restoration with identity-preserving blend.

Key Improvements (v4):
- 0% high-frequency grain pass-back (stops raw damage noise from being reinjected).
- Vertical offset Delta_y = 0.00 (centers on true anatomical eye/mouth position).
- Focused elliptical scale rx = 0.55 * W_box, ry = 0.65 * H_box.
- Anatomical cosine alpha core boundary at d <= 0.70, fading to 0 at d = 1.05.
- GFPGAN fidelity weight = 0.45 for balanced facial prior reconstruction.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import torch
import ai.compat  # noqa: F401 - compatibility shim for basicsr / gfpgan

from gfpgan import GFPGANer
from ai.models.base_model import BaseRestorer
from ai.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


class FaceRestorer(BaseRestorer):
    """GFPGAN face restoration with high identity fidelity and natural human expression."""

    def __init__(self) -> None:
        self.gfpganer: GFPGANer | None = None
        self.fidelity_weight: float = 0.85

    @property
    def model_name(self) -> str:
        return "FaceRestorer (GFPGAN)"

    def load(self, device: str) -> None:
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "weights", "GFPGANv1.4.pth",
        )
        if not os.path.exists(weights_path):
            logger.warning("GFPGAN weights not found at %s", weights_path)
            return

        weights_dir = os.path.dirname(weights_path)
        facelib_dir = os.path.join(weights_dir, "facelib")

        try:
            import facexlib.detection as _fd
            import facexlib.parsing as _fp
            import facexlib.utils.face_restoration_helper as _frh
            _orig_det = _fd.init_detection_model
            _orig_par = _fp.init_parsing_model

            def _patched_det(model_name, **kw):
                kw["model_rootpath"] = facelib_dir
                return _orig_det(model_name, **kw)

            def _patched_par(model_name, **kw):
                kw["model_rootpath"] = facelib_dir
                return _orig_par(model_name, **kw)

            _fd.init_detection_model = _patched_det
            _fp.init_parsing_model = _patched_par
            _frh.init_detection_model = _patched_det
            _frh.init_parsing_model = _patched_par
        except Exception as patch_err:
            logger.warning("facexlib patch failed (may re-download): %s", patch_err)

        try:
            self.gfpganer = GFPGANer(
                model_path=weights_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=torch.device(device),
            )
            logger.info("GFPGAN loaded on %s.", device)
        except Exception as exc:
            logger.error("Failed to load GFPGAN: %s", exc)

    def unload(self) -> None:
        self.gfpganer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_single_face(
        self,
        bgr_img: np.ndarray,
        damage_mask: Optional[np.ndarray] = None,
        mode: str = "damage_aware",
        context: Optional[PipelineContext] = None,
    ) -> np.ndarray:
        """
        Run GFPGAN model inference with damage-aware facial blending.
        """
        if self.gfpganer is None:
            return bgr_img

        try:
            # Step 1: Run GFPGAN inference with paste_back=False to obtain aligned 512x512 crops
            self.gfpganer.enhance(
                bgr_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=False,
                weight=self.fidelity_weight,
            )

            helper = self.gfpganer.face_helper
            num_detected = len(helper.cropped_faces)
            if num_detected == 0:
                logger.warning("GFPGAN: no aligned faces detected by face_helper.")
                if context:
                    context.add_log("[WARN] GFPGAN: face restoration skipped — no aligned faces detected by face_helper.")
                    warnings_list = context.image_metadata.setdefault("warnings", [])
                    if "face_restoration_skipped_no_faces_detected" not in warnings_list:
                        warnings_list.append("face_restoration_skipped_no_faces_detected")
                return bgr_img

            debug_enabled = (
                context is not None and
                os.getenv("PIXREVIVE_DEBUG", "").strip() in ("1", "true", "True")
            )
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug_output")
            if debug_enabled:
                os.makedirs(debug_dir, exist_ok=True)

            has_canvas_damage = (
                damage_mask is not None and
                damage_mask.shape[:2] == bgr_img.shape[:2] and
                np.any(damage_mask > 0)
            )

            # W1 paste-guard collection (Phase V0.2): per-face crop alphas are
            # rebuilt inside the loop below (they are composited per-face), so
            # we snapshot each face's final alpha + affine for canvas-level
            # verification BEFORE paste_faces_to_input_image executes.
            guard_alphas: List[np.ndarray] = []
            guard_affines: List[Any] = []

            # Step 2-4: Process each detected face crop
            for i in range(num_detected):
                orig_crop = helper.cropped_faces[i]      # (512, 512, 3) BGR uint8
                restored_crop = helper.restored_faces[i]  # (512, 512, 3) BGR uint8
                affine_mat = helper.affine_matrices[i]   # (2, 3) float affine transform
                crop_h, crop_w = orig_crop.shape[:2]
                face_size = (crop_w, crop_h)

                # Iris defect / emulsion flake pre-cleaning:
                # If bright emulsion flakes or damage fragments fell directly inside an iris,
                # GFPGAN misinterprets the bright spot as sclera or corneal opacity and generates
                # a distorted/hallucinated pupil. Inpainting the flake on the aligned crop using
                # surrounding iris texture and re-enhancing ensures natural, symmetrical gaze.
                if hasattr(helper, "all_landmarks_5") and len(helper.all_landmarks_5) > i:
                    raw_lm = helper.all_landmarks_5[i]
                    if raw_lm is not None and len(raw_lm) >= 2:
                        pts = np.array(raw_lm, dtype=np.float32)
                        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
                        al_pts = (affine_mat @ pts_homo.T).T
                        iris_cleaned = False
                        for e_idx in (0, 1):
                            ex, ey = int(round(al_pts[e_idx][0])), int(round(al_pts[e_idx][1]))
                            r = int(crop_w * 0.048)  # ~24 on 512x512
                            y0_i, y1_i = max(0, ey - r), min(crop_h, ey + r)
                            x0_i, x1_i = max(0, ex - r), min(crop_w, ex + r)
                            iris = orig_crop[y0_i:y1_i, x0_i:x1_i]
                            gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
                            flake = (gray > 140).astype(np.uint8) * 255
                            if np.sum(flake > 0) > 15:
                                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                flake_dil = cv2.dilate(flake, k)
                                orig_crop[y0_i:y1_i, x0_i:x1_i] = cv2.inpaint(iris, flake_dil, 3, cv2.INPAINT_TELEA)
                                iris_cleaned = True
                        if iris_cleaned:
                            try:
                                from torchvision.transforms.functional import normalize
                                from basicsr.utils import img2tensor, tensor2img
                                import torch
                                cropped_face_t = img2tensor(orig_crop / 255., bgr2rgb=True, float32=True)
                                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.gfpganer.device)
                                with torch.no_grad():
                                    output = self.gfpganer.gfpgan(cropped_face_t, return_rgb=False, weight=self.fidelity_weight)[0]
                                    restored_crop = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype(np.uint8)
                                helper.restored_faces[i] = restored_crop
                                if context:
                                    context.add_log(f"GFPGAN Face {i+1}: Repaired iris emulsion flake and re-enhanced eyes.")
                            except Exception as re_err:
                                logger.warning("GFPGAN iris re-enhance failed: %s", re_err)

                # N1 (re-audit 2026-09-30): this used to be `== "damage_aware"`
                # with no else — under mode=="full" the alpha/border-fade block
                # below raised NameError (mode unbound), silently swallowed by
                # the stage catch-all, so full mode skipped GFPGAN entirely.
                # Default keeps damage-aware compositing; full mode is the
                # documented opt-out (no paste at all).
                if mode != "full":
                    if has_canvas_damage:
                        # Warp the full-canvas damage mask into the exact 512x512 aligned face coordinate space
                        face_damage_mask = cv2.warpAffine(
                            damage_mask,
                            affine_mat,
                            face_size,
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        damage_pixels = int(np.sum(face_damage_mask > 0))
                    else:
                        face_damage_mask = None
                        damage_pixels = 0

                    # Determine clean skin prior weight:
                    # On clean/moderately degraded scans, provide robust prior weight (0.45 floor)
                    # so authentic facial contours, jawline structure, and crisp skin texture are preserved
                    # without being washed out by soft upstream denoising.
                    noise_score = float(context.image_metadata.get("noise_score", 0.0)) if context else 0.0
                    is_deferred = context.pipeline_flags.get("_running_deferred_face_restore", False) if context else False
                    skin_prior_cap = 0.45
                    if is_deferred or noise_score > 3.0:
                        skin_prior_cap = min(0.85, max(0.45, 0.45 + 0.05 * (noise_score - 3.0) if noise_score > 3.0 else 0.80))

                    # B3 (docs/REMEDIATION_PLAN_AB_G.md): GFPGAN skin is smooth;
                    # at the elevated caps the waxy composite sits next to crisp
                    # eyes — the classic "pasted-on" look. Measure how much
                    # micro-texture the restoration kept on non-feature skin
                    # and cap the skin alpha accordingly. Feature zones keep
                    # their floor; only flat-skin alpha is limited.
                    try:
                        _g_o = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        _g_r = cv2.cvtColor(restored_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        _skin = feature_zone < 0.05
                        if np.any(_skin):
                            _mo = float(np.var(cv2.Laplacian(_g_o, cv2.CV_32F)[_skin]))
                            _mr = float(np.var(cv2.Laplacian(_g_r, cv2.CV_32F)[_skin]))
                            if _mo > 40.0 and _mr < 0.5 * _mo:
                                skin_prior_cap = min(skin_prior_cap, 0.55)
                                if context:
                                    context.add_log(
                                        f"GFPGAN Face {i+1}: skin micro-texture retention "
                                        f"{_mr / max(_mo, 1e-6):.2f}x < 0.5 — skin alpha capped at 0.55 "
                                        "(anti-wax)."
                                    )
                    except Exception:
                        pass
                    # Anatomical feature zone (eyes, eyebrows, nose, mouth) crispness restoration:
                    # Guarantees that essential facial borders and contours (eyelids, irises, lashes, brows,
                    # nose bridge/nostrils, lip contours) receive crisp high-resolution definition (>= 0.72 floor).
                    feature_zone = np.zeros((crop_h, crop_w), dtype=np.float32)
                    # Eyes
                    cv2.circle(feature_zone, (int(crop_w * 0.375), int(crop_h * 0.46)), int(crop_w * 0.12), 1.0, -1)
                    cv2.circle(feature_zone, (int(crop_w * 0.625), int(crop_h * 0.46)), int(crop_w * 0.12), 1.0, -1)
                    # Eyebrows
                    cv2.ellipse(feature_zone, (int(crop_w * 0.375), int(crop_h * 0.37)), (int(crop_w * 0.13), int(crop_h * 0.06)), 0, 0, 360, 1.0, -1)
                    cv2.ellipse(feature_zone, (int(crop_w * 0.625), int(crop_h * 0.37)), (int(crop_w * 0.13), int(crop_h * 0.06)), 0, 0, 360, 1.0, -1)
                    # Nose
                    cv2.ellipse(feature_zone, (int(crop_w * 0.50), int(crop_h * 0.59)), (int(crop_w * 0.10), int(crop_h * 0.11)), 0, 0, 360, 1.0, -1)
                    # Mouth / lips
                    cv2.ellipse(feature_zone, (int(crop_w * 0.50), int(crop_h * 0.77)), (int(crop_w * 0.18), int(crop_h * 0.10)), 0, 0, 360, 1.0, -1)
                    feature_zone = cv2.GaussianBlur(feature_zone, (15, 15), 0)

                    gray_orig = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
                    lap_orig = cv2.Laplacian(gray_orig, cv2.CV_32F)
                    feat_mask = feature_zone > 0.3
                    sharpness_orig = float(np.var(lap_orig[feat_mask])) if np.any(feat_mask) else 100.0
                    blur_deficit = float(np.clip((280.0 - sharpness_orig) / 200.0, 0.0, 1.0))
                    feature_alpha_floor = max(0.72, skin_prior_cap + (0.85 - skin_prior_cap) * blur_deficit)

                    # Preserve authentic chin-to-neck/shoulder shadow boundary:
                    # In aligned 512x512 crops, the lower chin boundary sits at ~0.84*crop_h.
                    # Fade alpha out below 0.84*crop_h so GFPGAN never overwrites the authentic
                    # submental cast shadow separating chin from neck/shoulder with flat synthesized skin.
                    chin_y_limit = int(crop_h * 0.84)
                    chin_fade_zone = int(crop_h * 0.08)
                    chin_fade = np.ones((crop_h, crop_w), dtype=np.float32)
                    chin_fade[chin_y_limit:chin_y_limit + chin_fade_zone, :] = np.linspace(1.0, 0.0, chin_fade_zone)[:, None]
                    chin_fade[chin_y_limit + chin_fade_zone:, :] = 0.0

                    if damage_pixels > 0:
                        # Step 3: Compute anatomical face core alpha map
                        Y_g, X_g = np.ogrid[:crop_h, :crop_w]
                        core_dist = np.sqrt(((X_g - crop_w / 2.0) / (crop_w * 0.42)) ** 2 + ((Y_g - crop_h * 0.50) / (crop_h * 0.46)) ** 2)
                        
                        inner_face_alpha = np.clip((1.05 - core_dist) / 0.35, 0.0, skin_prior_cap)
                        inner_face_alpha = np.maximum(inner_face_alpha, feature_zone * feature_alpha_floor) * chin_fade
                        
                        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                        dilated_mask = cv2.dilate(face_damage_mask, k_dil, iterations=1)
                        feathered_damage = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (15, 15), 0)
                        
                        # Blend weight: 1.0 in defect zones, feature_alpha_floor in eyes/mouth/nose, skin_prior_cap on clean skin
                        alpha_2d = np.clip(np.maximum(inner_face_alpha, feathered_damage), 0.0, 1.0)
                        alpha_mask = cv2.GaussianBlur(alpha_2d, (15, 15), 0)[:, :, None]
                        
                        if context:
                            context.add_log(
                                f"GFPGAN Face {i+1}: High-fidelity damage-aware face reconstruction ({damage_pixels} defect px, "
                                f"clean skin cap {skin_prior_cap:.2f}, feature floor {feature_alpha_floor:.2f}, fidelity={self.fidelity_weight})."
                            )
                    else:
                        # Face region has no detected canvas damage: apply anatomical detail enhancement
                        Y_g, X_g = np.ogrid[:crop_h, :crop_w]
                        core_dist = np.sqrt(((X_g - crop_w / 2.0) / (crop_w * 0.42)) ** 2 + ((Y_g - crop_h * 0.50) / (crop_h * 0.46)) ** 2)
                        inner_face_alpha = np.clip((0.98 - core_dist) / 0.32, 0.0, skin_prior_cap)
                        inner_face_alpha = np.maximum(inner_face_alpha, feature_zone * feature_alpha_floor) * chin_fade
                        alpha_mask = cv2.GaussianBlur(inner_face_alpha, (15, 15), 0)[:, :, None]
                        if context:
                            context.add_log(
                                f"GFPGAN Face {i+1}: Applied anatomical detail restoration (prior weight max {skin_prior_cap:.2f}, "
                                f"feature floor {feature_alpha_floor:.2f}, fidelity={self.fidelity_weight})."
                            )
                else:
                    # mode == "full": full-weight paste, no anatomical alpha
                    # shaping. (N1: previously this branch was missing and the
                    # shared block below raised NameError on `mode`.)
                    alpha_mask = None
                # Harmonize tonal grading and color temperature between restored crop and body
                try:
                    lab_orig = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
                    lab_rest = cv2.cvtColor(restored_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
                    for ch in range(3):
                        m_orig, s_orig = float(np.mean(lab_orig[:, :, ch])), float(np.std(lab_orig[:, :, ch]))
                        m_rest, s_rest = float(np.mean(lab_rest[:, :, ch])), float(np.std(lab_rest[:, :, ch]))
                        if s_rest > 1e-3:
                            # 60% anchor to original body tone: eliminates color jumps while preserving GFPGAN clarity
                            scaled = (lab_rest[:, :, ch] - m_rest) * (s_orig / s_rest * 0.60 + 0.40) + (m_orig * 0.60 + m_rest * 0.40)
                            lab_rest[:, :, ch] = np.clip(scaled, 0, 255)
                    restored_crop = cv2.cvtColor(lab_rest.astype(np.uint8), cv2.COLOR_LAB2BGR)
                except Exception as _t_err:
                    logger.debug("Tonal harmonization exception: %s", _t_err)

                # Guarantee alpha_mask reaches strict 0.0 at the outer crop margins to eliminate any seam ring
                margin_crop = max(16, int(min(crop_h, crop_w) * 0.06))
                border_fade = np.ones((crop_h, crop_w), dtype=np.float32)
                border_fade[:margin_crop, :] = np.linspace(0.0, 1.0, margin_crop)[:, None]
                border_fade[-margin_crop:, :] = np.linspace(1.0, 0.0, margin_crop)[:, None]
                border_fade[:, :margin_crop] *= np.linspace(0.0, 1.0, margin_crop)[None, :]
                border_fade[:, -margin_crop:] *= np.linspace(1.0, 0.0, margin_crop)[None, :]
                alpha_mask = (
                    alpha_mask * border_fade[:, :, None]
                    if alpha_mask is not None else None
                )

                if alpha_mask is not None:
                    # Step 4: Alpha-composite in float32 space
                    blended_crop = (
                        restored_crop.astype(np.float32) * alpha_mask +
                        orig_crop.astype(np.float32) * (1.0 - alpha_mask)
                    )
                else:
                    # mode == "full": the restored crop IS the composite.
                    blended_crop = restored_crop.astype(np.float32)

                # Edge crispness enhancement on facial feature borders (LAB luminance channel)
                try:
                    lab_crop = cv2.cvtColor(np.clip(blended_crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                    L_crop = lab_crop[:, :, 0]
                    blur_L = cv2.GaussianBlur(L_crop, (0, 0), sigmaX=1.3)
                    gx_c = cv2.Sobel(L_crop, cv2.CV_32F, 1, 0, ksize=3)
                    gy_c = cv2.Sobel(L_crop, cv2.CV_32F, 0, 1, ksize=3)
                    edge_mag_c = np.sqrt(gx_c ** 2 + gy_c ** 2)
                    edge_gate_c = np.clip(edge_mag_c / 40.0, 0.0, 1.0)
                    L_sharp = np.clip(L_crop + (0.18 + 0.28 * edge_gate_c) * (L_crop - blur_L), 0, 255)
                    lab_crop[:, :, 0] = L_sharp
                    blended_uint8 = cv2.cvtColor(lab_crop.astype(np.uint8), cv2.COLOR_LAB2BGR)
                except Exception:
                    blended_uint8 = np.clip(blended_crop, 0, 255).astype(np.uint8)

                # Save debug visualization if debug mode is active
                if debug_enabled:
                    try:
                        cv2.imwrite(os.path.join(debug_dir, f"07_face_{i+1}_orig.png"), orig_crop)
                        cv2.imwrite(os.path.join(debug_dir, f"07_face_{i+1}_gfpgan_raw.png"), restored_crop)
                        alpha_dump = (
                            (alpha_mask * 255).astype(np.uint8)
                            if alpha_mask is not None
                            else np.zeros((crop_h, crop_w), np.uint8)
                        )
                        cv2.imwrite(os.path.join(debug_dir, f"07_face_{i+1}_alpha_mask.png"), alpha_dump)
                        cv2.imwrite(os.path.join(debug_dir, f"07_face_{i+1}_blended.png"), blended_uint8)
                    except Exception as d_err:
                        logger.debug("Debug face save error: %s", d_err)

                # Update restored_faces array inside face_helper
                helper.restored_faces[i] = blended_uint8
                guard_alphas.append(alpha_mask)
                guard_affines.append(affine_mat)

            # ── W1 Paste guard: verify every footprint BEFORE pasting ─────────
            # Measured defect (2026-09-29): GFPGAN's internal detector pasted a
            # 255x167 rect whose footprint excluded the pipeline's eye landmark
            # — a square of restored face beside unrestored anatomy. Reconcile
            # GFPGAN's paste geometry with the pipeline's own face list first.
            try:
                from ai.restoration.paste_guard import (
                    FEATHER_BLUR_K_SMALL, MISALIGNED, PHANTOM, SEAM_RING_PX,
                    build_footprints, changed_region, cliff_pixels,
                    evaluate_pastes, feather_band, feather_seam,
                    internal_seam_ratio, normalize_faces_to_canvas,
                    seam_band, seam_ring_ratio,
                )

                h_cv, w_cv = bgr_img.shape[:2]
                footprints = build_footprints(
                    guard_alphas, guard_affines, (h_cv, w_cv)
                )
                canvas_faces = normalize_faces_to_canvas(
                    (context.image_metadata.get("faces") or []) if context else [],
                    (context.image_metadata.get("faces_frame_size") if context else None),
                    (h_cv, w_cv),
                )
                verdicts = evaluate_pastes(
                    footprints,
                    canvas_faces,
                    (h_cv, w_cv),
                )

                suppressed = []
                softened = []
                for v in verdicts:
                    if v.verdict == PHANTOM:
                        helper.restored_faces[v.index] = helper.cropped_faces[v.index]
                        suppressed.append(v.index)
                    elif v.verdict == MISALIGNED:
                        orig_crop_i = helper.cropped_faces[v.index]
                        rest_i = helper.restored_faces[v.index]
                        alpha_soft = (guard_alphas[v.index] * 0.5)
                        blended_soft = (
                            rest_i.astype(np.float32) * alpha_soft[..., None]
                            + orig_crop_i.astype(np.float32) * (1.0 - alpha_soft[..., None])
                        )
                        helper.restored_faces[v.index] = np.clip(blended_soft, 0, 255).astype(np.uint8)
                        softened.append(v.index)

                # N2: context is Optional — library callers pass None. (Today
                # the None path yields no non-OK verdicts, but the guard must
                # not depend on that for its own logging.)
                if context is not None and suppressed:
                    context.add_log(
                        f"[WARN] PasteGuard: suppressed phantom paste(s) {suppressed} — "
                        "footprint matched no pipeline face."
                    )
                if context is not None and softened:
                    context.add_log(
                        f"[WARN] PasteGuard: softened misaligned paste(s) {softened} — "
                        "eye landmarks outside the paste footprint."
                    )
                if context is not None and (suppressed or softened):
                    meta_w = context.image_metadata.setdefault("warnings", [])
                    if "gfpgan_paste_misaligned" not in meta_w:
                        meta_w.append("gfpgan_paste_misaligned")
                    context.image_metadata["paste_guard"] = {
                        "verdicts": [
                            {"index": v.index, "verdict": v.verdict,
                             "overlap_fraction": round(v.overlap_fraction, 3),
                             "eyes_contained": v.eyes_contained}
                            for v in verdicts
                        ],
                        "suppressed": suppressed,
                        "softened": softened,
                    }
                if debug_enabled and context is not None:
                    try:
                        fp_vis = np.zeros((h_cv, w_cv), np.uint8)
                        for fi, fp in enumerate(footprints):
                            fp_vis[fp] = 255
                        cv2.imwrite(os.path.join(debug_dir, "07_paste_footprints.png"), fp_vis)
                    except Exception:
                        pass
            except Exception as guard_exc:
                logger.warning("PasteGuard verification failed (paste kept): %s", guard_exc)
                if context:
                    context.add_log(f"[WARN] PasteGuard verification failed (paste kept): {guard_exc}")

            # Step 5: Compute inverse affine matrices and inverse-warp the damage-aware blended crops back to canvas
            helper.get_inverse_affine(None)
            pasted_bgr = helper.paste_faces_to_input_image(upsample_img=bgr_img)

            # ── W1 advisory seam checks: footprint edge + INTERNAL alpha cliff ─
            # The measured E3 artifact was not the (feathered) footprint edge
            # but an internal step where the feature-zone floor's changed
            # island ends mid-anatomy. Measure both; feather whichever fires.
            try:
                before_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                after_gray = cv2.cvtColor(pasted_bgr, cv2.COLOR_BGR2GRAY)
                seam_ratios = []
                internal_ratios = []
                feathered_any = False
                for fi, fp in enumerate(footprints):
                    r = seam_ring_ratio(after_gray, fp)
                    if r is not None:
                        seam_ratios.append(round(float(r), 2))
                    ir = internal_seam_ratio(before_gray, after_gray, fp)
                    if ir is not None:
                        # Selective cliff feathering: only the band pixels that
                        # sit on an actual hard step (per-pixel sharpness), with
                        # a small kernel and at most 2 passes. Whole-band blur
                        # defocused the mid-face (measured on the eye sheet,
                        # 2026-09-29) because the island's smooth alpha sections
                        # must NOT be touched.
                        passes = 0
                        cur_gray = after_gray
                        while ir > 3.0 and passes < 2:
                            comp = changed_region(before_gray, cur_gray, fp)
                            if comp is None:
                                break
                            band = seam_band(comp.astype(np.uint8), SEAM_RING_PX)
                            cliffs = cliff_pixels(before_gray, cur_gray, band)
                            if cliffs.sum() < 8:
                                break
                            pasted_bgr = feather_band(
                                pasted_bgr, cliffs, blur_k=FEATHER_BLUR_K_SMALL
                            )
                            feathered_any = True
                            passes += 1
                            cur_gray = cv2.cvtColor(pasted_bgr, cv2.COLOR_BGR2GRAY)
                            ir = internal_seam_ratio(before_gray, cur_gray, fp)
                            if ir is None:
                                break
                        internal_ratios.append(round(float(ir if ir is not None else 0.0), 2))
                        if passes:
                            if context:
                                context.add_log(
                                    f"[WARN] PasteGuard: internal alpha-cliff seam on face {fi} "
                                    f"({passes} selective pass(es), final ratio "
                                    f"{internal_ratios[-1]})."
                                )
                if seam_ratios and max(seam_ratios) > 3.0:
                    worst_idx = int(np.argmax(seam_ratios))
                    pasted_bgr = feather_seam(pasted_bgr, footprints[worst_idx])
                    feathered_any = True
                    if context:
                        context.add_log(
                            f"[WARN] PasteGuard: footprint-edge seam ratio {max(seam_ratios):.1f} on face "
                            f"{worst_idx} exceeded 3.0 — ring feathered."
                        )
                if context is not None and (seam_ratios or internal_ratios):
                    pg = context.image_metadata.setdefault("paste_guard", {})
                    if isinstance(pg, dict):
                        pg["seam_ratios"] = seam_ratios
                        pg["internal_seam_ratios"] = internal_ratios
                        pg["feathered"] = feathered_any

                # A1/B3 eye harmony (docs/REMEDIATION_PLAN_AB_G.md): balance
                # the two eyes' sharpness and iris chroma on the COMPOSITED
                # canvas, where the actual user-visible mismatch lives.
                # Flag: PIXREVIVE_EYE_HARMONY=0 reverts.
                if os.environ.get("PIXREVIVE_EYE_HARMONY", "1").strip().lower() not in ("0", "false", "no", "off"):
                    try:
                        from ai.restoration.paste_guard import harmonize_eyes
                        _mono = False
                        if context is not None:
                            _mono = str(context.image_metadata.get("color_type", "color")) in ("grayscale", "sepia")
                        pasted_bgr, eye_verdicts = harmonize_eyes(
                            pasted_bgr,
                            canvas_faces,
                            source_mono=_mono,
                        )
                        if context is not None and any(v.get("adjusted") for v in eye_verdicts):
                            context.image_metadata["eye_harmony"] = eye_verdicts
                            context.add_log(
                                "[WARN] EyeHarmony: eye sharpness/iris chroma "
                                "mismatch harmonized on composited face(s)."
                            )
                    except Exception as eh_exc:
                        logger.debug("EyeHarmony skipped: %s", eh_exc)
            except Exception as seam_exc:
                logger.debug("PasteGuard seam check skipped: %s", seam_exc)

            return pasted_bgr

        except Exception as exc:
            logger.warning("GFPGAN damage-aware face restoration failed: %s", exc)
            return bgr_img

    def restore_faces(
        self,
        img_rgb: np.ndarray,
        faces: List[Dict[str, Any]],
        damage_mask: Optional[np.ndarray] = None,
        mode: str = "damage_aware",
    ) -> np.ndarray:
        """Direct helper to restore faces on an RGB image array."""
        if self.gfpganer is None or not faces:
            return img_rgb

        bgr_in = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        restored_bgr = self._restore_single_face(
            bgr_in, damage_mask=damage_mask, mode=mode
        )

        if np.array_equal(restored_bgr, bgr_in):
            return img_rgb

        return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    def restore(self, context: PipelineContext) -> np.ndarray:
        """Pipeline entry point for GFPGAN face restoration."""
        if self.gfpganer is None:
            context.add_log("GFPGAN not loaded — skipping face restoration.")
            return context.working_image

        faces = context.image_metadata.get("faces", [])
        if not faces:
            context.add_log("No faces detected — skipping face restoration.")
            return context.working_image

        # Determine face restoration mode: "damage_aware" (default) or "full"
        mode = str(context.image_metadata.get("face_restore_mode", "damage_aware")).lower()
        damage_mask = context.damage_masks.get("final")

        context.add_log(
            f"GFPGAN: restoring {len(faces)} face(s) (mode='{mode}', fidelity_weight={self.fidelity_weight})."
        )

        try:
            bgr_in = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2BGR)
            restored_bgr = self._restore_single_face(
                bgr_in,
                damage_mask=damage_mask,
                mode=mode,
                context=context,
            )

            if not np.array_equal(restored_bgr, bgr_in):
                context.working_image = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
                context.add_log(f"GFPGAN: damage-aware face restoration composite applied across {len(faces)} face(s).")

        except Exception as exc:
            context.add_log(f"GFPGAN failed: {exc}")

        return context.working_image
