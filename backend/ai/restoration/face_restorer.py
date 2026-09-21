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
import sys
from typing import Dict, List, Optional, Tuple, Any

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

    @staticmethod
    def compute_face_alpha_mask(
        shape: Tuple[int, int],
        face_boxes: List[List[int]],
        rx_scale: float = 0.52,
        ry_scale: float = 0.62,
        core_thresh: float = 0.45,
        outer_thresh: float = 1.15,
    ) -> np.ndarray:
        """
        Compute continuous anatomical cosine alpha mask for facial blending.
        """
        h, w = shape
        alpha_map = np.zeros((h, w), dtype=np.float32)
        Y, X = np.ogrid[:h, :w]

        for (x0, y0, x1, y1) in face_boxes:
            box_w = max(1, x1 - x0)
            box_h = max(1, y1 - y0)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            rx = max(16.0, box_w * rx_scale)
            ry = max(16.0, box_h * ry_scale)

            dist = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)

            face_alpha = np.zeros((h, w), dtype=np.float32)
            core_mask = dist <= core_thresh
            trans_mask = (dist > core_thresh) & (dist <= outer_thresh)
            face_alpha[core_mask] = 1.0
            trans_span = max(0.01, outer_thresh - core_thresh)
            face_alpha[trans_mask] = 0.5 * (1.0 + np.cos(np.pi * (dist[trans_mask] - core_thresh) / trans_span))

            alpha_map = np.maximum(alpha_map, face_alpha)

        if np.any(alpha_map > 0):
            k_smooth = max(15, int(min(h, w) * 0.035) | 1)
            alpha_map = cv2.GaussianBlur(alpha_map, (k_smooth, k_smooth), 0)

        return np.clip(alpha_map, 0.0, 1.0)

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

                if mode == "damage_aware":
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
                alpha_mask = alpha_mask * border_fade[:, :, None]

                # Step 4: Alpha-composite in float32 space
                blended_crop = (
                    restored_crop.astype(np.float32) * alpha_mask +
                    orig_crop.astype(np.float32) * (1.0 - alpha_mask)
                )

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
                        cv2.imwrite(os.path.join(debug_dir, f"07_face_{i+1}_alpha_mask.png"), (alpha_mask * 255).astype(np.uint8))
                        cv2.imwrite(os.path.join(debug_dir, f"07_face_{i+1}_blended.png"), blended_uint8)
                    except Exception as d_err:
                        logger.debug("Debug face save error: %s", d_err)

                # Update restored_faces array inside face_helper
                helper.restored_faces[i] = blended_uint8

            # Step 5: Compute inverse affine matrices and inverse-warp the damage-aware blended crops back to canvas
            helper.get_inverse_affine(None)
            pasted_bgr = helper.paste_faces_to_input_image(upsample_img=bgr_img)
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
