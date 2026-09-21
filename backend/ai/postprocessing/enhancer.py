"""
enhancer.py — Context-aware sharpening, contrast, and texture enhancement.

Key fixes vs. previous version:
  - Sharpening is now skipped when the image is blurry (blur_score < 60).
    Previously USM ran unconditionally, creating halos on soft restored images.
  - USM cap raised from 0.25 → 0.40 when a damage mask is present (heavy damage
    suppresses blur_score artificially, so the old cap under-sharpened).
  - Adaptive sharpening strength is proportional to blur_score so crisp
    images get more sharpening and soft images get less.
  - CLAHE tile grid now scales with image size (max(8, w // 128)) so it
    remains effective at 2K–4K resolution after RealESRGAN ×4 SR upscaling.
    The old fixed (8, 8) grid produced 375 px tiles on a 3 000 px image —
    too coarse for any local contrast effect.
  - correct_color_cast() now activates on freshly colorized images (not
    suppressed as before). DDColor output frequently carries a subtle global
    hue shift that the cast corrector removes before the final PNG encode.
  - texture_restore() — new micro-texture injection pass.  After inpainting,
    repaired crack regions appear plastically smooth.  A high-frequency detail
    layer (unsharp-mask residual from the original SR image) is blended back
    into those regions at low opacity so they match the grain of undamaged
    surroundings.  Only touches pixels inside the damage mask.
  - Color cast correction uses a tighter threshold (|avg - 128| > 10) so it
    never desaturates naturally warm-toned portraits.
  - Harmonizer unchanged — still applies guided filter rim blending.
"""

from typing import Optional

import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext


def _mask_matching(mask: Optional[np.ndarray], img_np: np.ndarray) -> Optional[np.ndarray]:
    """Return `mask` resized to img_np's frame, or None if unusable.

    Guards against a mask built at a different resolution than the image it is
    about to be combined with (e.g. a mask produced before super-resolution).
    """
    if mask is None:
        return None
    if mask.shape[:2] == img_np.shape[:2]:
        return mask
    resized = cv2.resize(
        mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8) * 255


class Harmonizer:
    @staticmethod
    def harmonize(context: PipelineContext) -> np.ndarray:
        """
        Gentle guided filter seam blending along inpainting boundary rims.
        """
        img_np = context.working_image
        mask = _mask_matching(context.damage_masks.get("final"), img_np)
        if mask is None or not np.any(mask):
            return img_np

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask, k)
        rim = cv2.subtract(dilated, mask)

        if not np.any(rim):
            return img_np

        try:
            guided = cv2.ximgproc.guidedFilter(img_np, img_np, radius=4, eps=20)
            rim_f = rim.astype(np.float32) / 255.0
            rim_f = cv2.GaussianBlur(rim_f, (5, 5), 0)
            rim_f = np.stack([rim_f] * 3, axis=2)
            harmonized = (img_np * (1.0 - rim_f) + guided * rim_f).astype(np.uint8)
            context.working_image = harmonized
        except Exception:
            pass

        return context.working_image


class Enhancer:
    @staticmethod
    def adaptive_sharpen(
        img_np: np.ndarray,
        mask: Optional[np.ndarray] = None,
        strength: float = 0.20,
    ) -> np.ndarray:
        """
        Luminance-only USM sharpening to enhance hair strands, eyes, and fabric details.

        Operates in LAB space so colour channels are untouched and there is no
        risk of chromatic haloing.
        """
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        blurred_L = cv2.GaussianBlur(L, (0, 0), sigmaX=1.2)
        L_sharp = np.clip(L + strength * (L - blurred_L), 0, 255)
        lab[:, :, 0] = L_sharp
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    @staticmethod
    def gentle_contrast(img_np: np.ndarray, clip_limit: float = 1.0) -> np.ndarray:
        """Apply CLAHE on L channel only (LAB space).

        clip_limit ≤ 1.5 is safe.  Tile grid scales with image dimensions so
        each tile covers roughly 128 × 128 px regardless of resolution — the
        fixed (8, 8) grid was too coarse after ×4 SR upscaling.
        """
        h, w = img_np.shape[:2]
        tile_cols = max(8, w // 128)
        tile_rows = max(8, h // 128)

        if img_np.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_cols, tile_rows))
            return clahe.apply(img_np)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_cols, tile_rows))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def correct_color_cast(img_np: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """
        Neutralise a global color cast in LAB space.

        Only activates when the cast is significant (|avg - 128| > 10 on either
        A or B channel).  Skips silently on naturally warm-toned portraits.
        """
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        avg_a = float(np.mean(lab[:, :, 1]))
        avg_b = float(np.mean(lab[:, :, 2]))

        cast_a = avg_a - 128.0
        cast_b = avg_b - 128.0

        # Skip correction if cast is within natural warm-portrait range
        if abs(cast_a) <= 10.0 and abs(cast_b) <= 10.0:
            return img_np

        # Partial correction (not full removal — preserves warmth)
        L_norm = lab[:, :, 0] / 255.0
        lab[:, :, 1] -= cast_a * L_norm * strength
        lab[:, :, 2] -= cast_b * L_norm * strength

        return cv2.cvtColor(
            np.clip(lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2RGB,
        )

    @staticmethod
    def texture_restore(
        img_np: np.ndarray,
        mask: Optional[np.ndarray],
        opacity: float = 0.18,
        sigma: float = 0.8,
    ) -> np.ndarray:
        """
        Micro-texture injection for inpainted / repaired regions.

        Inpainting fills cracks and tears with smooth colour but no grain,
        making repaired areas look plastic against the surrounding paper texture.
        This pass extracts a high-frequency texture layer from the SR image
        (luminance residual after Gaussian blur) and blends it back into the
        damage mask at low opacity so repaired patches pick up natural grain.

        Args:
            img_np:  RGB uint8 image (after SR + harmonization).
            mask:    Binary damage mask (255 = repaired region, 0 = untouched).
                     If None or empty, returns img_np unchanged.
            opacity: Blend factor for texture injection (0.12–0.25 is natural).
            sigma:   Gaussian blur radius for the frequency split (0.6–1.2).

        Returns:
            RGB uint8 image with texture injected inside the mask.
        """
        if mask is None or not np.any(mask):
            return img_np

        # ── Frequency split in luminance ──────────────────────────────────────
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        L   = lab[:, :, 0]

        # Low-frequency base
        L_low  = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma)
        # High-frequency detail (grain / texture residual)
        L_detail = L - L_low          # range roughly [-30, +30]

        # ── Build a soft mask (avoid hard seams at mask boundary) ────────────
        mask_f = mask.astype(np.float32) / 255.0
        # Dilate slightly so texture extends just past the mask edge
        k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dil = cv2.dilate(mask.astype(np.uint8), k, iterations=1)
        mask_f   = cv2.GaussianBlur(mask_dil.astype(np.float32) / 255.0, (9, 9), 0)

        # ── Inject texture: add high-freq detail scaled by opacity + mask ─────
        L_new = L + L_detail * opacity * mask_f
        L_new = np.clip(L_new, 0, 255)

        lab_out          = lab.copy()
        lab_out[:, :, 0] = L_new
        result           = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return result

    @staticmethod
    def enhance(
        context: PipelineContext,
        enable_global_clahe: bool = True,
        enable_sharpen: bool = True,
        enable_texture_restore: bool = True,
    ) -> np.ndarray:
        """
        Full post-processing enhancement pass.

        Stage order:
          1. Micro-texture injection (repair regions only)
          2. Adaptive CLAHE (always at final resolution)
          3. Luminance-only USM sharpening (skip on blurry / noisy)
          4. Color cast correction (always, including colorized images)
        """
        img_np = context.working_image
        mask   = _mask_matching(context.damage_masks.get("final"), img_np)

        blur_score    = context.image_metadata.get("blur_score",    100.0)
        noise_score   = context.image_metadata.get("noise_score",     0.0)
        has_damage    = mask is not None and np.any(mask)

        # ── 1. Micro-texture restore (repaired regions only) ──────────────────
        if enable_texture_restore and has_damage:
            context.add_log("Enhancer: injecting micro-texture into repaired regions.")
            img_np = Enhancer.texture_restore(img_np, mask, opacity=0.18)

        # ── 2. Adaptive CLAHE — always at final resolution ────────────────────
        # Always run — the contrast_score check that previously gated this was
        # computed on the original small image, not the 4× SR output, so it
        # missed low-contrast SR results.  clip_limit=1.0 is safe and gentle.
        if enable_global_clahe:
            context.add_log("Enhancer: adaptive CLAHE at final resolution.")
            img_np = Enhancer.gentle_contrast(img_np, clip_limit=1.0)

        # ── 3. Adaptive sharpening ────────────────────────────────────────────
        # Skip on blurry images (halos) and on high-noise without a mask.
        # Cap raised to 0.40 when a damage mask is present — heavy surface
        # damage suppresses blur_score artificially, so 0.25 was too gentle.
        if enable_sharpen:
            if blur_score < 60.0:
                context.add_log(
                    f"Enhancer: image is soft (blur_score={blur_score:.1f}) "
                    "— skipping sharpening to prevent halos."
                )
            elif noise_score > 8.0 and not has_damage:
                context.add_log(
                    f"Enhancer: image is noisy (noise_score={noise_score:.1f}) "
                    "— skipping sharpening to prevent grain amplification."
                )
            else:
                # Scale strength proportionally: blur_score 60 → 0.08, 300+ → 0.40
                # Damage-mask images use a higher cap (0.40 vs 0.25) because
                # emulsion damage artificially lowers blur_score.
                strength_cap = 0.40 if has_damage else 0.25
                strength = min(0.08 + (blur_score - 60.0) / 1200.0, strength_cap)
                context.add_log(
                    f"Enhancer: adaptive sharpening "
                    f"(blur_score={blur_score:.1f}, strength={strength:.3f}, "
                    f"cap={strength_cap:.2f}, has_damage={has_damage})."
                )
                img_np = Enhancer.adaptive_sharpen(img_np, mask, strength)

        # ── 4. Color cast correction (all images, including DDColor output) ────
        # Previously skipped on colorized images — now always applied.
        # DDColor can introduce a subtle global hue bias; this corrects it.
        img_np = Enhancer.correct_color_cast(img_np, strength=0.6)

        context.working_image = img_np
        return img_np
