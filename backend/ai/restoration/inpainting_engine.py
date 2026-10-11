"""
inpainting_engine.py — Multi-pass cascade inpainting for photo restoration.

CRITICAL FIX:
  Pass 4 (_feather_blend) was blending with context.original_image — the RAW
  damaged input. After LaMa fills flakes perfectly, Pass 4 re-introduced
  every single flake. Now Pass 4 blends with the PRE-inpainting working
  image (which has halftone suppressed but still has damage — the blend
  preserves undamaged areas while using inpainted content inside the mask).
"""

import cv2
import numpy as np

from ai.pipeline.pipeline_context import PipelineContext


class InpaintingEngine:
    """
    Multi-pass cascade inpainting engine.
    Pass 1: OpenCV Navier-Stokes (structural fill)
    Pass 2: Multi-scale Gaussian (large blob fill)
    Pass 3: LaMa deep refinement (if available)
    Pass 4: Feathered edge blend (smooth seam transitions)
    """

    def __init__(self, lama_model=None) -> None:
        self.lama_model = lama_model

    def inpaint(self, context: PipelineContext) -> np.ndarray:
        """Run two-tier inpainting. Returns inpainted image.

        Tier 1 (NS + multiscale): large damage blobs — LaMa hallucinates textures
        Tier 2 (LaMa):           small flakes / scratches — LaMa excels here
        """
        mask = context.damage_masks.get("final")
        if mask is None or not np.any(mask):
            context.add_log("InpaintingEngine: no damage mask — skipping.")
            return context.working_image

        # Prefer two-tier masks if available, else fall back to full mask
        lama_mask = context.damage_masks.get("lama_mask", mask)
        ns_mask   = context.damage_masks.get("ns_mask",   None)

        # Save pre-inpainting state for final blend
        pre_inpaint = context.working_image.copy()
        img = context.working_image.copy()

        damage_ratio = float(np.sum(mask > 0) / mask.size)
        context.add_log(
            f"InpaintingEngine: cascade on {damage_ratio*100:.1f}% damage "
            f"(NS={np.sum(ns_mask>0)/mask.size*100:.1f}% "
            f"LaMa={np.sum(lama_mask>0)/mask.size*100:.1f}%)."
            if ns_mask is not None else
            f"InpaintingEngine: cascade on {damage_ratio*100:.1f}% damage."
        )

        # ── Tier 1: NS + multi-scale for large blobs ──────────────────────────
        if ns_mask is not None and np.any(ns_mask):
            img = self._ns_inpaint(img, ns_mask, context)
            ns_ratio = float(np.sum(ns_mask > 0) / ns_mask.size)
            if ns_ratio > 0.03:
                img = self._multiscale_fill(img, ns_mask, context)

        # ── Tier 2: LaMa for small components ────────────────────────────────
        if np.any(lama_mask):
            if self.lama_model is not None and getattr(self.lama_model, "model", None) is not None:
                img = self._lama_refine(img, lama_mask, context)
            else:
                # LaMa unavailable — use NS for small components too
                img = self._ns_inpaint(img, lama_mask, context)

        # ── Final: Feathered edge blend ───────────────────────────────────────
        img = self._feather_blend(pre_inpaint, img, mask)

        context.working_image = img
        context.add_log("InpaintingEngine: cascade complete.")
        return img


    # ─── Pass 1: Navier-Stokes ────────────────────────────────────────────────

    @staticmethod
    def _ns_inpaint(img, mask, context, radius: int = 8):
        """Navier-Stokes inpainting with adaptive radius.

        The radius controls how far OpenCV looks for fill pixels. Thin 1-3px
        scratches need r=8, but a wide 15px diagonal crack needs r=18-20 to
        properly bridge across. We pick the minimum needed for the mask.
        """
        try:
            # Estimate max component width to choose an appropriate radius
            num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_lbl > 1:
                widths = [min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                          for i in range(1, num_lbl)]
                max_crack_width = int(max(widths)) if widths else 0
                # Use ceil(crack_width * 1.5), clamped to [8, 20]
                radius = max(8, min(20, int(max_crack_width * 1.5)))

            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_dilated = cv2.dilate(mask, k3, iterations=1)
            inpainted = cv2.inpaint(bgr, mask_dilated, radius, cv2.INPAINT_NS)
            result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
            context.add_log(f"InpaintingEngine [Pass 1] NS: done (r={radius}).")
            return result
        except Exception as exc:
            context.add_log(f"InpaintingEngine [Pass 1] NS failed: {exc}")
            return img

    # ─── Pass 2: Multi-scale Gaussian fill ────────────────────────────────────

    @staticmethod
    def _multiscale_fill(img, mask, context):
        try:
            h, w = img.shape[:2]
            current = img.copy()

            for scale in [4, 2]:
                sw, sh = max(1, w // scale), max(1, h // scale)
                small_img  = cv2.resize(current, (sw, sh), interpolation=cv2.INTER_AREA)
                small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

                if not np.any(small_mask > 0):
                    continue

                small_bgr = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR).astype(np.float32)
                invalid = (small_mask > 0).astype(np.float32)
                valid = 1.0 - invalid

                for _ in range(12):
                    blurred = cv2.GaussianBlur(small_bgr * valid[:,:,None], (0,0), sigmaX=6)
                    blurred_w = cv2.GaussianBlur(valid[:, :, None], (0, 0), sigmaX=6)
                    if blurred_w.ndim == 2:
                        blurred_w = blurred_w[:, :, None]
                    filled = np.where(
                        blurred_w > 1e-6,
                        blurred / (blurred_w + 1e-6),
                        small_bgr
                    )
                    small_bgr = np.where(
                        invalid[:,:,None] > 0, filled, small_bgr
                    ).clip(0, 255).astype(np.float32)

                filled_full = cv2.resize(
                    small_bgr.astype(np.uint8), (w, h), interpolation=cv2.INTER_LANCZOS4
                )
                filled_rgb = cv2.cvtColor(filled_full, cv2.COLOR_BGR2RGB)
                mask_3d = (mask[:,:,None] > 0)
                current = np.where(mask_3d, filled_rgb, current).astype(np.uint8)

            context.add_log("InpaintingEngine [Pass 2] multi-scale: done.")
            return current
        except Exception as exc:
            context.add_log(f"InpaintingEngine [Pass 2] failed: {exc}")
            return img

    # ─── Pass 3: LaMa refinement ──────────────────────────────────────────────

    def _lama_refine(self, img, mask, context):
        try:
            import torch
            import torch.nn.functional as F_nn

            assert self.lama_model is not None, "lama_model must be set before calling _lama_refine"
            net    = self.lama_model.model
            device = self.lama_model.device

            k7     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_d = cv2.dilate(mask, k7, iterations=1)

            img_f  = img.astype(np.float32) / 255.0
            mask_f = mask_d.astype(np.float32) / 255.0

            img_t  = torch.from_numpy(img_f).permute(2,0,1).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(device)

            h, w = img_t.shape[2:]
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            if ph or pw:
                img_t  = F_nn.pad(img_t,  (0, pw, 0, ph), mode="reflect")
                mask_t = F_nn.pad(mask_t, (0, pw, 0, ph), mode="reflect")

            with torch.no_grad():
                out_t = net(img_t, mask_t)

            out_np = (
                out_t[:, :, :h, :w].squeeze(0).permute(1,2,0).cpu().numpy().clip(0,1) * 255
            ).astype(np.uint8)

            # Blend LaMa output ONLY inside the mask (feathered)
            feather = cv2.GaussianBlur(mask_d.astype(np.float32)/255.0, (15,15), 0)
            feather_3d = feather[:,:,None]
            blended = (
                out_np.astype(np.float32) * feather_3d
                + img.astype(np.float32) * (1.0 - feather_3d)
            ).astype(np.uint8)

            context.add_log("InpaintingEngine [Pass 3] LaMa: done.")
            return blended
        except Exception as exc:
            context.add_log(f"InpaintingEngine [Pass 3] LaMa failed: {exc}")
            return img

    @staticmethod
    def _feather_blend(pre_inpaint, inpainted, mask, feather_px=5):
        """
        Final blend: inside mask → inpainted pixels, outside → pre_inpaint.
        Applies a smooth continuous Gaussian feathered alpha ramp to eliminate
        hard cut edges and halo seams.
        """
        if not np.any(mask):
            return pre_inpaint

        if inpainted.shape[:2] != pre_inpaint.shape[:2]:
            inpainted = cv2.resize(
                inpainted, (pre_inpaint.shape[1], pre_inpaint.shape[0])
            )

        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask, k_dil, iterations=1)
        k = max(7, feather_px * 2 + 1)
        feather = cv2.GaussianBlur(
            mask_dilated.astype(np.float32) / 255.0, (k, k), 0
        )
        feather = np.clip(feather * 1.15, 0.0, 1.0)[:, :, None]

        blended = (
            inpainted.astype(np.float32) * feather
            + pre_inpaint.astype(np.float32) * (1.0 - feather)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)
