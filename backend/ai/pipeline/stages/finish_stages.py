"""Finishing stages: 3-zone denoise, GFPGAN face restoration, pre-SR
anti-aliasing, super-resolution, sharpening, exhibition grading
(mixins on RestorationPipeline; see stage_registry.py)."""

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.debug_logger import DebugLogger
from ai.pipeline.stages._helpers import _refresh_sharpness
from ai.analysis.face_detector import FaceDetector
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.restoration.super_resolution import SuperResolution

logger = logging.getLogger(__name__)

# Phase H2 "Show what's real": per-task evidence-zone artifacts live next to
# the result files in temp_results (served at /api/process/{id}/evidence and
# swept with them), fulfilling design §7's auditability promise — the
# real/generated claim is provable, not asserted.
EVIDENCE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "temp_results"))
from ai.restoration.deblurrer import ImageDeblurrer, estimate_optical_blur


class FinishStagesMixin:
    if TYPE_CHECKING:  # pragma: no cover — host-object contract only
        from ai.restoration.lama_restorer import LamaRestorer
        from ai.colorization.ddcolor_colorizer import DDColorColorizer
        from ai.restoration.face_restorer import FaceRestorer

    # Host-object contract (pyright, plan 6.3): these attributes live on
    # RestorationPipeline, which composes this mixin.
    _device: str
    _model_load_failures: Dict[str, str]
    face_detector: FaceDetector
    face_restorer: "FaceRestorer"
    super_resolution: SuperResolution
    colorizer: "DDColorColorizer"
    _lama: Optional["LamaRestorer"]

    def _stage_11_output_verify(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 11: closed-loop output verification.

        Re-detects damage residue on the RESTORED image (the repair stages
        trust their own masks), micro-cleans specks outside faces, and runs
        color-coherence / texture-plausibility checks. Findings land in
        metadata["output_verification"] and the Restoration Report flags.
        """
        from ai.validation.output_verifier import verify_and_clean

        if on_progress:
            on_progress("output_verify", 99, "Verifying the restored output...")
        context.start_timer("output_verify")
        try:
            record = verify_and_clean(context)
            context.image_metadata["output_verification"] = record
        except Exception as exc:  # verification must never destroy a result
            from ai.pipeline.debug_logger import DebugLogger

            context.image_metadata["output_verification"] = {"ran": False, "error": str(exc)}
            context.add_log(f"Stage 11 OutputVerify: skipped ({type(exc).__name__}: {exc}).")
        finally:
            context.end_timer("output_verify")

    @staticmethod
    def _save_evidence_overlay(task_key: str, working_image: np.ndarray,
                               keep: np.ndarray) -> str:
        """Write the colorized evidence-zone overlay for a hybrid result.

        Green tint over surviving original pixels, translucent magenta over
        AI-re-imagined zones, on a dimmed copy of the output — one PNG that
        shows exactly where the original evidence ends and the re-imagination
        begins (design doc §7 'Show what's real').
        """
        import cv2

        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        safe = "".join(c for c in task_key if c.isalnum() or c in "-_")[:80]
        if not safe:
            safe = "adhoc"
        out_path = os.path.join(EVIDENCE_DIR, f"{safe}_evidence.png")

        h, w = working_image.shape[:2]
        overlay = (working_image.astype(np.float32) * 0.45)
        green = np.array([46, 204, 113], dtype=np.float32)   # RGB original
        magenta = np.array([214, 69, 176], dtype=np.float32)  # RGB re-imagined
        overlay[keep] = overlay[keep] + green * 0.55
        overlay[~keep] = overlay[~keep] + magenta * 0.55
        cv2.imwrite(out_path, cv2.cvtColor(
            np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return out_path

    def _release_standard_models(self) -> None:
        """Free LaMa/GFPGAN/Real-ESRGAN/DDColor VRAM before Stage 12 re-imagine.

        Stage 12 is the final content stage: nothing re-reads these weights
        this request. The re-imagine engine's resource gate needs the floor
        they were holding (measured: only 0.94 GB free with them resident).
        Idempotent and failure-tolerant — an unload failure must not kill a
        completed restoration.
        """
        for attr in ("_lama", "face_restorer", "super_resolution", "colorizer"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "unload"):
                try:
                    obj.unload()
                except Exception as exc:  # noqa: BLE001 - never kill the run
                    logger.warning("Stage 12: unloading %s failed: %s", attr, exc)
        # RetinaFace (face_detector) and the U-Net defect segmenter are the
        # other big GPU residents (~2.4 GB together). Both self-heal lazily:
        # detect() reloads det_net, EnhancedDamageDetector recreates the
        # segmenter when None.
        fd = getattr(self, "face_detector", None)
        if fd is not None and hasattr(fd, "unload"):
            try:
                fd.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Stage 12: unloading face_detector failed: %s", exc)
        ed = getattr(self, "enhanced_detector", None)
        seg = getattr(ed, "defect_segmenter", None) if ed is not None else None
        if seg is not None and getattr(seg, "is_loaded", False):
            seg.model = None
            seg.is_loaded = False  # EnhancedDamageDetector recreates it lazily
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_standard_models(self) -> None:
        """Reload what _release_standard_models freed (best-effort).

        Reuses the canonical _reload_models_on path (same one device switching
        uses): it reloads LaMa/GFPGAN/Real-ESRGAN/DDColor, rewires the
        inpainting engine, records failures in _model_load_failures for
        /api/health, and re-sets the colorizer-ready event. A failed reload
        degrades the NEXT request, never this one.
        """
        try:
            self._reload_models_on(self._device)
        except Exception as exc:  # noqa: BLE001 - next request recovers lazily
            logger.warning("Stage 12: standard-model reload failed: %s", exc)
        # Detectors reload lazily on next use, but warm them now so the next
        # request starts clean (failure here only degrades the next request).
        fd = getattr(self, "face_detector", None)
        if fd is not None:
            try:
                fd._ensure_det_net()
            except Exception:  # noqa: BLE001
                pass
        ed = getattr(self, "enhanced_detector", None)
        seg = getattr(ed, "defect_segmenter", None) if ed is not None else None
        if seg is not None and not getattr(seg, "is_loaded", False):
            try:
                seg.load()
            except Exception:  # noqa: BLE001
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _stage_12_reimagine(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 12 (Phase H1, opt-in beta): re-imagine unrecoverable regions.

        Fires only when the user opted in (registry route_gate) AND the
        coverage gate agrees the damage is hopeless (>= 35% of frame or
        >= 55% of any face is Destroy-class). The engine composites every
        surviving pixel back unconditionally (bytewise-asserted), verifies
        identity via SFace with bounded rollback, and records the full
        evidence/resource disclosure into metadata for the report.
        """
        if on_progress:
            on_progress("reimagine", 99, "Re-imagining unrecoverable regions (beta)...")
        context.start_timer("reimagine")
        try:
            from ai.restoration.reimaginer import reimagine

            meta = context.image_metadata
            masks = context.damage_masks
            destroy = None
            for key in ("final", "inpaint", "damage"):
                m = masks.get(key)
                if m is not None and np.any(m > 0):
                    destroy = (m > 0).astype(np.uint8) * 255
                    break
            # Coverage gate: hopeless = >= 35% of frame or >= 55% of a face.
            frame_pct = (
                float(np.sum(destroy > 0) / destroy.size * 100)
                if destroy is not None else 0.0
            )
            face_pct = None
            faces = meta.get("faces") or []
            if destroy is not None and faces:
                h, w = destroy.shape[:2]
                for face in faces:
                    bbox = face.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x0, y0, x1, y1 = [int(round(v)) for v in bbox[:4]]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        continue
                    roi = destroy[y0:y1, x0:x1]
                    pct = float(np.sum(roi > 0) / roi.size * 100)
                    face_pct = pct if face_pct is None else max(face_pct, pct)
            hopeless = frame_pct >= 35.0 or (face_pct is not None and face_pct >= 55.0)
            # Two different "did not fire" stories must be told apart
            # (live-audit find 2026-09-27): "no unrecoverable damage was
            # detected at all" (detector saw a clean/repairable image — a
            # 98%-destroyed synthetic probe reported frame 0.0% because its
            # blotches sat outside the trained damage taxonomy) vs "damage
            # was detected but is below the hopeless bar" (LaMa already
            # healed it). Saying "coverage 0.0%" to the first user reads as
            # "your photo is pristine"; name the detection outcome instead.
            if destroy is None:
                reason = (
                    "no unrecoverable damage was detected — the standard "
                    "pipeline already handled everything the detector "
                    "recognized (the beta only fires on hopeless coverage: "
                    "≥35% of frame or ≥55% of a face)"
                )
            else:
                reason = (
                    "coverage below the hopeless bar "
                    f"(frame {frame_pct:.1f}%, worst face "
                    f"{face_pct if face_pct is not None else 'n/a'})"
                )
            record = {
                "ran": True,
                "status": "skipped",
                "reason": reason,
                "coverage": {"frame_pct": round(frame_pct, 2),
                             "worst_face_pct": (round(face_pct, 2)
                                                if face_pct is not None else None)},
            }
            if hopeless:
                # VRAM bridge (integration fix, measured 2026-09-27): inside
                # the live pipeline the standard models (LaMa/GFPGAN/Real-
                # ESRGAN/DDColor) hold enough VRAM that only ~0.9 GB stays
                # free and the reimagine resource gate can never clear its
                # 3.0 GB MINIMAL floor. Stage 12 is the FINAL content stage —
                # nothing needs those weights again this request — so release
                # them before the engine profiles, and reload them in the
                # finally below so later requests are unaffected.
                self._release_standard_models()
                try:
                    result = reimagine(
                        context.working_image,
                        destroy if destroy is not None
                        else np.zeros(context.working_image.shape[:2], np.uint8),
                        task_id=str(meta.get("task_id", "adhoc")),
                    )
                finally:
                    self._restore_standard_models()
                if result.get("status") == "done":
                    pre = context.working_image
                    context.working_image = result.pop("result")
                    keep = (destroy == 0)
                    result["evidence_preserved"] = bool(
                        np.array_equal(context.working_image[keep], pre[keep]))
                    context.sync_geometry_to_working_image()
                    # "Show what's real" (design §7): persist the evidence
                    # zones as a colorized overlay (green = original pixels,
                    # translucent magenta = AI-re-imagined) next to the result
                    # file, served at /api/process/{task_id}/evidence. The
                    # real/generated claim becomes auditable, not asserted.
                    try:
                        task_key = str(meta.get("task_id", "adhoc"))
                        self._save_evidence_overlay(
                            task_key, context.working_image, keep)
                        result["evidence_overlay"] = (
                            f"/api/process/{task_key}/evidence")
                    except Exception as exc:  # noqa: BLE001 - audit art only
                        logger.warning("evidence overlay save failed: %s", exc)
                record.update(result)
                if record.get("status") == "done":
                    # A success record must not carry the earlier skip reason.
                    record.pop("reason", None)
            meta["reimagine"] = record
            context.add_log(
                f"Stage 12 Reimagine: {record.get('status')} — "
                f"{record.get('reason', 'no further detail.')[:160]}"
            )
        except Exception as exc:  # never destroy a completed restoration
            logger.exception("Stage 12 Reimagine failed")
            context.image_metadata["reimagine"] = {
                "ran": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            context.add_log(f"Stage 12 Reimagine failed: {type(exc).__name__}: {exc}.")
        finally:
            context.end_timer("reimagine")
            try:
                from ai.restoration.reimaginer import unload_reimagine_models
                unload_reimagine_models()
            except Exception:  # noqa: BLE001 - unload is best-effort
                pass

    def _stage_4_5_deblur(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 4.5 (T-3): Restormer deblur, strictly gated on optical blur.

        Two-path probe-calibrated gate (see deblurrer.estimate_optical_blur):
        path A = defocus via edge-spread acutance, path B = motion via H/V
        gradient anisotropy. The whole real corpus measures inert. The 100MB
        Restormer loads lazily on the FIRST gate-fire: images that don't need
        it never pay the VRAM.
        """
        DebugLogger.save_stage(context, "05_before_deblur")
        if on_progress:
            on_progress("deblur", 56, "Correcting optical blur...")

        measurement = estimate_optical_blur(context.working_image)
        if not measurement["is_optical"]:
            context.add_log(
                f"Stage 4.5 Deblur: not optical blur "
                f"(spread={measurement.get('edge_spread')}, "
                f"dir={measurement.get('directional_ratio')}) - skipping."
            )
            return

        try:
            if self._deblur is None:
                self._deblur = ImageDeblurrer()
            if not self._deblur.is_loaded():
                context.add_log("Stage 4.5 Deblur: loading Restormer weights (first fire)...")
                self._deblur.load(self._device)
            context.start_timer("deblurring")
            result, info = self._deblur.deblur(context.working_image, device=self._device)
            if info.get("applied"):
                context.working_image = result
                context.add_log(
                    f"Stage 4.5 Deblur: Restormer applied (path={info.get('fire_path')}, "
                    f"spread={info.get('edge_spread')}, dir={info.get('directional_ratio')})."
                )
            else:
                context.add_log(f"Stage 4.5 Deblur skipped: {info.get('reason', info.get('error'))}")
        except FileNotFoundError as exc:
            context.add_log(f"Stage 4.5 Deblur unavailable: {exc}")
            self._model_load_failures["deblur"] = str(exc)
        except Exception as exc:
            context.add_log(f"Stage 4.5 Deblur failed: {exc}")
            self._model_load_failures["deblur"] = str(exc)

    def _stage_5_spatially_gated_denoise(
        self,
        context: PipelineContext,
        faces: Optional[List[Dict[str, Any]]] = None,
        on_progress: Optional[Callable] = None,
    ) -> None:
        """
        Stage 5: Spatially-Gated 3-Zone Denoising.
        - Face zone: Gentle NLM (h ≈ 0.95σ, capped 10.5) preserving skin texture.
        - Subject zone: Balanced NLM (h ≈ 1.0σ).
        - Background (undamaged, high σ): Moderate NLM (h = 0.85σ, capped 11.5) — preserves photographic grain.
        - Background (damaged or low σ): Aggressive NLM (h = 1.4σ) for deep grain/scratch suppression.

        `faces` defaults to the faces already recorded in context metadata, so
        the registry loop can call this stage uniformly (method(context)).
        """
        if faces is None:
            faces = list(context.image_metadata.get("faces") or [])
        DebugLogger.save_stage(context, "04b_before_denoise")

        if on_progress:
            on_progress("denoise", 50, "Applying spatially-gated 3-zone denoising...")
        context.start_timer("denoising")

        try:
            # G1 (docs/REMEDIATION_PLAN_AB_G.md): structure/HF guard switches.
            # Default on; PIXREVIVE_STRUCTURE_GUARD=0 / PIXREVIVE_HF_GUARD=0
            # restore the pre-G1 behavior exactly.
            _struct_guard = os.environ.get("PIXREVIVE_STRUCTURE_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")
            _hf_guard = os.environ.get("PIXREVIVE_HF_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")
            pre_canvas = context.working_image.copy()

            h_img, w_img = context.working_image.shape[:2]
            face_mask_2d = np.zeros((h_img, w_img), dtype=np.float32)

            for f in faces:
                bbox = f.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    cv2.rectangle(face_mask_2d, (x0, y0), (x1, y1), 1.0, -1)
                landmarks = f.get("landmarks")
                if landmarks is not None and len(landmarks) >= 3:
                    pts = np.array(landmarks, dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(face_mask_2d, hull, 1.0)

            if np.any(face_mask_2d > 0):
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                face_mask_2d = cv2.dilate(face_mask_2d, k_dil, iterations=1)
                face_mask_2d = cv2.GaussianBlur(face_mask_2d, (21, 21), 0)

            mask_face = np.clip(face_mask_2d, 0.0, 1.0)

            # Mid-zone mask: dilated subject zone around face
            if np.any(mask_face > 0):
                k_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))
                mid_zone = cv2.dilate((mask_face > 0.1).astype(np.uint8), k_mid)
                mid_zone_f = cv2.GaussianBlur(mid_zone.astype(np.float32), (35, 35), 0)
                mid_mask = np.clip(mid_zone_f * (1.0 - mask_face), 0.0, 1.0)
            else:
                mid_mask = np.full((h_img, w_img), 0.4, dtype=np.float32)

            bg_mask = np.clip(1.0 - mask_face - mid_mask, 0.0, 1.0)

            # Denoise strictly on L* channel in CIELAB space
            lab = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)

            # Measure high-frequency noise floor sigma via MAD on L* channel.
            # On images with large flat margins the global MAD collapses to 0
            # (the margin is the modal residual), so blend in Stage 1's
            # content-only sigma as a floor — measured on the photo, not the
            # scanner bed. Capped so a stale estimate can never over-drive.
            med = cv2.medianBlur(l_chan, 3).astype(np.float32)
            mad = float(np.median(np.abs(l_chan.astype(np.float32) - med)))
            sigma_global = 1.4826 * mad
            sigma_meta = float(context.image_metadata.get("content_sigma", 0.0))
            sigma_noise = max(sigma_global, min(sigma_meta, 20.0)) if sigma_meta > 0 else sigma_global

            # Check if face is small/low-res or deferred
            is_small_face = any(
                min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) < 90
                for f in faces if "bbox" in f and len(f["bbox"]) >= 4
            ) or min(h_img, w_img) < 320 or context.pipeline_flags.get("face_restore_deferred_to_post_sr", False)

            # Check damage mask coverage in background area
            final_mask = context.damage_masks.get("final")
            bg_damaged = False
            if final_mask is not None and np.any(final_mask > 0):
                bg_pix = np.asarray(bg_mask) > 0.5
                if np.any(bg_pix):
                    bg_cov = float(np.sum(final_mask[bg_pix] > 0)) / max(1.0, float(np.sum(bg_pix)))
                    if bg_cov > 0.02:
                        bg_damaged = True

            # Adaptive filter strengths: scale h proportionally to measured noise floor.
            # bg_damaged drives aggressive smoothing; high sigma alone does NOT — we want
            # photographic grain preserved when the background has no damage (e.g. noisy Lena).
            # J-8 (2026-10-12 forensic follow-up): damage-inflated sigma caps.
            # Residual LaMa fill and surviving flakes inflate the MAD sigma
            # estimate (measured: stain + fill drove sigma ~14 on the teared
            # portrait), and h = 1.4*sigma then over-smoothed the whole
            # background into the waxy watercolour look. Caps bind ONLY far
            # above the floors (clean-image log contract h_face=5.5,
            # h_mid=7.5, h_bg=6.5 untouched) so the pure-noise sigma-15
            # reduction contract still holds in the undamaged branches.
            # PIXREVIVE_NLM_CAP=0 reverts to the unbounded legacy scales.
            _nlm_cap = float(os.environ.get("PIXREVIVE_NLM_CAP", "14.0") or "0")
            if bg_damaged:
                h_bg = max(11.0, 1.4 * sigma_noise)
                if _nlm_cap > 0:
                    h_bg = min(h_bg, _nlm_cap)
            elif sigma_noise > 7.0:
                h_bg = max(11.5, 1.25 * sigma_noise)
            else:
                h_bg = max(6.5, 0.95 * sigma_noise)
            # G1 note: an h_bg cap here was tried and removed — it broke the
            # noise-reduction contract on all-noise frames (sigma 15 -> 8.9).
            # Texture/contour preservation is owned by the coherence-gated
            # protect map below and the HF re-injection guard, not by
            # weakening the filter globally. The J-8 cap above targets only
            # the DAMAGE branch, where σ is polluted by damage energy.
            h_mid = max(7.5, 1.10 * sigma_noise)
            if _nlm_cap > 0 and bg_damaged:
                h_mid = min(h_mid, _nlm_cap * 0.85)
            # Face zone: cap at 1.10x sigma regardless of small-face flag to retain skin texture.
            if is_small_face and sigma_noise > 4.0:
                h_face = min(max(7.0, 1.10 * sigma_noise), 12.0)
            else:
                h_face = max(5.5, 0.95 * sigma_noise)
            if _nlm_cap > 0:
                h_face = min(h_face, _nlm_cap)

            # 1. Background: Substantial NLM
            l_bg = cv2.fastNlMeansDenoising(l_chan, None, h=h_bg, templateWindowSize=7, searchWindowSize=21)
            # 2. Mid-zone (subject clothing/hair): Balanced NLM
            l_mid = cv2.fastNlMeansDenoising(l_chan, None, h=h_mid, templateWindowSize=7, searchWindowSize=21)
            # 3. Face: Adaptive NLM preserving anatomy or smoothing noise
            l_face = cv2.fastNlMeansDenoising(l_chan, None, h=h_face, templateWindowSize=5, searchWindowSize=15)

            # Compute luminance edge energy using pre-smoothed L-channel so high-frequency noise spikes
            # do not register as structural edges.
            l_smooth = cv2.GaussianBlur(l_chan, (3, 3), 0)
            gx = cv2.Sobel(l_smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(l_smooth, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = np.sqrt(gx ** 2 + gy ** 2)

            local_mean = cv2.blur(l_smooth.astype(np.float32), (7, 7))
            local_sq = cv2.blur(l_smooth.astype(np.float32) ** 2, (7, 7))
            local_var = np.maximum(0.0, local_sq - local_mean ** 2)

            # Noise-aware edge and texture protection:
            # Detects genuine anatomical boundaries (chin shadow, shoulder, jawline, hat brim).
            edge_thresh = max(35.0, 2.5 * sigma_noise)
            edge_weight = np.clip((edge_mag - 1.5 * sigma_noise) / edge_thresh, 0.0, 1.0)

            noise_floor_var = max(16.0, sigma_noise ** 2)
            raw_texture = np.clip((local_var - 2.0 * noise_floor_var) / (2.5 * noise_floor_var), 0.0, 1.0)
            # G1: the old weight *vanished* as sigma grew (zero at sigma>=15),
            # i.e. texture protection was weakest exactly when NLM h was
            # strongest — the direct cause of waxy skin and melted hair on
            # noisy inputs. Keep a 35% floor in textured zones; clean images
            # (sigma small) take the original value unchanged via max().
            texture_weight = np.maximum(
                raw_texture * max(0.0, 1.0 - sigma_noise / 15.0),
                raw_texture * 0.35,
            ) if _struct_guard else raw_texture * max(0.0, 1.0 - sigma_noise / 15.0)

            protect_map = np.maximum(edge_weight, texture_weight)

            l_fused = (
                l_face.astype(np.float32) * mask_face +
                l_mid.astype(np.float32) * mid_mask +
                l_bg.astype(np.float32) * bg_mask
            )
            # Retain authentic edge sharpness: blend with edge-preserving reference rather than raw noisy pixels.
            # If image has substantial noise, use median-filtered L to avoid re-injecting single-pixel noise spikes.
            if sigma_noise > 4.0:
                l_edge_ref = cv2.medianBlur(l_chan, 3).astype(np.float32)
                max_protect = max(0.20, 0.40 - 0.015 * sigma_noise)
            else:
                l_edge_ref = l_chan.astype(np.float32)
                max_protect = 0.50

            l_fused = l_fused * (1.0 - max_protect * protect_map) + l_edge_ref * (max_protect * protect_map)
            l_fused_uint8 = np.clip(l_fused, 0, 255).astype(np.uint8)

            lab_fused = cv2.merge([l_fused_uint8, a_chan, b_chan])
            context.working_image = cv2.cvtColor(lab_fused, cv2.COLOR_LAB2RGB)

            med_after = cv2.medianBlur(l_fused_uint8, 3).astype(np.float32)
            mad_after = float(np.median(np.abs(l_fused_uint8.astype(np.float32) - med_after)))
            sigma_after = 1.4826 * mad_after

            context.add_log(
                f"Stage 5 Denoise: Spatially-gated adaptive NLM applied "
                f"(sigma={sigma_noise:.2f} -> {sigma_after:.2f}; "
                f"h_face={h_face:.1f}, h_mid={h_mid:.1f}, h_bg={h_bg:.1f})."
            )
            # Evidence flag for the Restoration Report (T-2).
            context.image_metadata["denoise_applied"] = True

            # G1: HF-retention guard. Laplacian-variance of the content region
            # after denoise vs before. Below 0.55 the stage over-smoothed:
            # blend pre-denoise high frequencies back into textured zones.
            # Below 0.40 the canvas is wax — roll the stage back entirely.
            if _hf_guard:
                try:
                    from ai.analysis.content_region import content_mask as _cm
                    _cmask = _cm(cv2.cvtColor(pre_canvas, cv2.COLOR_RGB2GRAY)) > 0
                    if _cmask.any():
                        g0 = cv2.cvtColor(pre_canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        g1 = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        v0 = float(np.var(cv2.Laplacian(g0, cv2.CV_32F)[_cmask]))
                        v1 = float(np.var(cv2.Laplacian(g1, cv2.CV_32F)[_cmask]))
                        ratio = v1 / max(v0, 1e-6)
                        context.image_metadata["hf_retention_denoise"] = round(ratio, 3)
                        # The guard is only meaningful when the pre-denoise
                        # canvas HAS oriented structure (coherence map, median-
                        # filtered tensor). On a signal-free noise field losing
                        # ~all HF is the job — measured: pure sigma-15 noise
                        # rolled back at 0.05x retention, un-doing a perfect
                        # denoise (sigma 8.90 -> 0.00).
                        from ai.analysis.structure import structure_gate as _sgate
                        _sg = _sgate(g0)
                        _has_structure = bool(float((_sg > 0.25).mean()) > 0.002)
                        if not _has_structure:
                            context.image_metadata["hf_guard_skipped_no_structure"] = True
                        elif ratio < 0.15:
                            # Catastrophic (measured 0.20 on the sepia figure:
                            # sigma 7.41 -> 0.00). No amount of re-injection
                            # rebuilds this; keep the pre-denoise canvas.
                            context.working_image = pre_canvas
                            context.add_log(
                                f"[WARN] HF guard: denoise retained only {ratio:.2f}x "
                                "high-frequency energy — stage rolled back."
                            )
                            warnings_list = context.image_metadata.setdefault("warnings", [])
                            if "denoise_hf_rollback" not in warnings_list:
                                warnings_list.append("denoise_hf_rollback")
                        elif _has_structure and ratio < 0.55:
                            # Selective HF re-injection (G1): the denoised base
                            # keeps flat zones clean; the pre-denoise canvas'
                            # high-pass is re-added ONLY on oriented structure.
                            # The gate is structure-tensor coherence
                            # (ai.analysis.structure), NOT plain Sobel/texture:
                            # isotropic grain has high gradient energy too, and
                            # re-injecting it failed the noise-reduction
                            # contract (sigma 1.48 -> 8.9 on the regression
                            # fixture). Hair strands, collar and jaw lines are
                            # locally oriented — they pass; noise does not.
                            alpha = float(np.clip((0.55 - ratio) / 0.40, 0.0, 0.85))
                            pre_f = cv2.cvtColor(pre_canvas, cv2.COLOR_RGB2LAB).astype(np.float32)
                            cur_f = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2LAB).astype(np.float32)
                            pre_l = pre_f[:, :, 0]
                            pre_hp = pre_l - cv2.GaussianBlur(pre_l, (0, 0), sigmaX=1.2)
                            from ai.analysis.structure import structure_gate as _sgate
                            _pre_gray = cv2.cvtColor(pre_canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
                            # Hair guard (debug session 2026-10-01, RC-3): the
                            # default mag floor (45) sits ABOVE real hair-strand
                            # gradients on small canvases (measured p50 hair grad
                            # 26 on the 447px sepia portrait), so the shipped
                            # gate re-injected collar/jaw but skipped hair and
                            # NLM's 0.20x HF retention froze it into the
                            # "puffed, stamped" look. Floor 24 admits strands;
                            # the coherence term still rejects isotropic grain.
                            # PIXREVIVE_HAIR_GUARD=0 reverts to floor 45.
                            _hair_guard = os.environ.get("PIXREVIVE_HAIR_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")
                            gate = _sgate(_pre_gray, mag_floor=24.0 if _hair_guard else 45.0)
                            # Jaw/neck band boost (G2): bottom 30% of each face
                            # bbox extended 15% below — chin shadow, jawline,
                            # collar region.
                            jaw_band = np.zeros((h_img, w_img), dtype=np.float32)
                            # Hair band boost (RC-3): the region ABOVE each face
                            # bbox (top-of-head hair) plus narrow side bands
                            # (framing hair). The boost only amplifies pixels the
                            # coherence gate already passed — it cannot create
                            # structure in flat background.
                            hair_band = np.zeros((h_img, w_img), dtype=np.float32)
                            for f in faces:
                                bbox = f.get("bbox")
                                if bbox and len(bbox) >= 4:
                                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                                    x0 = max(0, x0); y0 = max(0, y0)
                                    x1 = min(w_img, x1); y1 = min(h_img, y1)
                                    bw_, bh_ = x1 - x0, y1 - y0
                                    if bw_ <= 0 or bh_ <= 0:
                                        continue
                                    hy0 = max(0, y0 - int(0.60 * bh_))
                                    hy1 = min(h_img, y0 + int(0.08 * bh_))
                                    hx0 = max(0, x0 - int(0.15 * bw_))
                                    hx1 = min(w_img, x1 + int(0.15 * bw_))
                                    hair_band[hy0:hy1, hx0:hx1] = 1.0
                                    sy0, sy1 = max(0, y0 - int(0.1 * bh_)), min(h_img, y1 + int(0.1 * bh_))
                                    lx0, lx1 = max(0, x0 - int(0.25 * bw_)), min(w_img, x0 + int(0.05 * bw_))
                                    rx0, rx1 = max(0, x1 - int(0.05 * bw_)), min(w_img, x1 + int(0.25 * bw_))
                                    hair_band[sy0:sy1, lx0:lx1] = 1.0
                                    hair_band[sy0:sy1, rx0:rx1] = 1.0
                            jaw_band = cv2.GaussianBlur(jaw_band, (31, 31), 0)
                            hair_band = cv2.GaussianBlur(hair_band, (31, 31), 0)
                            tw = np.clip(gate * (1.0 + 0.5 * jaw_band + 0.5 * hair_band), 0.0, 1.0) * alpha
                            cur_f[:, :, 0] = np.clip(cur_f[:, :, 0] + tw * pre_hp, 0, 255)
                            context.working_image = cv2.cvtColor(
                                cur_f.astype(np.uint8), cv2.COLOR_LAB2RGB)
                            context.add_log(
                                f"HF guard: denoise retained {ratio:.2f}x HF energy — "
                                f"structure high-frequencies re-injected (alpha={alpha:.2f}"
                                + (", hair band armed" if _hair_guard else "") + ")."
                            )
                except Exception as hf_exc:
                    context.add_log(f"HF guard check failed: {hf_exc}")
        except Exception as exc:
            context.add_log(f"Stage 5 Denoise failed: {exc}")

        context.end_timer("denoising")
        DebugLogger.save_stage(context, "05_after_denoise")
        _refresh_sharpness(context)

    def _stage_7_face_restoration(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 7: GFPGAN Face Restoration (Delta_y=0, scales 0.55/0.65, 0% noise passback)."""
        faces = context.image_metadata.get("faces", [])
        if not faces:
            # Fallback re-scan on current image state
            try:
                self.face_detector.detect(context)
                faces = context.image_metadata.get("faces", [])
            except Exception:
                faces = []

        if not faces:
            context.add_log("Stage 7 FaceRestoration: No faces to restore; skipping.")
            return

        # Pre-SR Small Face / Low-Res Canvas Deferral Check:
        # RetinaFace landmark reliability drops sharply on faces < 90px or canvases < 320px.
        # If SR is enabled and face is small, defer to post-SR where 4x upscale makes landmarks robust.
        is_deferred_pass = context.pipeline_flags.get("_running_deferred_face_restore", False)
        if not is_deferred_pass and context.pipeline_flags.get("enable_sr", True):
            h_curr, w_curr = context.working_image.shape[:2]
            min_dim = min(h_curr, w_curr)
            face_sizes = [
                min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1])
                for f in faces if "bbox" in f and len(f["bbox"]) >= 4
            ]
            min_face_size = min(face_sizes) if face_sizes else 999
            # J-4a: previously the second clause was `min_dim < 320`, which
            # skipped the deferral whenever the canvas was >= 320px even when
            # the face itself was tiny — the tiny face then got GFPGAN at
            # input resolution (upscale=1) and SR re-upscaled the blended
            # result, doubling the small face's artifacts. Now the small-face
            # condition alone defers to the hub after SR.
            if min_dim < 320 or min_face_size < 90:
                context.pipeline_flags["face_restore_deferred_to_post_sr"] = True
                context.image_metadata["gfpgan_deferred"] = True
                context.add_log(
                    f"Stage 7 FaceRestoration: Face size ({min_face_size}px) or canvas ({min_dim}px) "
                    "below landmark reliability threshold (<90px / <320px). Deferring to post-SR."
                )
                return

        DebugLogger.save_stage(context, "06c_before_face_restore")

        if on_progress:
            on_progress("face_restore", 78, "Restoring facial details with GFPGAN...")
        context.start_timer("face_restoration")

        try:
            self.face_restorer.restore(context)
            context.image_metadata["gfpgan_applied"] = True
            context.add_log("Stage 7 FaceRestoration: GFPGAN completed.")

            # J-9 (2026-10-11 forensic pass): face/neck chroma consistency
            # validator. GFPGAN blends RGB pastes that may discard or alter
            # the chroma acquired in Stage 6, producing the measured waxy
            # face / mismatched iris / neck seam. If the mean LAB a/b of the
            # largest face diverges from the neck band below it by > 12,
            # the paste boundary is visibly tone-shifted — log the stat so
            # the restoration report can flag it (non-destructive).
            try:
                _faces = context.image_metadata.get("faces") or []
                if _faces:
                    lb = _faces[0].get("bbox") or []
                    if len(lb) == 4:
                        x0, y0, x1, y1 = (int(v) for v in lb)
                        h_i, w_i = context.working_image.shape[:2]
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(w_i, x1), min(h_i, y1)
                        if x1 > x0 and y1 > y0 and y1 + 40 < h_i:
                            lab_now = cv2.cvtColor(
                                context.working_image, cv2.COLOR_RGB2LAB
                            ).astype(np.float32)
                            face_l = lab_now[y0:y1, x0:x1]
                            neck_band = lab_now[y1:min(h_i, y1 + int((y1 - y0) * 0.4)),
                                                x0:x1]
                            if neck_band.size > 256:
                                fa, fb = face_l[:, :, 1].mean() - 128, face_l[:, :, 2].mean() - 128
                                na, nb = neck_band[:, :, 1].mean() - 128, neck_band[:, :, 2].mean() - 128
                                dev = abs(fa - na) + abs(fb - nb)
                                context.image_metadata["face_neck_chroma_dev"] = round(dev, 2)
                                if dev > 24:
                                    context.add_log(
                                        f"[WARN] FaceNeck Chroma: face vs neck "
                                        f"chroma dev {dev:.1f} > 24 — paste "
                                        "boundary likely tone-shifted."
                                    )
                                # J-9b (2026-10-12): active rebalance. GFPGAN
                                # pastes RGB crops whose chroma can diverge
                                # from Stage 6's colorized skin (measured: waxy
                                # face + green neck seam on the teared
                                # portrait). Shift the FACE band's a/b toward
                                # the neck band's mean by the EXCESS deviation
                                # above 12 (keep a floor so a legitimately
                                # different tonal zone is not leveled), through
                                # a feathered face mask so the correction never
                                # touches hair/background. PIXREVIVE_FACE_NECK_FIX=0
                                # reverts to validator-only behaviour.
                                if (os.environ.get("PIXREVIVE_FACE_NECK_FIX", "1").strip().lower()
                                        not in ("0", "false", "no", "off") and dev > 12):
                                    fb_mask = np.zeros(
                                        (lab_now.shape[0], lab_now.shape[1]), np.float32)
                                    cv2.rectangle(fb_mask, (x0, y0), (x1, y1), 1.0, -1)
                                    fb_mask = cv2.GaussianBlur(fb_mask, (0, 0), sigmaX=max(3.0, (x1 - x0) * 0.06))
                                    excess = min(max(dev - 12.0, 0.0), 20.0) / 2.0
                                    da = np.clip(fa - na, -excess, excess)
                                    db = np.clip(fb - nb, -excess, excess)
                                    lab_fix = lab_now.copy()
                                    lab_fix[:, :, 1] = np.clip(
                                        lab_fix[:, :, 1] - da * fb_mask, 0, 255)
                                    lab_fix[:, :, 2] = np.clip(
                                        lab_fix[:, :, 2] - db * fb_mask, 0, 255)
                                    context.working_image = cv2.cvtColor(
                                        lab_fix.astype(np.uint8), cv2.COLOR_LAB2RGB)
                                    context.add_log(
                                        f"FaceNeck Chroma: face chroma nudged toward "
                                        f"neck (da={-da:.1f}, db={-db:.1f}, mask feather "
                                        f"linear-in-bbox)."
                                    )
            except Exception as _chroma_exc:
                context.add_log(f"[J-9] face/neck chroma check skipped: {_chroma_exc}")
        except Exception as exc:
            context.add_log(f"Stage 7 FaceRestoration failed: {exc}")

        context.end_timer("face_restoration")
        DebugLogger.save_stage(context, "07_after_face_restore")

    def _stage_8_presr_antialiasing(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 8: Pre-SR Bilateral Anti-Aliasing (adaptive on L*, plus gentle chroma diffusion on noisy scans)."""
        DebugLogger.save_stage(context, "07b_before_pre_sr")

        if on_progress:
            on_progress("pre_sr", 85, "Applying pre-SR anti-aliasing filter...")

        try:
            lab = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            
            noise_score = float(context.image_metadata.get("noise_score", 0.0))
            sigma_c = max(7.0, 1.0 * noise_score) if noise_score > 6.0 else 7.0

            # Hair guard (RC-3b): the bilateral's job is killing grain spikes
            # before SR amplifies them, but measured on the teared portrait it
            # also halved residual hair-strand gradients (p50 1.94 -> 1.18) —
            # the last crispness left after NLM. Blend the bilateral result
            # toward the original L wherever the structure gate says oriented
            # texture lives, so strands survive to SR. PIXREVIVE_HAIR_GUARD=0
            # reverts to the unconditioned filter.
            l_smooth = cv2.bilateralFilter(l_chan, d=5, sigmaColor=sigma_c, sigmaSpace=3.0)
            if os.environ.get("PIXREVIVE_HAIR_GUARD", "1").strip().lower() not in ("0", "false", "no", "off"):
                try:
                    from ai.analysis.structure import structure_gate as _sg8
                    _g8 = _sg8(l_chan.astype(np.float32), mag_floor=24.0)
                    if float(_g8.mean()) > 0.002:
                        _keep = np.clip(_g8 * 1.2, 0.0, 0.85)
                        l_smooth = (
                            l_smooth.astype(np.float32) * (1.0 - _keep)
                            + l_chan.astype(np.float32) * _keep
                        ).astype(l_chan.dtype)
                except Exception:
                    pass

            # On noisy scans, also diffuse residual color grain on a* and b* before 4x Real-ESRGAN sharpening
            if noise_score > 8.0:
                a_smooth = cv2.bilateralFilter(a_chan, d=5, sigmaColor=12.0, sigmaSpace=3.0)
                b_smooth = cv2.bilateralFilter(b_chan, d=5, sigmaColor=12.0, sigmaSpace=3.0)
            else:
                a_smooth, b_smooth = a_chan, b_chan

            lab_clean = cv2.merge([l_smooth, a_smooth, b_smooth])
            context.working_image = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)
            context.add_log(f"Stage 8 Pre-SR Anti-Aliasing: Bilateral filter applied (sigmaColor={sigma_c:.1f}).")

            # -- T-6: Banding pass (registry stage 'deband') -----------------
            # Plan T-6: dedicated bit-depth-expansion + dithering pass applied
            # pre-SR when banding_score is high. Gated on the noise router's
            # banding measurement (corpus max 0.503; gate 0.6 — synthetic
            # posterization reads 1.0). Super-resolution then amplifies
            # whatever stairs remain, so this is the last cheap exit.
            try:
                from ai.postprocessing.deband import deband
                debanded, deband_info = deband(context.working_image)
                if deband_info.get("applied"):
                    context.working_image = debanded
                    context.add_log(
                        f"Stage 8 Deband: quantization stairs smoothed "
                        f"(score={deband_info.get('banding_score')}) (T-6)."
                    )
            except Exception as deband_exc:
                context.add_log(f"Stage 8 Deband skipped: {deband_exc}")
        except Exception as exc:
            context.add_log(f"Stage 8 Pre-SR Anti-Aliasing failed: {exc}")

        DebugLogger.save_stage(context, "08_after_pre_sr")

    def _stage_9_super_resolution(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
        """
        Stage 9: Real-ESRGAN x4+ Super-Resolution.

        Fix 9: Delegates to SuperResolution.restore(context) instead of calling
        _realesrgan_tile_forward() directly. SuperResolution.restore() provides:
          - OOM safety cap (output capped at 4096×4096)
          - Pre-SR halftone/mesh suppression for scanned prints
          - 2K output normalisation (longest side → 2048px)
          - Graceful bicubic fallback with correct aspect ratio
          - Post-SR LAB sharpening (strength=0.22)
        """
        DebugLogger.save_stage(context, "08b_before_super_res")

        if on_progress:
            on_progress("super_res", 90, "Upscaling with Real-ESRGAN via SuperResolution.restore()...")
        context.start_timer("super_resolution")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.super_resolution.restore(context)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            h_out, w_out = context.working_image.shape[:2]
            context.add_log(
                f"Stage 9 SuperResolution: completed via SuperResolution.restore() — "
                f"output {w_out}x{h_out}."
            )
        except Exception as exc:
            context.add_log(
                f"Stage 9 SuperResolution failed ({exc}) — applying Lanczos 4x fallback."
            )
            h_in, w_in = context.working_image.shape[:2]
            context.working_image = cv2.resize(
                context.working_image,
                (w_in * 4, h_in * 4),
                interpolation=cv2.INTER_LANCZOS4,
            )

        context.end_timer("super_resolution")
        DebugLogger.save_stage(context, "09_after_super_res")

    def _stage_10_post_sr_sharpen(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
        """
        Stage 10: Post-SR Luminance Crispness Calibration.
        
        Applies edge-directed multi-scale sharpening strictly in LAB luminance (L*) space
        to enhance internal borders (facial contours, jawline, eyes, lips, hair borders,
        hat brims, cloth folds, silhouette boundaries) without amplifying flat background noise
        or creating chromatic halos.
        """
        if on_progress:
            on_progress("sharpen", 95, "Enhancing internal borders and contours...")
        context.start_timer("post_sr_sharpen")

        try:
            img = context.working_image
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            l_chan = lab[:, :, 0]
            a_chan = lab[:, :, 1]
            b_chan = lab[:, :, 2]

            # J-4c: residual speck sweep (pre-sharpen). Stage 0's global
            # tophat only kills specks evaluated against a uniform background;
            # isolated bright flecks that survived it get 4x LARGER after SR
            # and are then amplified by sharpening. Remove connected bright
            # AND dark specks below a size cap that is far below anything with
            # real photographic content (faces/strands are >> 64px connected).
            PIXREVIVE_SPECK_SWEEP = os.environ.get(
                "PIXREVIVE_SPECK_SWEEP", "1"
            ).strip().lower() not in ("0", "false", "no", "off")
            if PIXREVIVE_SPECK_SWEEP:
                try:
                    h_sp, w_sp = l_chan.shape[:2]
                    speck_cap = max(24, int((h_sp * w_sp) ** 0.5 * 0.010))  # ~1% of mean side
                    u8 = lab.astype(np.uint8)
                    l_u8 = u8[:, :, 0]
                    a_u8 = u8[:, :, 1]
                    b_u8 = u8[:, :, 2]
                    # Local background via median blur; residual = what departs from it.
                    l_bg = cv2.medianBlur(l_u8, 9)
                    l_res = cv2.absdiff(l_u8, l_bg)
                    iso = (l_res > 22).astype(np.uint8)
                    if iso.any():
                        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(iso, connectivity=8)
                        kill = np.zeros_like(iso)
                        removed = 0
                        for i in range(1, n_lbl):
                            area = int(stats[i, cv2.CC_STAT_AREA])
                            if area <= speck_cap:
                                kill[lbl == i] = 255
                                removed += 1
                        if removed:
                            kill_f = cv2.GaussianBlur(
                                (kill.astype(np.float32) / 255.0), (0, 0), sigmaX=1.1
                            )[..., None]
                            # Neutralise by pulling isolated speck pixels to the
                            # local median background in LAB (both L and chroma).
                            a_bg = cv2.medianBlur(a_u8, 9).astype(np.float32)
                            b_bg = cv2.medianBlur(b_u8, 9).astype(np.float32)
                            l_bg_f = l_bg.astype(np.float32)
                            lab_f = lab
                            lab_f[:, :, 0] = lab_f[:, :, 0] * (1.0 - kill_f[..., 0]) + l_bg_f * kill_f[..., 0]
                            lab_f[:, :, 1] = lab_f[:, :, 1] * (1.0 - kill_f[..., 0]) + a_bg * kill_f[..., 0]
                            lab_f[:, :, 2] = lab_f[:, :, 2] * (1.0 - kill_f[..., 0]) + b_bg * kill_f[..., 0]
                            l_chan = lab[:, :, 0]
                            a_chan = lab[:, :, 1]
                            b_chan = lab[:, :, 2]
                            context.add_log(
                                f"Stage 10 Speck sweep: {removed} residual post-SR specks "
                                f"neutralised (cap={speck_cap}px)."
                            )
                except Exception as sweep_exc:
                    context.add_log(f"Stage 10 speck sweep skipped: {sweep_exc}")

            # 1. Edge & Contour Detection via Sobel Gradient Magnitude
            gx = cv2.Sobel(l_chan, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(l_chan, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            # Smooth edge gate: 0.0 in smooth areas, ramping up to 1.0 along defined edges/borders
            edge_weight = np.clip((grad_mag - 12.0) / 45.0, 0.0, 1.0)
            # Soft-clamp extreme contrast transitions to avoid ringing halos
            edge_weight = edge_weight * np.clip(1.0 - (grad_mag - 130.0) / 100.0, 0.35, 1.0)

            # 2. Multi-Scale Luminance Unsharp Masking
            # Fine scale (sigma=1.2): recovers micro-textures (hair strands, fabric weave, eyelashes)
            diff_micro = l_chan - cv2.GaussianBlur(l_chan, (0, 0), sigmaX=1.2)
            # Contour / border scale (sigma=3.2): specifically sharpens 4-8px object borders & facial contours
            diff_contour = l_chan - cv2.GaussianBlur(l_chan, (0, 0), sigmaX=3.2)

            # Combine: gentle micro-detail everywhere + pronounced crispness on internal borders
            l_crisp = l_chan + 0.25 * diff_micro + 0.45 * edge_weight * diff_contour

            # J-4b face-aware damping: inside detected-face ROIs, cap the contour
            # gain at 60% so GFPGAN-synthesised skin is not re-etched into a
            # plastic/stylised look. Background keeps full gain (that is where
            # crispness was missing). PIXREVIVE_FACE_SHARPEN=0 reverts.
            if os.environ.get("PIXREVIVE_FACE_SHARPEN", "1").strip().lower() not in ("0", "false", "no", "off"):
                faces = (context.image_metadata or {}).get("faces") or []
                h_img, w_img = l_chan.shape[:2]
                face_damp = np.ones((h_img, w_img), dtype=np.float32)
                any_face = False
                for f in faces:
                    box = f.get("bbox") if isinstance(f, dict) else None
                    if not box or len(box) < 4:
                        continue
                    px1, py1, px2, py2 = (int(round(box[0])), int(round(box[1])),
                                          int(round(box[2])), int(round(box[3])))
                    # Face boxes are rescaled to the working canvas by
                    # sync_geometry_to_working_image(); clamp defensively.
                    px1 = max(0, min(w_img - 1, px1))
                    py1 = max(0, min(h_img - 1, py1))
                    px2 = max(px1 + 1, min(w_img, px2))
                    py2 = max(py1 + 1, min(h_img, py2))
                    any_face = True
                    face_damp[py1:py2, px1:px2] = 0.6
                if any_face:
                    face_damp = cv2.GaussianBlur(face_damp, (0, 0), sigmaX=float(max(3.0, min(h_img, w_img) * 0.01)))
                    face_damp = np.clip(face_damp, 0.6, 1.0)[..., None]
                    # Damp only the contour term inside faces; keep micro detail.
                    l_crisp = l_chan + 0.25 * diff_micro + (0.45 * face_damp) * edge_weight * diff_contour
                else:
                    face_damp = None

            # B5 (docs/REMEDIATION_PLAN_AB_G.md): clip guard — unsharp
            # undershoot crushes dark hair to L*<=1. Attenuate the sharpening
            # contribution where the result would clip and the pre-sharpen
            # canvas was not already clipped. PIXREVIVE_CLIP_GUARD=0 reverts.
            if os.environ.get("PIXREVIVE_CLIP_GUARD", "1").strip().lower() not in ("0", "false", "no", "off"):
                pre_clip_dark = (l_chan <= 1.5)
                would_clip = (l_crisp <= 0.5) & ~pre_clip_dark
                if would_clip.any():
                    damp = np.where(would_clip, 0.35, 1.0)
                    l_crisp = l_chan + damp * (l_crisp - l_chan)
                n_saved = int(would_clip.sum()) if would_clip.any() else 0
            l_crisp = np.clip(l_crisp, 0, 255)

            lab[:, :, 0] = l_crisp
            context.working_image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            context.add_log(
                "Stage 10 Sharpen: Edge-directed multi-scale crispness applied "
                "(micro=0.25, border_contour=0.45 on L* channel"
                + (f"; clip guard damped {n_saved}px)." if os.environ.get("PIXREVIVE_CLIP_GUARD", "1").strip().lower() not in ("0", "false", "no", "off") and n_saved else ")."
                )
            )
        except Exception as exc:
            context.add_log(f"Stage 10 Sharpen failed: {exc}")

        context.end_timer("post_sr_sharpen")
        DebugLogger.save_stage(context, "10_after_sharpen")

    def _stage_10b_projector_grade(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
        """Stage 10b: Projector Display Grading (CLAHE, shadow-lifted S-curve, and calibrated tonal grading)."""
        DebugLogger.save_stage(context, "10_before_projector_grade")

        if on_progress:
            on_progress("projector_grade", 98, "Applying projector exhibition grading...")
        context.start_timer("projector_grade")

        try:
            img = context.working_image
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Step 1: Calibrated CLAHE on L* with tamed clipLimit and larger tile grid
            # Prevents local-contrast halos and the 'glossy embossed/varnish' look on skin/fabric
            h_img, w_img = img.shape[:2]
            grid_size = max(16, min(32, w_img // 128))
            # B5: cap the CLAHE clip so post-CLAHE shadow clipping stays within
            # input+1% (dynamic check, not a hardcoded device assumption).
            clip_limit = 0.85
            if os.environ.get("PIXREVIVE_CLIP_GUARD", "1").strip().lower() not in ("0", "false", "no", "off"):
                _pre_dark = float(np.mean(l <= 1))
                clip_limit = 0.85
                for _ in range(3):
                    _probe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)).apply(l)
                    if float(np.mean(_probe <= 1)) <= _pre_dark + 0.01:
                        break
                    clip_limit *= 0.6
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            l_clahe = clahe.apply(l)

            # Step 2: Smooth natural filmic tone curve (gentle shadow lift, neutral midtones)
            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                x = i / 255.0
                # Filmic exhibition response: lifts true blacks slightly (0.01) with linear-preserving gamma
                y = 0.01 + 0.99 * (x ** 0.96)
                lut[i] = int(np.clip(y * 255, 0, 255))
            l_curved = cv2.LUT(l_clahe, lut)

            # Step 3: Merge calibrated tonal response
            lab_graded = cv2.merge([l_curved, a, b])
            context.working_image = cv2.cvtColor(lab_graded, cv2.COLOR_LAB2RGB)
            context.add_log("Stage 10b ProjectorGrade: Tamed CLAHE (clip=0.85, grid=16-32) and natural film curve applied.")

            # J-10 (2026-10-12 forensic pass): film-grain re-injection.
            # Stages 5 + 9: the NLM flattens the grain floor and Real-ESRGAN
            # synthesises detail but never photographic grain — the composite
            # result reads waxy/watercolour even when structure survives.
            # Inject resolution-scaled, midtone-weighted grain on the L*
            # channel over the CONTENT region only (scanner-bed margins stay
            # clean), slightly correlated (sigma 0.8) so it reads as emulsion
            # grain, not snow. Amplitude 2.2 LAB units at 1932px, scaled by
            # sqrt(canvas/1932) so the visible grain density is stable.
            # PIXREVIVE_GRAIN=0 reverts.
            if os.environ.get("PIXREVIVE_GRAIN", "1").strip().lower() not in ("0", "false", "no", "off"):
                try:
                    from ai.analysis.content_region import content_mask as _cm10b
                    _gray10b = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY)
                    _cmask = (_cm10b(_gray10b) > 0).astype(np.float32)
                    if _cmask.any():
                        h_g, w_g = l_curved.shape[:2]
                        rng = np.random.default_rng()
                        g_noise = rng.normal(0.0, 1.0, (h_g, w_g)).astype(np.float32)
                        g_noise = cv2.GaussianBlur(g_noise, (0, 0), 0.8)
                        g_noise = g_noise / max(float(g_noise.std()), 1e-6)
                        l_f32 = l_curved.astype(np.float32)
                        midtone = np.clip(1.0 - np.abs((l_f32 - 128.0) / 127.0), 0.0, 1.0) ** 1.5
                        amp = 2.2 * float(np.sqrt(min(h_g, w_g) / 1932.0))
                        l_grain = np.clip(
                            l_f32 + amp * midtone * g_noise * _cmask, 0, 255
                        ).astype(np.uint8)
                        lab_graded = cv2.merge([l_grain, a, b])
                        context.working_image = cv2.cvtColor(lab_graded, cv2.COLOR_LAB2RGB)
                        context.add_log(
                            f"Stage 10b Grain: film-grain re-injected over content "
                            f"(amp={amp:.1f}, coverage={float(_cmask.mean()):.2f})."
                        )
                except Exception as grain_exc:
                    context.add_log(f"Stage 10b grain re-injection skipped: {grain_exc}")
        except Exception as exc:
            context.add_log(f"Stage 10b ProjectorGrade failed: {exc}")

        context.end_timer("projector_grade")
        DebugLogger.save_stage(context, "10b_after_projector_grade")

    # ── Main Restore Entry Point ──────────────────────────────────────────────


    def _stage_13_semantic_validate(self, context: PipelineContext,
                                    on_progress: Optional[Callable]) -> None:
        """Stage 13 (Phase V0, docs/SEMANTIC_VALIDATION_PLAN.md): report-only
        semantic rule battery over the evidence mask.

        NEVER touches pixels (Prime Directive). Classifies the lost-evidence
        regions (anatomy eye quadrants, hair band, garment torso) and emits
        findings into metadata["semantic"] for the report. In V0 every finding
        is disclosure-only; V1 promotes them to targeted re-imagine masks.
        """
        # Internal stage (Phase V0): report-only and emits NO progress event —
        # the same contract as refresh_faces — so the frontend dot row and the
        # stream audit's known-id set (stage_ids()) stay untouched.
        context.start_timer("semantic_validate")
        try:
            from ai.validation.semantic_validator import run_semantic_validation

            meta = context.image_metadata
            masks = context.damage_masks
            evidence = None
            for key in ("final", "inpaint", "damage"):
                m = masks.get(key)
                if m is not None and np.any(m > 0):
                    evidence = (m > 0).astype(np.uint8)
                    break
            if evidence is None:
                evidence = np.zeros(context.working_image.shape[:2], np.uint8)
            block = run_semantic_validation(
                context.working_image, evidence,
                meta.get("faces") or [])
            meta["semantic"] = block
            n = len(block.get("findings") or [])
            context.add_log(
                f"Stage 13 Semantic: {n} finding(s) across "
                f"{block.get('faces_evaluated', 0)} face(s) (report-only).")
        except Exception as exc:  # noqa: BLE001 — validator must never break the pipeline
            logger.warning("Stage 13 semantic validation failed: %s", exc)
            context.image_metadata["semantic"] = {
                "version": "v0-report-only", "error": str(exc),
                "findings": [], "summary": {"eyes": 0, "hair": 0, "garment": 0},
            }
        finally:
            context.end_timer("semantic_validate")
