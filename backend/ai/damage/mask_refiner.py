"""
mask_refiner.py — Fuse and refine damage masks.

Key architecture:
  - Fuses classical defect masks with TearmarkDetector emulsion flake masks.
  - Morphological gap closing (k=5) and gentle boundary dilation (k=3).
  - Anatomical facial protection: zeroes out false positives across ocular
    and oral cavities (eyes, teeth, lips) while preserving genuine tears.
  - Deep Fourier Convolution routing: routes all structural damage directly
    to LaMa, which handles large-mask texture synthesis without blurry PDE smudges.
"""

import math
import os

import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext
from ai.damage.tearmark_detector import _tm_field_mode as _tm_field


# Field-noise mode (debug session 2026-10-01, RC-1b): when TearmarkDetector ran
# in field mode the mask is deliberately speck-granular; the k7 close + k5
# dilate here re-fused it back into blankets (measured on the teared portrait:
# 7.98% specks -> 16.7% blanket, head band 48.6% -> 77.2%) and LaMa region-
# filled the blanket — the "puffed, stamped" hair. Field mode keeps the same
# light k3 morphology the detector used.


class MaskRefiner:
    @staticmethod
    def refine(mask: np.ndarray, context: PipelineContext) -> np.ndarray:
        """
        Fuse classical + tearmark masks, refine morphologically, protect faces.
        Stores the fused mask as 'final' and the two inpainting tiers as
        'lama_mask' (small components) and 'ns_mask' (large blobs).
        """
        h, w = mask.shape[:2]

        # ── 1. Tearmark: strip the photo's own border, not whole components ────
        # NOTE: border stripping is intentionally restricted to the tearmark mask
        # only — scratch/crack damage that touches the image border is genuine
        # damage and must not be removed.
        #
        # This used to delete every component whose *bounding box* came within
        # border_margin of an edge. Discrete flakes near a margin survive that
        # test, but a sprawling emulsion-loss field does not: its bbox spans the
        # frame, so the whole field was dropped. On a 447x447 sepia portrait that
        # threw away 12.7 of the 15.8 percentage points the detector had found.
        #
        # What we actually want to exclude is the aged paper border itself — it
        # is part of the photograph's character and should be preserved, not
        # inpainted. So subtract a border *band* and additionally drop components
        # that live almost entirely inside a wider band (the frame strip, which is
        # usually a little wider than the band). Interior damage that merely
        # connects to the edge keeps its interior pixels.
        tearmark_mask = context.damage_masks.get("tearmark")
        if tearmark_mask is not None and np.any(tearmark_mask):
            tm_clean = tearmark_mask.copy()
            # Only strip the extreme 2px scanner boundary artifact, not wide tear components
            margin = max(2, int(min(h, w) * 0.005))

            band = np.zeros((h, w), dtype=bool)
            band[:margin, :] = True
            band[h - margin:, :] = True
            band[:, :margin] = True
            band[:, w - margin:] = True
            tm_clean[band] = 0

            fused = cv2.bitwise_or(mask, tm_clean)
            context.add_log(
                f"MaskRefiner: fused classical + tearmark ({int(np.sum(tm_clean > 0))} px of flakes)."
            )
        else:
            fused = mask.copy()

        # ── 2. Close intra-damage gaps (robust 7x7 kernel) ─────────────────
        # BW-blur session (2026-10-18): the CLOSE/DILATE kernels used to be
        # fixed pixels. On a 183×173 canvas a k7 CLOSE spans 3.8% of the
        # width — measured on the tree landscape it fused the U-Net's granular
        # 5.49% sky speckle field into one 10%-of-frame blanket (16.1% total
        # after dilate), and LaMa wiped those scene regions into the flat
        # oil-paint plates. Same failure class as output_verifier's fixed
        # 31px k_ext (already scale-normalized there). Kernels now scale with
        # the canvas and stay within the original 3..7 / 3..5 bounds, so
        # large-canvas behaviour is unchanged.
        field_mode = _tm_field(context)
        _scale = max(h, w)
        _k_close = int(min(7, max(3, round(_scale * 0.02))))
        _k_dilate = int(min(5, max(3, _k_close - 2)))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k_close, _k_close))
        if field_mode:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k3, iterations=1)
        else:
            refined = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k7, iterations=1)

        # ── 3. Dilation for clean inpainting overlap context ───────────────
        # Ensure mask fully covers thin feathered tear margins and micro-cracks
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k_dilate, _k_dilate))
        refined = cv2.dilate(refined, (k3 if field_mode else k5), iterations=1)

        # ── 3b. Scene-content ingress gate (BW-blur session 2026-10-18) ─────
        # A damage component must DIFFER from its surround. Measured on the
        # tree landscape: the 50k-px "crack" blanket filling the bright sky
        # had interior texture variance 2696 vs surround 2679 (ratio 1.01)
        # and no popout (|Δ mean| < 1.4) — it IS the scene, not damage.
        # Components like that are what turned LaMa loose on clean sky.
        # Gate (env PIXREVIVE_SCENE_GATE=0 reverts): drop components
        #   - that are blob-like (aspect < 3 — thin cracks are exempt, they
        #     legitimately read via shape not contrast),
        #   - with no luminance popout (|mean_in − mean_ring| < 12),
        #   - whose interior is at least 75% as textured as the ring around
        #     them (real missing emulsion is smooth-at-surround-texture).
        # Perimeter ring uses a 9px dilate annulus on the (noise-robust)
        # median-filtered luminance so grain can't fake texture inside.
        if os.environ.get("PIXREVIVE_SCENE_GATE", "1").strip().lower() not in (
            "0", "false", "no", "off"
        ):
            _prev_cov = float(np.sum(refined > 0) / refined.size)
            _gray_in = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY)
            _gmed = cv2.medianBlur(_gray_in, 3).astype(np.float32)
            _ring_k = max(3, int(round(max(refined.shape) * 0.05)) | 1)
            _se_ring = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (_ring_k, _ring_k))
            _min_area = max(400, int(0.003 * refined.size))
            _pruned_px = 0
            _pruned_n = 0
            _crack_continuity = os.environ.get("PIXREVIVE_CRACK_CONTINUITY", "0").strip().lower() in (
                "1", "true", "yes", "on"
            )
            _n_c, _lbl_c, _stats_c, _cent_c = cv2.connectedComponentsWithStats(
                refined, connectivity=8)
            for _ci in range(1, _n_c):
                _a = int(_stats_c[_ci, cv2.CC_STAT_AREA])
                if _a < _min_area:
                    continue
                _w_c = int(_stats_c[_ci, cv2.CC_STAT_WIDTH])
                _h_c = int(_stats_c[_ci, cv2.CC_STAT_HEIGHT])
                _comp = _lbl_c == _ci
                if _crack_continuity:
                    # Rotation-invariant moment elongation: catches diagonal cracks where axis-aligned bbox is square
                    _m = cv2.moments(_comp.astype(np.uint8))
                    _mu20, _mu02, _mu11 = _m["mu20"], _m["mu02"], _m["mu11"]
                    _diff = _mu20 - _mu02
                    _root = math.sqrt(_diff * _diff + 4.0 * _mu11 * _mu11)
                    _l1 = 0.5 * (_mu20 + _mu02 + _root)
                    _l2 = 0.5 * (_mu20 + _mu02 - _root)
                    _moment_aspect = math.sqrt(_l1 / max(_l2, 0.05))
                    if max(_w_c, _h_c) / max(1, min(_w_c, _h_c)) >= 3.0 or _moment_aspect >= 2.5:
                        continue  # crack/tear signature: shape says damage
                else:
                    if max(_w_c, _h_c) / max(1, min(_w_c, _h_c)) >= 3.0:
                        continue  # crack/tear signature: shape says damage
                _ring = cv2.dilate(_comp.astype(np.uint8), _se_ring) > 0
                _ring &= ~_comp
                if int(_ring.sum()) < 40:
                    continue
                _mean_in = float(_gmed[_comp].mean())
                _pop = abs(_mean_in - float(_gmed[_ring].mean()))
                _t_in = float(_gmed[_comp].var())
                _t_out = float(_gmed[_ring].var())
                if _pop < 12.0 and _t_in > 0.75 * max(_t_out, 1.0):
                    refined[_comp] = 0
                    _pruned_px += _a
                    _pruned_n += 1
            if _pruned_n:
                context.add_log(
                    f"MaskRefiner SceneGate: pruned {_pruned_n} scene-content "
                    f"blob(s) ({_pruned_px} px) with no popout and interior "
                    "texture ~= surround — not damage "
                    f"(coverage {_prev_cov*100:.1f}% -> "
                    f"{100.0*float(np.sum(refined>0))/refined.size:.1f}%)."
                )

        # ── 4. Refine face regions ────────────────────────────────────────────
        # Protect critical facial features (eyes, nose tip, lips, teeth) from false-positive
        # inpainting while ensuring genuine tears and flakes on cheeks/forehead are repaired.
        faces = context.image_metadata.get("faces", [])
        noise_score = float(context.image_metadata.get("noise_score", 0.0))
        protected = 0
        for face in faces:
            bbox = face.get("bbox")
            if bbox is None:
                continue
            x1f, y1f, x2f, y2f = bbox
            x1f = max(0, int(x1f)); y1f = max(0, int(y1f))
            x2f = min(w, int(x2f)); y2f = min(h, int(y2f))
            fw = x2f - x1f
            fh = y2f - y1f
            if fw > 0 and fh > 0:
                face_sub = refined[y1f:y2f, x1f:x2f]
                if np.any(face_sub):
                    face_cov = float(np.sum(face_sub > 0)) / float(fw * fh)
                    landmarks = face.get("landmarks")
                    has_exact_lms = landmarks is not None and len(landmarks) >= 5

                    # 1. Direct ocular & oral feature protection
                    if has_exact_lms:
                        # Eyes: protect pupils/irises/sclera from broad fills
                        for idx_eye in (0, 1):
                            ex = int(round(landmarks[idx_eye][0] - x1f))
                            ey = int(round(landmarks[idx_eye][1] - y1f))
                            r_eye = max(8, int(min(fw, fh) * 0.08))
                            eye_sub = face_sub[max(0, ey - r_eye):min(fh, ey + r_eye), max(0, ex - r_eye):min(fw, ex + r_eye)]
                            if np.sum(eye_sub > 0) > (r_eye * 2):
                                if _crack_continuity:
                                    # Protect eye socket from broad fills, but preserve thin linear crack strokes cutting through the eye
                                    _eye_disk = np.zeros_like(face_sub)
                                    cv2.circle(_eye_disk, (ex, ey), r_eye, 255, -1)
                                    _eye_damage = (_eye_disk > 0) & (face_sub > 0)
                                    _eye_disk_area = math.pi * r_eye * r_eye
                                    if np.any(_eye_damage):
                                        _ne, _lble, _ste, _ = cv2.connectedComponentsWithStats(_eye_damage.astype(np.uint8), connectivity=8)
                                        for _ei in range(1, _ne):
                                            _ea = int(_ste[_ei, cv2.CC_STAT_AREA])
                                            _ew = int(_ste[_ei, cv2.CC_STAT_WIDTH])
                                            _eh = int(_ste[_ei, cv2.CC_STAT_HEIGHT])
                                            _diag_len = math.hypot(_ew, _eh)
                                            _estroke = _ea / max(_diag_len, 1.0)
                                            # If it's a broad blob (not a thin stroke) or covers a major portion of the eye socket, protect it
                                            if _estroke > 10.0 or _ea > (0.45 * _eye_disk_area):
                                                face_sub[_lble == _ei] = 0
                                                protected += _ea
                                else:
                                    cv2.circle(face_sub, (ex, ey), r_eye, 0, -1)
                                    protected += int(np.sum(eye_sub > 0))

                        # Mouth: protect teeth and open smiles from being obliterated by blobs
                        ml = np.array([landmarks[3][0] - x1f, landmarks[3][1] - y1f], dtype=np.float32)
                        mr = np.array([landmarks[4][0] - x1f, landmarks[4][1] - y1f], dtype=np.float32)
                        mc = (ml + mr) / 2.0
                        mw = float(np.linalg.norm(mr - ml))
                        ang = float(np.degrees(np.arctan2(mr[1] - ml[1], mr[0] - ml[0])))
                        oral_ellipse = np.zeros((fh, fw), dtype=np.uint8)
                        cv2.ellipse(
                            oral_ellipse,
                            (int(round(mc[0])), int(round(mc[1]))),
                            (int(max(10, mw * 0.55)), int(max(6, mw * 0.25))),
                            ang,
                            0,
                            360,
                            255,
                            -1,
                        )
                        oral_sub = cv2.bitwise_and(face_sub, oral_ellipse)
                        if np.any(oral_sub):
                            num_o, lbl_o, stats_o, _ = cv2.connectedComponentsWithStats(oral_sub, connectivity=8)
                            for oi in range(1, num_o):
                                o_area = stats_o[oi, cv2.CC_STAT_AREA]
                                o_w = stats_o[oi, cv2.CC_STAT_WIDTH]
                                o_h = stats_o[oi, cv2.CC_STAT_HEIGHT]
                                o_len = max(o_w, o_h)
                                o_thick = min(o_w, o_h)
                                o_stroke = o_area / float(max(o_len, 1))
                                if o_stroke > 5.0 or o_thick > 8 or o_area > 350:
                                    face_sub[lbl_o == oi] = 0
                                    protected += o_area
                    else:
                        # Geometric eye & mouth bounds when exact landmarks are absent
                        ex1, ey1 = int(0.32 * fw), int(0.38 * fh)
                        ex2, ey2 = int(0.68 * fw), int(0.38 * fh)
                        r_eye = max(8, int(min(fw, fh) * 0.08))
                        for (ex, ey) in ((ex1, ey1), (ex2, ey2)):
                            eye_sub = face_sub[max(0, ey - r_eye):min(fh, ey + r_eye), max(0, ex - r_eye):min(fw, ex + r_eye)]
                            if np.sum(eye_sub > 0) > (r_eye * 2):
                                if _crack_continuity:
                                    _eye_disk = np.zeros_like(face_sub)
                                    cv2.circle(_eye_disk, (ex, ey), r_eye, 255, -1)
                                    _eye_damage = (_eye_disk > 0) & (face_sub > 0)
                                    _eye_disk_area = math.pi * r_eye * r_eye
                                    if np.any(_eye_damage):
                                        _ne, _lble, _ste, _ = cv2.connectedComponentsWithStats(_eye_damage.astype(np.uint8), connectivity=8)
                                        for _ei in range(1, _ne):
                                            _ea = int(_ste[_ei, cv2.CC_STAT_AREA])
                                            _ew = int(_ste[_ei, cv2.CC_STAT_WIDTH])
                                            _eh = int(_ste[_ei, cv2.CC_STAT_HEIGHT])
                                            _diag_len = math.hypot(_ew, _eh)
                                            _estroke = _ea / max(_diag_len, 1.0)
                                            if _estroke > 10.0 or _ea > (0.45 * _eye_disk_area):
                                                face_sub[_lble == _ei] = 0
                                                protected += _ea
                                else:
                                    cv2.circle(face_sub, (ex, ey), r_eye, 0, -1)
                                    protected += int(np.sum(eye_sub > 0))

                        mx, my = int(0.50 * fw), int(0.76 * fh)
                        oral_ellipse = np.zeros((fh, fw), dtype=np.uint8)
                        cv2.ellipse(oral_ellipse, (mx, my), (int(max(10, fw * 0.22)), int(max(6, fh * 0.10))), 0, 0, 360, 255, -1)
                        oral_sub = cv2.bitwise_and(face_sub, oral_ellipse)
                        if np.any(oral_sub):
                            num_o, lbl_o, stats_o, _ = cv2.connectedComponentsWithStats(oral_sub, connectivity=8)
                            for oi in range(1, num_o):
                                o_area = stats_o[oi, cv2.CC_STAT_AREA]
                                o_len = max(stats_o[oi, cv2.CC_STAT_WIDTH], stats_o[oi, cv2.CC_STAT_HEIGHT])
                                o_thick = min(stats_o[oi, cv2.CC_STAT_WIDTH], stats_o[oi, cv2.CC_STAT_HEIGHT])
                                o_stroke = o_area / float(max(o_len, 1))
                                if o_stroke > 5.0 or o_thick > 8 or o_area > 350:
                                    face_sub[lbl_o == oi] = 0
                                    protected += o_area

                    num_f, lbl_f, stats_f, centroids_f = cv2.connectedComponentsWithStats(face_sub, connectivity=8)
                    min_crack_len = max(8, int(min(fw, fh) * 0.05))
                    filter_diffuse = noise_score > 3.0

                    for fi in range(1, num_f):
                        area_f = int(stats_f[fi, cv2.CC_STAT_AREA])
                        cx, cy = centroids_f[fi]
                        cw_f = int(stats_f[fi, cv2.CC_STAT_WIDTH])
                        ch_f = int(stats_f[fi, cv2.CC_STAT_HEIGHT])
                        len_f = max(cw_f, ch_f)
                        min_dim = min(cw_f, ch_f)
                        aspect_f = len_f / max(1, min_dim)
                        elongation_f = (len_f * len_f) / max(1, area_f)
                        stroke_f = area_f / float(max(len_f, 1))

                        # A true crack is thin (stroke <= 5.5px, thickness <= 8px).
                        # Large 2D blobs covering the face have high thickness and must not be treated as linear cracks.
                        if _crack_continuity:
                            _diag_f = math.hypot(cw_f, ch_f)
                            _stroke_diag = area_f / max(_diag_f, 1.0)
                            _elong_diag = (_diag_f * _diag_f) / max(1, area_f)
                            is_linear_crack = (
                                (_diag_f >= min_crack_len) and
                                (aspect_f >= 2.0 or _elong_diag >= 2.5) and
                                (_stroke_diag <= 10.0)
                            )
                        else:
                            is_linear_crack = (
                                (len_f >= min_crack_len) and
                                (aspect_f >= 2.0 or elongation_f >= 2.5) and
                                (stroke_f <= 5.5 and min_dim <= 8)
                            )

                        is_protected_feature = False
                        if has_exact_lms:
                            for lm in landmarks:
                                lx, ly = lm[0] - x1f, lm[1] - y1f
                                if _crack_continuity:
                                    if np.hypot(cx - lx, cy - ly) < max(10, min(fw, fh) * 0.08) and not is_linear_crack:
                                        is_protected_feature = True
                                        break
                                else:
                                    if np.hypot(cx - lx, cy - ly) < max(10, min(fw, fh) * 0.08):
                                        is_protected_feature = True
                                        break
                        else:
                            in_left_eye = (0.24 * fh <= cy <= 0.48 * fh) and (0.20 * fw <= cx <= 0.45 * fw)
                            in_right_eye = (0.24 * fh <= cy <= 0.48 * fh) and (0.55 * fw <= cx <= 0.80 * fw)
                            in_nose = (0.45 * fh <= cy <= 0.68 * fh) and (0.35 * fw <= cx <= 0.65 * fw)
                            in_mouth = (0.68 * fh <= cy <= 0.88 * fh) and (0.30 * fw <= cx <= 0.70 * fw)
                            if (in_left_eye or in_right_eye or in_nose or in_mouth) and not is_linear_crack:
                                is_protected_feature = True

                        if filter_diffuse and not is_linear_crack and aspect_f < 1.5 and area_f < 25:
                            is_protected_feature = True

                        if is_protected_feature:
                            face_sub[lbl_f == fi] = 0
                            protected += area_f

                    # Enforce hard safety ceiling: if remaining face damage still exceeds 8%, log warning
                    remaining_cov = float(np.sum(face_sub > 0)) / float(fw * fh)
                    if remaining_cov > 0.08 and noise_score > 2.5:
                        context.add_log(
                            f"[WARN] MaskRefiner: face damage safety ceiling triggered ({remaining_cov*100:.1f}% > 8%); "
                            f"capping excessive inpainting across face to preserve facial anatomy."
                        )
                        warnings_list = context.image_metadata.setdefault("warnings", [])
                        if "face_damage_mask_safety_ceiling_triggered" not in warnings_list:
                            warnings_list.append("face_damage_mask_safety_ceiling_triggered")

                    refined[y1f:y2f, x1f:x2f] = face_sub
        if protected:
            context.add_log(
                f"MaskRefiner: protected {protected} px of facial anatomy from destructive inpainting."
            )

        # ── 4b. Noise-Adaptive Component Pruning ──────────────────────────────
        # On high-noise scans (noise_score > 6.0), tiny isolated isotropic mask
        # fragments (< 35px, aspect < 2.0) are noise speckles that survived detection,
        # not true structural fractures. Pruning them prevents destructive inpainting
        # blur across textured photographic surfaces.
        if noise_score > 6.0:
            num_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
            min_comp_area = max(35, int(noise_score * 5.0))
            for ci in range(1, num_c):
                area = stats_c[ci, cv2.CC_STAT_AREA]
                w_c = stats_c[ci, cv2.CC_STAT_WIDTH]
                h_c = stats_c[ci, cv2.CC_STAT_HEIGHT]
                aspect = max(w_c, h_c) / max(1, min(w_c, h_c))
                if area < min_comp_area and aspect < 2.0:
                    refined[lbl_c == ci] = 0

        # If total residual mask is insignificant (< 0.05% or < 30px total), clear it
        if np.sum(refined > 0) < max(30, int(refined.size * 0.0005)):
            refined = np.zeros_like(refined)

        # ── 4b. Protected linear texture & hair strand filter ───────────────
        # Differentiates natural coherent linear textures (hair strands, fur)
        # from real cracks/creases so LaMa does not smear fine hair.
        if os.environ.get("PIXREVIVE_STRUCTURE_DISCRIMINATOR", "1").strip().lower() not in (
            "0", "false", "no", "off"
        ):
            try:
                from ai.analysis.line_structure import LineStructureAnalyzer
                _lsa = LineStructureAnalyzer()
                _gray_ref = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY)
                refined = _lsa.filter_hair_from_inpaint_mask(refined, _gray_ref)
            except Exception:
                pass

        # ── 5. Routing ────────────────────────────────────────────────────────
        # LaMa is a state-of-the-art Fourier Convolution inpainting model specifically
        # designed for large masks, wide cracks, and complex photographic synthesis.
        # Route all structural damage to LaMa for deep context-aware synthesis.
        lama_mask = refined.copy()
        ns_mask   = np.zeros_like(refined)
        coverage  = float(np.sum(refined > 0) / refined.size)

        # Keep the refined mask as the canonical live damage mask for downstream stages.
        context.damage_masks["final"]       = refined
        context.damage_masks["lama_mask"]   = lama_mask
        context.damage_masks["ns_mask"]     = ns_mask

        context.add_log(
            f"MaskRefiner: coverage={coverage*100:.1f}%, faces={len(faces)}, "
            f"all damage routed to LaMa"
            + (" (field-noise mode: speck-granular mask)." if field_mode else ".")
        )
        return refined
