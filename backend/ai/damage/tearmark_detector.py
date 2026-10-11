"""
tearmark_detector.py — White emulsion flake / paper-flake detector.

Three passes, unioned:

  1. Multi-scale local contrast + absolute brightness floor — small/medium
     discrete flakes.
  2. Coarse-scale contrast only — large flake clusters, whose fine-scale
     contrast is low because the whole neighbourhood is bright.
  3. Flake-scale white top-hat — everything the absolute brightness floor in
     passes 1 and 2 structurally cannot reach.

Pass 3 exists because a global brightness floor is the wrong instrument for a
print whose damage spans both dark and light regions. Measured on a 447x447
sepia portrait, cream flakes on the dark shirt sit at L ~= 180 while healthy
forehead skin sits at L ~= 200: no floor separates them, so the shipped
two-pass detector missed the shirt entirely while any relaxation masked the
forehead as damage. A top-hat measures brightness excess against the *local*
background, so it is blind to that distinction, and it cancels smooth features
larger than its kernel outright — a forehead cannot be flagged at all.

Two guards on the saturation test are worth knowing about, because both were
silent-failure sources:

  * The relative term (`S < S_local - k`) needs the channel to carry signal. On
    a desaturated input S == 0 everywhere, so the test is `0 < -k` — false at
    every pixel, and every pass silently returns an empty mask. The pipeline
    used to neutralise sepia before this stage ran, which is exactly what
    happened: a visibly flake-covered portrait reported 0.00%.
  * The absolute term (`S < 80`) assumes a flake is near-neutral paper white.
    That holds for a white-stock print, not a sepia-toned one, where bare stock
    is mid-tan: the portrait above averages S = 138, so `S < 80` rejected most
    of the frame and was measurably the tightest gate in the whole detector.
    Pass 3 therefore uses the relative term alone.
"""

import os

import cv2
import numpy as np
from ai.models.base_model import BaseDetector
from ai.pipeline.pipeline_context import PipelineContext


# ── Field-noise mode (debug session 2026-10-01, RC-1) ──────────────────────────
# On a mottled sepia scan the Stage-2 grain gate may already have declared the
# noise-derived masks grain ("mask is grain, not damage") — and then this
# detector re-admitted the same field as "emulsion flakes": measured 24.19%
# flagged (pass2 XL-contrast 6.3% + pass3 p90 top-hat 9.8%), which the k5×2
# CLOSE below fused into a head-shaped blanket (raw specks 7.98% → 20.6% in
# the head band) and LaMa re-synthesized wholesale — the "puffed, stamped"
# hair and watercolor background on the teared portrait.
#
# Field mode engages only on frames the grain gate claimed (or whose content
# sigma is high with genuinely flaky color_type), and changes three things:
#   1. pass2 (coarse XL-contrast) is dropped — it admits the mottle field.
#   2. pass3 anchors at p93/40 instead of p90/32 (real flakes live at the
#      amplitude tail: p95 tophat = 119 on the mottle field vs p50–p75 =
#      24–45).
#   3. Morphology uses a single k3 close — k5×2 fuses adjacent specks into
#      blankets; speck granularity is what keeps LaMa from region-filling.
# PIXREVIVE_TM_FIELD=0 reverts to the shipped behaviour.
def _tm_field_enabled() -> bool:
    return os.environ.get("PIXREVIVE_TM_FIELD", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _tm_field_mode(context: PipelineContext) -> bool:
    if not _tm_field_enabled():
        return False
    color_type = context.image_metadata.get("color_type", "color")
    if color_type not in ("sepia", "color"):
        return False
    if context.image_metadata.get("grain_field_mask_discarded", False):
        return True
    sigma = float(context.image_metadata.get("content_sigma", 0.0) or 0.0)
    return sigma > 6.0


class TearmarkDetector(BaseDetector):

    @property
    def model_name(self) -> str:
        return "TearmarkDetector"

    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def detect(self, context: PipelineContext) -> np.ndarray:
        img_np     = context.working_image
        h, w       = img_np.shape[:2]
        color_type = context.image_metadata.get("color_type", "color")
        field_mode = _tm_field_mode(context)

        # ── Grayscale guard ───────────────────────────────────────────────────
        # Tearmarks are white emulsion flakes on sepia/colour photos.
        # On pure grayscale images S=0 everywhere, so the saturation guard is
        # meaningless and noise spikes trigger false positives.
        # Classical TopHat handles real scratches; tearmark is not needed.
        if color_type in ("grayscale", "monochrome"):
            context.add_log("TearmarkDetector: grayscale/monochrome image — skipping (no saturation signal).")
            empty = np.zeros((h, w), dtype=np.uint8)
            context.damage_masks["tearmark"] = empty
            return empty

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        L   = lab[:, :, 0].astype(np.float32)

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        S   = hsv[:, :, 1].astype(np.float32)

        photo_median = float(np.median(L))
        photo_mean   = float(np.mean(L))
        photo_std    = float(np.std(L))

        if photo_median > 215:
            context.add_log("TearmarkDetector: high-key image — skipping.")
            empty = np.zeros((h, w), dtype=np.uint8)
            context.damage_masks["tearmark"] = empty
            return empty

        # ── Kernel sizes ──────────────────────────────────────────────────────
        base = max(h, w)
        ks_s = max(9,  int(base * 0.015)) | 1
        ks_m = max(21, int(base * 0.035)) | 1
        ks_l = max(41, int(base * 0.065)) | 1
        ks_xl = max(81, int(base * 0.12))  | 1   # extra-large for blob detection

        lm_s  = cv2.blur(L, (ks_s,  ks_s))
        lm_m  = cv2.blur(L, (ks_m,  ks_m))
        lm_l  = cv2.blur(L, (ks_l,  ks_l))
        lm_xl = cv2.blur(L, (ks_xl, ks_xl))

        # ── Brightness floor: mean + 2.5σ, capped at 200 ─────────────────────
        abs_floor = min(max(photo_mean + 2.5 * photo_std, photo_median + 20), 200)

        S_local = cv2.blur(S, (ks_m, ks_m))

        # Flakes are bare paper showing through a tinted emulsion, so they read as
        # locally *less* saturated than their surroundings. That test needs the
        # channel to actually carry signal: on a desaturated input S == 0
        # everywhere, so `S < S_local - 3` is 0 < -3, false at every pixel, and
        # both passes below silently produce an empty mask. The pipeline used to
        # neutralise sepia before this stage ran, which is exactly what happened —
        # a visibly flake-covered portrait reported 0.00%. Fall back to the
        # absolute ceiling alone when the channel is flat, and say so.
        s_std = float(np.std(S))
        if s_std > 2.0:
            sat_ok = (S < 80) & (S < (S_local - 3))
            sat_ok_loose = (S < 80) & (S < (S_local - 2))
            # Purely relative variant, with no absolute ceiling. The `S < 80`
            # term above assumes a flake is near-neutral paper white, which is
            # true of a white-stock print but not of a sepia-toned one: on the
            # 447x447 portrait the whole frame averages S = 138, so `S < 80`
            # rejected most of it and was measurably the tightest gate in the
            # detector (dropping it alone took pass 2 from 3.4% to 13.1%).
            # Pass 3 keeps the physically meaningful half of the test — a flake
            # is less saturated than the emulsion around it — and drops the
            # magic constant.
            sat_rel = S < (S_local - 2)
        else:
            context.add_log(
                f"TearmarkDetector: saturation channel is flat (std={s_std:.2f}) — "
                "relative-saturation test disabled, using brightness only."
            )
            sat_ok = sat_ok_loose = sat_rel = np.ones(S.shape, dtype=bool)

        # ── Pass 1: Small/medium flakes (multi-scale contrast) ────────────────
        ts, tm, tl = (18, 14, 10) if color_type == "sepia" else (20, 16, 12)
        pass1 = (
            (L - lm_s > ts) &
            (L - lm_m > tm) &
            (L - lm_l > tl) &
            (L > abs_floor)  &
            sat_ok
        )

        # ── Pass 2: Large flake clusters (coarse-scale only) ─────────────────
        # Large clusters appear locally uniform so the fine-scale contrast is
        # low. Use only the XL-scale contrast (whole-image context).
        # Threshold is very loose — rely on brightness and saturation guard.
        large_floor = max(abs_floor - 10, photo_median + 15)
        pass2 = (
            (L - lm_xl > 8)   &  # brighter than large neighbourhood
            (L > large_floor)  &
            sat_ok_loose
        )

        # ── Pass 3: Flake-scale white top-hat (background-level blind) ────────
        # Passes 1 and 2 both gate on an absolute brightness floor, and on a
        # print whose damage spans dark and light regions no single such floor
        # exists. Measured on the 447x447 sepia portrait: the cream flakes on
        # the dark shirt sit at L ~= 180 while healthy forehead skin sits at
        # L ~= 200, so every floor either missed the shirt entirely (floor 200,
        # the shipped behaviour) or masked the forehead as damage (floor 190 and
        # below). Sweeping the floor across p75..p92 moved coverage by under a
        # point, which is what showed the floor was never the binding term.
        #
        # A white top-hat has no such blind spot. `L - opening(L, k)` keeps only
        # bright structures smaller than the structuring element, measured
        # against whatever the local background happens to be, so a flake reads
        # the same on a dark shirt as on light paper. Conversely a smooth
        # feature *larger* than the kernel — a forehead, a cheek — survives the
        # opening and cancels to zero, so it cannot be flagged at all. That is
        # the property the brightness floor was failing to provide.
        #
        # The threshold is anchored to the top-hat's own tail so it tracks scan
        # contrast, but floored in absolute L units so the pass stays inert on
        # undamaged input: a bare percentile flags its complement's worth of
        # every image, damaged or not. Measured p92 of the k=9 top-hat is 96 on
        # the damaged scan, 67 after fine-speck suppression, and 12 on the
        # restored result, so a floor of 40 separates damaged from clean with
        # room to spare.
        ks_flake = min(15, max(9, int(base * 0.02)) | 1)
        se_flake = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_flake, ks_flake))
        tophat = L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se_flake)
        th_std = float(np.std(tophat))
        if field_mode:
            flake_thresh = max(float(np.percentile(tophat, 93)), 40.0)
        elif th_std > 8.0:
            # Damaged print: catch clear discrete flakes without masking natural hair strands
            flake_thresh = max(float(np.percentile(tophat, 90)), 32.0)
        else:
            flake_thresh = max(float(np.percentile(tophat, 93)), 40.0)
        pass3 = (tophat > flake_thresh) & sat_rel

        if field_mode:
            combined = (pass1 | pass3).astype(np.uint8) * 255
        else:
            combined = (pass1 | pass2 | pass3).astype(np.uint8) * 255

        # ── Morphological cleanup ─────────────────────────────────────────────
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        if field_mode:
            # Speck-granular morphology: one k3 close (bridges 1px gaps only).
            # k5×2 fuses neighbouring specks into blankets and LaMa then
            # region-fills the whole blanket — the measured "puffed hair".
            mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k3, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        else:
            mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k5, iterations=2)
            mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  k3, iterations=1)

        # ── Component area filter ─────────────────────────────────────────────
        # The upper bound is solidity-aware, matching MaskRefiner's inpainting-
        # tier routing. A flat `area > 20%` cap is a solid-blob guard — it exists
        # to reject a large bright *object* (a lit wall, a white garment) that the
        # brightness tests mistook for paper. A sprawling emulsion-loss field is
        # also large, but it is a lace of specks with valid pixels everywhere
        # between them, so a flat cap deletes exactly the damage we want. Solidity
        # (area over convex hull area) tells the two apart: measured on the
        # 447x447 portrait the flaked hair scores 0.50 while solid border strips
        # score 0.87-0.98. Drop a large component only when it is also solid.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        area_cap = h * w * 0.20
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 4:
                mask[labels == i] = 0
                continue
            if area > area_cap:
                comp = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(
                    comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)
                if hull_area > 0 and area / hull_area >= 0.85:
                    mask[labels == i] = 0

        # ── Safety valve ──────────────────────────────────────────────────────
        ratio = float(np.sum(mask > 0) / mask.size)
        if field_mode and ratio > 0.12:
            # Strongest-first component cap for field-noise frames: keep the
            # 12% of frame area with the highest mean top-hat amplitude (real
            # flakes are the amplitude tail; the mottle field is the bulk).
            num_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            comps = []
            for i in range(1, num_c):
                m = lbl_c == i
                comps.append((float(tophat[m].mean()) if m.any() else 0.0, i, int(stats_c[i, cv2.CC_STAT_AREA])))
            comps.sort(reverse=True)
            keep = np.zeros_like(mask)
            budget = 0.12 * mask.size
            used = 0
            for _amp, i, area in comps:
                if used + area > budget:
                    continue
                used += area
                keep[lbl_c == i] = 255
            context.add_log(
                f"TearmarkDetector: field-noise mode — capped {ratio*100:.1f}% -> "
                f"{used/mask.size*100:.1f}% strongest-first (p93 top-hat, no pass2)."
            )
            mask = keep
            ratio = used / mask.size
        if not field_mode and ratio > 0.45:
            context.add_log(
                f"TearmarkDetector: {ratio*100:.1f}% flagged (>45%) - tightening."
            )
            # Keep the top-hat term in the fallback, at a stricter percentile.
            # Without it the tightened mask reverts to a brightness floor and
            # discards all of pass 3's recall on dark regions.
            sat_tight = sat_rel if color_type == "sepia" else (S < 60)
            cand_tight = (
                ((L - lm_s > ts + 8) & (L - lm_m > tm + 6) &
                 (L > abs_floor + 10) & sat_tight)
                | ((tophat > max(float(np.percentile(tophat, 95)), 45.0)) & sat_rel)
            )
            mask = cand_tight.astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)

        # ── Facial anatomy protection ─────────────────────────────────────────
        # Teeth (bright enamel, low saturation) and eyes (white sclera, specular glints)
        # naturally mimic emulsion flakes. Zero out false-positive flake detections
        # inside anatomical eye and oral zones to preserve authentic human expressions.
        faces = context.image_metadata.get("faces", [])
        for f in faces:
            bbox = f.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
            fw, fh = max(1, x1 - x0), max(1, y1 - y0)
            landmarks = f.get("landmarks")
            if landmarks is not None and len(landmarks) >= 5:
                # Left eye, right eye
                for idx_eye in (0, 1):
                    ex, ey = int(round(landmarks[idx_eye][0])), int(round(landmarks[idx_eye][1]))
                    r_eye = max(8, int(min(fw, fh) * 0.08))
                    cv2.circle(mask, (ex, ey), r_eye, 0, -1)

                # Mouth cavity (lips & teeth) between left and right mouth corners
                m_l = np.array(landmarks[3], dtype=np.float32)
                m_r = np.array(landmarks[4], dtype=np.float32)
                m_center = (m_l + m_r) / 2.0
                m_width = float(np.linalg.norm(m_r - m_l))
                angle_deg = float(np.degrees(np.arctan2(m_r[1] - m_l[1], m_r[0] - m_l[0])))
                axes = (int(max(10, m_width * 0.58)), int(max(6, m_width * 0.28)))
                cv2.ellipse(
                    mask,
                    (int(round(m_center[0])), int(round(m_center[1]))),
                    axes,
                    angle_deg,
                    0,
                    360,
                    0,
                    -1,
                )
            else:
                # Fallback to standard anthropometric facial zones
                ex1, ey1 = int(x0 + 0.32 * fw), int(y0 + 0.38 * fh)
                ex2, ey2 = int(x0 + 0.68 * fw), int(y0 + 0.38 * fh)
                r_eye = max(8, int(min(fw, fh) * 0.08))
                cv2.circle(mask, (ex1, ey1), r_eye, 0, -1)
                cv2.circle(mask, (ex2, ey2), r_eye, 0, -1)
                mx, my = int(x0 + 0.50 * fw), int(y0 + 0.76 * fh)
                axes = (int(max(10, fw * 0.22)), int(max(6, fh * 0.10)))
                cv2.ellipse(mask, (mx, my), axes, 0, 0, 360, 0, -1)

        final_ratio = float(np.sum(mask > 0) / mask.size)
        context.add_log(
            f"TearmarkDetector: {final_ratio*100:.2f}% flagged as flakes "
            f"(floor={abs_floor:.0f}, color_type={color_type}, "
            f"pass1={float(np.sum(pass1)/pass1.size*100):.1f}%, "
            f"pass2={float(np.sum(pass2)/pass2.size*100):.1f}%, "
            f"pass3={float(np.sum(pass3)/pass3.size*100):.1f}% "
            f"[tophat k={ks_flake} thr={flake_thresh:.0f}])."
        )
        context.damage_masks["tearmark"] = mask
        return mask
