# Remediation Plan — Image A (teared sepia) & Image B (speckle Lena) + G1–G3

Recon date: 2026-09-30. Pipeline: 14 stages orchestrated in
`backend/ai/pipeline/restoration_pipeline.py` (see `CODEBASE_MAP.md`).
Every change below sits behind an env flag (`PIXREVIVE_*`, default **on**,
revertible by setting `=0`) and is self-gated by measurement so clean images
take the same path as before. No stage is rewritten; each fix is a localized,
measured intervention.

Test inputs: **A** = `test demos/teared.jpg` (case9_teared_sepia),
**B** = `test demos/SigShrink-…speckle…png` (= case8_speckle_lena_margins,
196×148 → 4× SR).

## Defect → root cause → exact change

### A2 — residual dust / white specks
- **Root cause:** `restore()` (restoration_pipeline.py L513) runs LaMa only on
  the *structural* mask; Stage 0 ROAD filter removes only tiny impulse
  outliers; Stage 11 `output_verifier.detect_residue` micro-cleans only
  *bright* outliers *outside face boxes*. Mid-size isolated blobs (dust,
  stains, emulsion dots — bright OR dark) that no detector claimed survive to
  the output.
- **Change:** new `suppress_isolated_blobs()` in
  `ai/preprocessing/impulse_denoiser.py` (top-hat + black-hat + median-diff
  detector → CC size ≤ `blob_max_area`, isolation filter — component far from
  any structure edge → dilate 3 → `cv2.inpaint` TELEA r=3), wired into
  `_stage_0_despeckle` after the ROAD pass. Flag `PIXREVIVE_BLOB_INPAINT`.
- **Metric:** residual speck count on output (existing top-hat CC counter) —
  target: ≈0 visible; harness reports before/after.

### G1 — over-smoothing (A3 wax, B1 painterly, B2 structure loss)
- **Root cause:** `_stage_5_spatially_gated_denoise` (finish_stages.py):
  (1) `texture_weight = raw_texture * max(0, 1 − σ/15)` — texture protection
  *vanishes exactly when NLM h is largest* (σ≈14 on B); (2) `h_bg` reaches
  11.5–1.4σ on noisy frames even where the background is undamaged;
  (3) no HF-retention accountability after denoise / SR.
- **Change (flag `PIXREVIVE_STRUCTURE_GUARD`, `PIXREVIVE_HF_GUARD`):**
  1. texture protection floor: `texture_weight = max(raw_texture, raw_texture
     * (1 − σ/15) )` → never below 0.35·raw_texture in high-texture zones;
     raise `max_protect` ceiling for texture pixels.
  2. `h_bg` cap 10.0 when the background zone is undamaged (grain-preserving
     path the E7 calibration already assumes).
  3. HF retention guard: after Stage 5 (and after Stage 2b), Laplacian
     variance of the working canvas on the content region vs the pre-stage
     canvas; if ratio < 0.55, blend back the pre-stage canvas' median-HF in
     protected zones; if ratio < 0.40, roll the stage canvas back and log a
     warning + report flag.
- **Metric:** HF retention ratio (Laplacian var out/in, content-masked);
  visual: hair/feather/fabric texture survives.

### A1 / B3 — eye inconsistency & pasted GFPGAN look
- **Root cause:** `face_restorer._restore_single_face` composites with a
  fixed feature-zone floor (≥0.72) and chin fade; nothing measures the two
  eyes' sharpness/iris chroma after compositing; skin around features gets
  GFPGAN's smooth texture at full alpha when the source skin was textured
  (waxy look, B3).
- **Change (flags `PIXREVIVE_EYE_HARMONY`, `PIXREVIVE_SKIN_TEXTURE_BLEND`):**
  in the W1 seam-check section, per face: eye windows (from landmarks, 24px
  on 512 crops) → if Lap-var ratio > 1.8, unsharp the weaker eye toward the
  sharper (bounded 2 passes); iris hue: sample iris-ring hue/sat both eyes →
  desaturate the more vivid iris toward the mean (cap S ≤ 60 for
  grayscale/sepia sources), record `eye_harmony` verdicts in metadata.
  Skin-texture blend: scale `feature_alpha_floor` toward 0.55 in *non-feature
  skin* when restored-crop micro-texture variance < 0.5× original (waxy
  indicator), keeping eyes/mouth at full floor.
- **Metric:** eye sharpness ratio (target < 1.5), iris hue Δ (target ≈ 0).

### G2 — missing collar / jawline / shoulder edges (A5, B6)
- **Root cause:** Stage 2b cluster smoothing + Stage 5 NLM smooth low-contrast
  structure (collar, jaw shadow) before any edge-aware mechanism sees it;
  nothing measures contour retention along jaw/shoulder/collar lines.
- **Change (flag `PIXREVIVE_STRUCTURE_GUARD`):**
  1. Stage 1 analyze computes a multi-scale **structure mask** (Sobel +
     tensor-energy, low threshold, content-masked) stored once;
     Stage 2b skips voxels where the structure mask is high; Stage 5 raises
     `max_protect` along it (replaces the current post-hoc protect cap).
  2. `face_restorer`: narrow protected band along the jawline (landmark-based
     when available, anatomical otherwise) — chin fade already protects
     submental; band extends protection to the jawline itself, plus a subtle
     soft contact-shadow gradient under the chin (opacity ≤ 0.10, sepia-safe).
  3. Collar HF restoration: HoughLinesP in the neck/shoulder ROI on the input
     vs post-SR canvas; lost line segments get the pre-denoise canvas' HF
     blended back along them at Stage 10 (edge-masked unsharp).
- **Metric:** gradient magnitude along jaw/shoulder/collar ROI, out/in ratio
  (target ≥ 0.6; below → report "contour lost" flag).

### G3 — color discipline (A6, B4)
- **Root cause:** `ddcolor_colorizer` applies one global saturation cap
  (160) and opacity to all sources; sepia sources get the same vivid chroma
  as color photos; a/b channels are not luminance-guided, so hue bleeds
  across luminance edges; irises can get different hues.
- **Change (flags `PIXREVIVE_CHROMA_DISCIPLINE`):**
  1. `ddcolor_colorizer.colorize`: for `color_type == "sepia"` lower blend
     opacity (×0.65) and max sat ceiling 110; grayscale keeps existing caps.
  2. `color_refiner.reduce_artifacts`: luminance-guided a/b smoothing pass
     (guidedFilter with L as guide) after colorization — hue changes align
     with luminance edges (stops bleed; helps collar/shirt & chin/neck
     separation).
  3. `restoration_report`: when source was grayscale/sepia and colorization
     ran, flagged line gains "source is monochrome — colors are synthetic".
- **Metric:** chroma-bleed score (chroma-gradient not co-located with
  luminance edges) before/after.

### A4 — seams / posterized patches / tile blocks
- **Root cause:** `_seam_smooth` feathers a 7px band with guided filter but
  LaMa infill tone steps inside large regions persist; SR tile forward blends
  overlap padding linearly (check `_realesrgan_tile_forward`) — hard tile
  edges on posterized zones; BandingReducer only runs inside
  `reduce_artifacts` when colorize ran.
- **Change (flag `PIXREVIVE_SEAM_QUALITY`):**
  1. seam-step detector along mask boundary (|Δ| across band); when step >
     threshold, gradient-domain (Poisson, `cv2.seamlessClone`) blend of the
     inpainted region — replaced fallback: wider feather.
  2. SR tile blending: cosine feather on the 32px overlap (replaces replace-
     style write if it is one).
  3. Final light BandingReducer pass on L* for all outputs (not only
     colorized) behind the same flag.
- **Metric:** seam-step score along mask boundary + tile grid periodicity.

### B5 — crushed shadows
- **Root cause:** Stage 10 micro-sharpen clips dark hair (L* negative
  undershoot at 0.25 weight) and Stage 10b CLAHE (0.85) + S-curve cannot
  recover clipped pixels; nothing measures clipping.
- **Change (flag `PIXREVIVE_CLIP_GUARD`):** after Stage 10 and 10b, measure
  clipped-pixel fraction (L* ≤ 1 or ≥ 254) on the content region; if dark
  clip > 2% and input was lower, attenuate the stage's contribution
  (re-blend with pre-stage canvas weighted by clip excess); cap CLAHE clip
  dynamically so post-CLAHE clip fraction ≤ input + 1%.
- **Metric:** shadow clip fraction before/after; visual: hair detail returns.

### B2 — noise mislabeled as damage
- **Root cause:** GrainField gate (σ>6 & damage<6) already discards
  noise-derived masks; MaskRefiner prunes tiny isotropic fragments on noisy
  frames. Residual risk: mid-size noise clusters (speck fields) pass the
  linear-crack test on speckle-heavy inputs.
- **Change:** extend `MaskRefiner` component filter: on `content_sigma > 4`,
  require component edge-co-location (component must overlap the structure
  mask from Stage 1) — noise islands in flat regions are dropped from the
  LaMa tier. Flag `PIXREVIVE_STRUCTURE_GUARD` (shared).
- **Metric:** Stage-2 final-mask coverage on B before/after (should drop);
  LaMa no longer repaints background.

### Validators (report-only)
- **Change:** new `ai/validation/quality_audit.py` — pure functions:
  `eye_symmetry`, `hf_retention_ratio`, `residual_speck_count`,
  `chroma_bleed_score`, `seam_step_score`, `edge_retention_ratio` (jaw/
  shoulder/collar ROIs from faces/landmarks + structure mask). Wired into
  Stage 11 `verify_and_clean` record as `quality_audit` block; surfaced in
  `restoration_report` flagged list ("contour lost", "eye asymmetry",
  "chroma bleed", "HF retention low"). Report-only — never alters pixels.
- **Tests:** unit tests for each metric on synthetic images (flat vs textured,
  symmetric vs asymmetric eyes, bled chroma).

## Execution order (per PROCESS)
1. Metric harness (`.freebuff/ab_metrics.py`) + baseline on A & B.
2. A2 → 3. G1 → 4. A1/B3 → 5. G2 → 6. G3 → 7. A4 → 8. B5 → 9. validators.
   After each: both images, before/after numbers, visual note.
10. Full pytest suite (backend paused) vs 384-baseline; restart servers.

## VRAM note
All offline quality runs execute with the dev server paused (4 GB GTX 1650
cannot hold server models + offline pipeline). Runtime budgeting in code
stays dynamic (resource_profiler tiers) — nothing hardcoded.

---

## Debug session 2026-10-01 (RC-1..RC-4): output-quality degradation

User evidence: `restored_teared.png` + `restored_SigShrink-...png` (Downloads),
byte-identical to live-server results — all pre-existing flags were active;
degradation was current pipeline behavior. Root causes verified by probes
(`.freebuff/probe_tearmark_field.py`, `probe_tearmark_v5.py`,
`probe_honeycomb_hair.py`), user's primary complaint: hair looks "puffed,
stamped", not crisp.

| RC | Root cause (measured) | Fix | Flag |
|----|------------------------|-----|------|
| RC-1 | TearmarkDetector flags mottle field as flakes (24.19%); k5×2 CLOSE + MaskRefiner k7+dilate fuse specks into a head-shaped blanket (7.98%→16.7%); LaMa region-fills it → watercolor bg + wiped hair | field-noise mode: no pass2, pass3@p93/40, k3×1 close, 12% strongest-first cap; MaskRefiner keeps k3 morphology | `PIXREVIVE_TM_FIELD` |
| RC-3 | HF re-injection mag floor 45 > real hair gradients (p50 26); pre-SR bilateral halves residual strand gradients → NLM's 0.20x HF retention freezes hair "puffed" | floor 24 + hair-band boost above face bbox; bilateral blended back on structure gate (mean>0.002) | `PIXREVIVE_HAIR_GUARD` |
| RC-2 | 163 bright residual "drip" patches survive in-face (verifier excluded faces entirely) | `dissolve_face_drips`: compact bright specks inside faces, eyes/mouth landmark-protected, ≤0.15%/comp, ≤1% face area, Telea r=2 | `PIXREVIVE_FACE_DRIP` |
| RC-4a | 435×342 noisy grayscale figure → DDColor invents blue/orange wash (4 hues, passes Q-3 single-hue); SR posterizes into blobs | low-res gray skip: ≤1.2MP grayscale + (noise>2.5 ∨ sigma>4 ∨ grain-type detected) → stay gray; report disclosure | `PIXREVIVE_GRAY_SKIP` |
| RC-4b | Sepia mottle: DDColor wash passed Q-3 (multi-hue) and replaced paper warmth (b +31→+4, a=+8.4 magenta lean) — pink cast at viewing scale | background-wash guard on sepia: strong field (frac|ab|>15 > 35%) OR sepia-drift (post b < 0.6×pre-b, or a>4∧a>b with pre-b≥8) → luminance-preserving rollback | (shared with Q-3 path) |

Measured (ab_metrics, offline cuda, backend paused):
- A (teared): contour 0.34–0.43 → **0.45–0.47**; chroma_bleed 0.0116 → **0.0094–0.0144**;
  sepia warmth preserved (final bg b=+19.9 kept, pink field eliminated).
  Stage-2 mask 32.95% → **9.9%** on case9 fixture (blanket no longer forms;
  shipped-mode 35.03% recoverable via `PIXREVIVE_TM_FIELD=0`). Speck_count 255→438–502:
  authentic texture (incl. damage specks) survives instead of being blanket-inpainted —
  deliberate anti-watercolor trade accepted with the user.
- B (case8 speckle Lena): stays grayscale (chroma_bleed 0.0308 → **0.0096**);
  contour 0.888 → **0.988**; eye 0.967; hf 0.99; specks 333 → 169.
- case9 contract tests recalibrated 10% → 8% (field-mode mask 9.89% vs
  blanket-era 35%; intent — heavy tears route to inpainting — unchanged).

New tests: `tests/test_rc_debug_fixes.py` (26).
