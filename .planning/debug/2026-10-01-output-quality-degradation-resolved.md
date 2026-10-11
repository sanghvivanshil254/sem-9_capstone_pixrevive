# Debug Session: Output quality degradation (teared portrait + SigShrink Lena)

**Status:** RESOLVED (2026-10-01) — all RCs fixed, measured, suite green
**Started:** 2026-10-01 ~22:15 (user local)
**Mode:** debug (implement fixes worst-first, user approved)
**User evidence:** `C:\Users\sangh\Downloads\restored_teared.png` (1341x1341),
`restored_SigShrink-...png` (1305x1026) — both byte-identical to live-server results
(`backend/temp_results/0129e16bf11b.png`, `758df71ee30e.png`).

## Verified provenance (not stale, not a regression)
- Backend PID 13292 started 00:50:19 > last source edit 00:39 → all remediation flags active.
- Downloads ≡ server results (sha256 match). User downloads ≡ `.freebuff/ab_out_*_SHIP4_*` within
  CPU/GPU nondeterminism (mean|diff| 1.76 / 1.47, 99.98% px within ±8).
- So: degradation = current fully-fixed pipeline behavior.

## User symptom ranking (asked via request_user_input)
1. Hair / silky fibres "puffed / stamped", not crisp — **primary complaint**.
2. All four artifact families (wax skin, watercolor bg, drips, hallucinated color).
3. A (teared) first — "lots of problems: honeycomb pattern, salt&pepper, missing/flaked head parts";
   B "butter smoothed" — fix both.
4. B colorization: "see what is best and act accordingly".

## Root causes (evidence)
- **RC-1** TearmarkDetector over-flags sepia mottle as emulsion flakes: 24.19% flagged
  → mask 32.95% → LaMa inpaints 50.6% of frame → watercolor background. (debug_output/07_full_log.txt)
- **RC-2** 163 bright residual patches remain in face zone (semantic B3 finding) → whitish drips.
- **RC-3** Hair crispness loss: Stage-5 NLM retained only 0.20x HF energy; Stage-8 pre-SR bilateral
  (sigmaColor 9.8) flattens hair micro-texture; RealESRGAN amplifies smoothness → "puffed" hair.
- **RC-4** B input = 561x420 grayscale paper figure; colorize=True default → DDColor invents
  blue/orange/yellow; SR+sharpen posterize color blobs.

## Plan (worst-first, user-approved)
1. Probe A honeycomb periodicity + hair-band gradients (hypothesis: LaMa inpaint texture + CLAHE).
2. RC-1: mottle-vs-flake discrimination in TearmarkDetector + coverage cap.
3. RC-3: hair-band HF retention (user's #1 complaint).
4. RC-2: dissolve face residual drips.
5. RC-4: grayscale low-res source → keep gray + disclosure.
6. Re-measure via .freebuff/ab_metrics.py per fix; suite must stay green; backend reload.

## Learnings
- (appended as fixes land)

## RESOLVED 2026-10-01 ~23:59 (all four RCs fixed + measured; suite 425/1/105)
- RC-1 + RC-3 fixed (PIXREVIVE_TM_FIELD, PIXREVIVE_HAIR_GUARD): stage-2 mask
  32.95% -> 9.9% (case9), hair HF re-injection now reaches strand gradients,
  pre-SR bilateral no longer flattens oriented texture.
- RC-2 fixed (PIXREVIVE_FACE_DRIP): in-face bright drips dissolved with
  landmark protection; verified eyes/mouth untouched by probe.
- RC-4 fixed (PIXREVIVE_GRAY_SKIP + sepia-drift background-wash guard): B stays
  grayscale (chroma 0.031 -> 0.0096, contour 0.99); A keeps sepia warmth —
  the SHIP6 pink field (DDColor replaced paper warmth b+31 -> +4, a=+8.4) is
  rolled back to luminance-preserved grayscale like Q-3.
- Verification: tests/test_rc_debug_fixes.py (26), full suite 425 passed /
  1 skipped / 105 subtests; visual A/B in frontend/public/comparison.html.
- Backend relaunched ready:true (cuda) with all fixes loaded.

## Learnings (session)
1. Morphological closing is a blanket-maker: k5x2 close (detector) + k7 close +
   k5 dilate (refiner) turned 8% speck coverage into a 20-35% region fill that
   LaMa re-synthesizes wholesale. Speck-granular masks are the anti-"puffed
   hair" instrument, not weaker detection.
2. structure_gate's mag floor (45) sat above real hair gradients (p50 26) on
   small canvases — protection that skips the texture it was built to protect.
   Floor 24 keeps the pure-noise contract (<0.0013 mean).
3. Q-3 single-hue guard is necessary but not sufficient: DDColor hallucinates
   multi-hue washes (pink field used several bins). The sepia-drift test
   (post-colorize background warmth vs the pre-colorization canvas) closes it.
4. Global noise scores under-rate small-canvas grain (B: 3.7 global vs obvious
   speckle field) — noise_profile.detected_types is the better evidence.
5. User downloads were byte-identical to server temp_results — always verify
   provenance (mtime + sha) before debugging "stale output" hypotheses.
