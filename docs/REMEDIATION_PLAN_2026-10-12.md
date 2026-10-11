# PixRevive Remediation Plan — 2026-10-12

**Source:** QA audit of this thread (teared.jpg output scored 3/10 restoration, 2/10 colorization) + unverified "shipped fixes" list. Evidence in this plan is tagged `OBSERVED` (measured in this workspace, this session) vs `INFERRED` (audit hypothesis, not yet proven).

---

## 0. STATE VERIFICATION (verified this session, live tree)

### 0.1 Status of the audit's "shipped" claims

| Claimed fix | Evidence it exists | How to confirm | If missing → action |
|--|--|--|--|
| L-only DDColor input | `grep PIXREVIVE_DDCOLOR_LONLY backend/ai/colorization/ddcolor_colorizer.py` → hit | `grep -n "LONLY" backend/ai/colorization/ddcolor_colorizer.py` shows both read + default-on branch | Confirm checkpoint logic; if the branch silently falls through when weights are missing, add a loud log line |
| Multi-scale flake detection | `grep PIXREVIVE_MS_FLAKES backend/ai/damage/enhanced_detector.py` → hit | Read `detect_white_flakes` multi-scale block; verify it *fuses* with existing pipeline masks, not replaces | PORT 1 |
| Ragged-border fallback | `grep PIXREVIVE_RAGGED_BORDER backend/ai/preprocessing/image_normalizer.py` → hit; `detect_ragged_borders` exists at line 57 | Synthetic 1932×1932 image with cream border strips → runs and fires correctly | PORT 1 |
| NLM damage-branch cap | `grep PIXREVIVE_NLM_CAP backend/ai/pipeline/stages/finish_stages.py` → hit | Synthetic high-σ image with `bg_damaged=True` → damage-branch h respects the cap | PORT 2 |
| Grain re-inject | `grep PIXREVIVE_GRAIN backend/ai/pipeline/stages/finish_stages.py` → hit | Inspect that it fires only on low-lapvar content zones and wins against NLM without watering flat regions | PORT 2 |
| Face/neck chroma nudger | `grep PIXREVIVE_FACE_NECK_FIX ...` → hit | Feed a face + neck with pre-measured mismatch, target Δab reduction verified | PORT 2 |
| Chroma validator | `grep PIXREVIVE_CHROMA ...` → hit | Force low-chroma stub → validator should WARN, not crash (still confirm the actual warning value surfaced in logs/metadata) | PORT 2 |

### 0.2 Git status of the working tree — CRITICAL

`git status --short` shows the four fix-bearing files as **UNCOMMITTED modifications**:
- `backend/ai/pipeline/stages/finish_stages.py`
- `backend/ai/pipeline/stages/color_stages.py`
- `backend/ai/preprocessing/image_normalizer.py`
- `backend/ai/pipeline/restoration_pipeline.py`
- plus ~40 additional uncommitted deletions/modifications elsewhere (`.freebuff/` probe scripts, `.gitignore`, `CHANGELOG.md`, `CODEBASE_MAP.md`, `.codex/skills/...` skill updates)

`git log -- backend/ai/colorization/ddcolor_colorizer.py` shows the last COMMIT touching that file is `c558e25` ("Initial baseline commit"), meaning **every fix represented in these four files is sitting uncommitted on the working tree**. Cleanup commit deleted some probe scripts.

**Action required before anything else in this plan:**
- **NOT** "revert everything" — there is *no* committed anchor anywhere except `c558e25`/`38d7de1`/`8ad9822` (Phase V0).
- Instead: **commit the working tree in two clean batches**:
  1. **Batch A (code):** the 4 fix files + `docs/REMEDIATION_PLAN_2026-10-12.md` (this doc)
  2. **Batch B (housekeeping):** `.gitignore`, `.codex/skills/...`, the deleted probe scripts, `CHANGELOG.md`, `CODEBASE_MAP.md`, `tests/` if modified
  - Review each batch with `git add -p`-style hunk review; do NOT `git add -A`.
  - Suggested messages: `Phase Q1 fix batch: LONLY DDColor + MS flakes + ragged border + NLM cap + grain reinject + J-9b nudger (all behind PIXREVIVE_* flags, default on)` and `Housekeeping: retire retired probe scripts, refresh codebase map`
  - Without this, a stray `git checkout .` loses ~a week of forensic work.

### 0.3 Flag/callout hygiene (see also §6 Risk register)

Seven new env flags (`PIXREVIVE_DDCOLOR_LONLY`, `_MS_FLAKES`, `_NLM_CAP`, `_RAGGED_BORDER`, `_GRAIN`, `_FACE_NECK_FIX`, `_CHROMA_SNAP_RAW`) now exist in three files. Two hardening actions (LOW priority, post-baseline):
- Single source of truth `backend/ai/config/flags.py` (ASSUMED path — if missing, create; do not guess an existing module) reading all `PIXREVIVE_*` keys once at import, caching the result, so defaults can be tuned in one file.
- `docs/FLAGS.md` table: flag | default | effect | rollback (one line each).

### 0.4 Test baseline

`pytest --collect-only` = **435 tests collected**. Baseline run must be executed on the working tree *as-is* (no code touched) and the pass/fail count recorded in Phase 0; after that, the suite is the harness for every gate in §4.

---

## 1. SEVERITY RANKING

### 1.1 Severity model

Weights chosen for this problem class — **(a) visual damage 0.35** (this is a consumer-facing restoration product; visible artifacts are what the user sees), **(b) frequency across fixtures 0.25** (a bug that hits every scan is worse than a one-off), **(c) blast radius 0.25** (downstream stages inherit upstream errors multiplicatively, not additively), **(d) identity/trust risk 0.15** (settlement matters for vintage portraits: a family will not tolerate a wrong face). Each issue scores 1–5 on each axis; Severity = Σ(weight × axis).

| # | Issue | Codes | (a) | (b) | (c) | (d) | Σ (Σ·5=severity 25-max) |
|--|--|--|--|--|--|--|--|
| 1 | Whole-frame mustard/olive C-CAST | C-CAST, C-INC, C-SEM | 5 | 3 | 5 | 2 | **4.25 → 21.3/25** |
| 2 | Large flake blobs survive → LaMa cream fill | R-DMG | 5 | 4 | 4 | 1 | **4.00 → 20.0/25** |
| 3 | Surviving flakes amplified by SR+sharpen | R-SR, R-SHP | 4 | 4 | 3 | 1 | **3.20 → 16.0/25** |
| 4 | Ragged cream borders survive | R-DMG | 4 | 4 | 3 | 0 | **3.00 → 15.0/25 issue 3** |
| 5 | Waxy skin / lost stubble/fabric | R-OVS | 4 | 4 | 3 | 1 | **3.15 → 15.8/25 (issue 4)** |
| 6 | Iris/lip chroma post-GFPGAN | C-INC, C-SEM | 3 | 3 | 2 | 2 | **2.45 → 12.3/25** |
| 7 | Chroma snapped to ragged L edges | C-BLEED | 3 | 3 | 2 | 0 | **2.10 → 10.5/25** |
| 8 | Hair scratch lines survive | R-DMG | 2 | 2 | 2 | 0 | **1.35 → 6.8/25** |
| 9 | Greyscale-skip / border-strip risks (image3, Lena) | OTHER | ? | 2 | 2 | 1 | unrated 1/25 until an output exists |
| 10 | Test infra gaps | — | 0 | 5 | 5 | 1 | 3.75/25 (infrastructure, not user-facing) |
| 11 | Unverified shipped-claim risk (0.2 above) | — | 0 | 4 | 4 | 2 | 3.5/25 (process risk) |

### 1.2 Pure severity order

1. C-CAST (21.3)
2. R-DMG flake blobs (20.0)
3. R-OVS waxy skin (15.8)
4. Ragged borders (15.0)
5. Flakes amplified by SR+sharpen (16.0 — ranks between 2 and 3 by score, but see dependency note)
6. Iris/lip chroma (12.3)
7. C-BLEED (10.5)
8. Hair scratches (6.8)
9. Greyscale/fixture risk (1.0)
10. Test infra + commit hygiene (process)

### 1.3 Dependency graph

- **H0 COMIT-HYGIENE → everything:** the current fixes are uncommitted; Phase 0 must first pin a clean commit hash so every later diff is meaningful. `COMMIT HYGIENE` gates ALL fix cards.
- **FIX-CAST (DDColor input prep) → FIX-FLAKE baked-in stain inherited:** the cast strictly comes from surviving flakes/LaMa fill — but the cast fix also *changes what DDColor sees*, so it must come (a) after the flake fix so flake removal is measured on a system whose colorizer already produces honest chroma, AND (b) before any SR/sharpen stage fix, since SR amplifies whatever chroma exists at the time.
- **FIX-FLAKE + FIX-BORDER both precede FIX-AMPLIFY (SR amplification)**: SR will amplify the residual flakes/borders from H2/H3-H4, so both upstream fixes must be in place BEFORE measuring SR amplification; otherwise FIX-AMPLIFY fails its acceptance test falsely.
- **FIX-WAXY (NLM cap + grain reinject) can run in either order relative to flake/border fixes** — they touch different stages, but grain re-injection competes with the flake detector for what "flat" means. Safer sequence: grain re-inject AFTER flake fix, since adequate flake removal means fewer "flat" false-positives to over-inject.
- **FIX-IRIS (face/neck chroma nudger) depends on FIX-CAST** (the chroma distribution DDColor produces changes what nudging means).
- **FIX-BLEED depends on FIX-CAST + FIX-FLAKE** — the guided-filter snapping problem is confirmed on *damaged* luminance; after flake removal the guide changes.
- **GREY-SKIP / fixture-specific (image3 tear, Lena, dark vignette) are LAST** — they depend on all core fixes for any measurement to be meaningful, and their acceptance requires the new border detector + colorizer skip logic to be already in place.

### 1.4 FINAL EXECUTION ORDER

| Order | Fix | Deviation from pure severity … and why |
|--|--|--|
| 0 | **COMMIT HYGIENE** — 2-batch commit (§0.2) | Process prerequisite, unranked |
| P0 | **Baseline & harness** (§2) | All gates depend on a recorded baseline |
| 1 | **FIX 1 — Flake-blob detection** (R-DMG, sev 20.0) | promoted above C-CAST: C-CAST is *caused by* surviving flakes; fixing flakes first removes the largest confound from the cast measurement |
| 2 | **FIX 2 — Ragged-border crop** (sev 15.0) | pairs with FIX 1: both are pre-inpaint gates; measuring either in isolation leaves the other as a confound |
| 3 | **FIX 3 — C-CAST root cause + cast validator** (sev 21.3) | demoted from rank 1 only because its own acceptance test requires a de-flaked, de-borderd canvas; colase colorizer changes otherwise confounded |
| 4 | **FIX 4 — Waxy skin (NLM cap + grain re-inject)** (sev 15.8) | after flake: grain re-injection must not paint over the flakes we just removed |
| 5 | **FIX 5 — Surviving-flake SR/sharpen amplification** (sev 16.0) | depends on FIX 1–3: only then is a clean canvas available to measure SR amplification honestly |
| 6 | **FIX 6 — Iris/lip chroma post-GFPGAN** (sev 12.3) | depends on FIX 3 (the colour DDColor produces changes what the nudger should do) |
| 7 | **FIX 7 — Chroma snapping to ragged edges** (sev 10.5) | depends on FIX 1+3 — the 'ragged guide' is largely gone after flake/border fixes; re-measure before touching the guided filter again |
| 8 | **FIX 8 — Hair scratch lines** (sev 6.8) | lowest visual severity; may be subsumed by FIX 1 if the multi-scale detector catches it |
| 9 | **FIX 9 — Special-case plans (image3 tear mask, image1 vignette, grey-skip routing, halftone pre-step)** (§5) | fixture-class gates; require the full matrix to be green first |

---

## 2. PHASE 0 — BASELINE & TEST HARNESS

### 2.1 Fixture matrix

| Fixture | Role | Must-fix defects | Must-not-fire on |
|--|--|--|--|
| `teared.jpg` (1932-input-class) | Primary regression target | flake blobs, borders, cast, waxiness, iris/lip, scratch lines | — |
| `image1.jpeg` | fleck field | dust flecks (hair mass), NOT the dark soft vignette | border crop on dark gradient; denoise-first routing (σ≈0) |
| `image3.jpg.jpeg` | hardest inpaint | facial tear (thin mask), collar cracks, mould flecks | nose/mouth geometry rebuild; over-wide mask (LaMa hallucinates) |
| `SigShrink-Lena` figure | border/grey-skip fixture | baked white right-edge strip | colorizer (grayscale skip), denoise-first (already denoised) |
| `case1_noisy_lena.jpg` (ASSUMED — in dataset per codebase map; verify presence) | speckle-path fixture | genuine speckle field | — |
| **NEW: `fixture_clean_photo.png`** | false-positive control | (none) | any mask, any border crop, any grain re-inject visible in flat zones |
| **NEW: `fixture_colour_photo.png`** | colorizer-should-skip or no-recolor control | (none — already colour) | any recolorization of a well-coloured photo; any chroma抹 over-boost |

Both new fixtures are synthesized (small, scripted) — they exist to catch the *pipeline firing where it must not*; no existing fixture currently covers this.

### 2.2 Run-harness script (one command, dumps everything)

**File:** `PixRevive/scripts/run_fixture_matrix.py` (ASSUMED directory exists per repo layout — verify; if not, create).

Behaviour:
- CLI: `python scripts/run_fixture_matrix.py --fixtures teared,image1,image3,lena_figure --outdir debug_output/matrix_<date>/`
- For each fixture, runs `RestorationPipeline.process()` with all current env flags left as defaults, and copies out of `debug_output/`:
  - `00_input.png`
  - `02_after_damage_cluster_smooth.png` (mask overlay available separately)
  - `03_after_inpaint.png`
  - `05_after_denoise.png`
  - `06_after_colorize.png` + a second dump of the RAW DDColor output before ColorDiffusion (see §2.4 note)
  - `07_after_face_restore.png`
  - `final.png`
- Also writes per fixture `meta_<fixture>.json`: degradation analysis (`noise_score`, `color_type`, `damage_percentage`, `content_sigma`), validator findings (`chroma_validator`, `face_neck_chroma_dev`, `output_verification`), stage timings, and the damage-mask coverage %.

**If `DebugLogger` does not already dump a RAW DDColor (pre-ColorDiffusion) snapshot**, extend `color_stages.py` to call `DebugLogger.save_stage(context, "06a_raw_ddcolor")` immediately after `_blend_chroma` before ColorDiffusion runs, behind flag `PIXREVIVE_DUMP_RAW_DDCOLOR` (default on, rollback = flag 0, zero behavioural change). This is Phase-0-only instrumentation, never pruned.

### 2.3 Metrics script

**File:** `PixRevive/scripts/compute_fixture_metrics.py`

For each fixture output, computes and writes `metrics_<fixture>.json`:

| Metric | Definition | ROI |
|--|--|--|
| cast | mean a*, mean b*, hue angle atan2(b,a) of skin ROI | [225,255,523,591] (0-1000 normalized on 1932-equivalent) |
| inter-iris Δab | \|ab_left − ab_right\| L2 | left [316,440,352,507], right [321,600,352,668] |
| lip-vs-cheek a* ↓ | cheek a* minus lip a* | lip [484,487,523,621], cheek ring around it |
| flake-blob count | connected components of (L > local-mean + 22) with area > 100px² | whole frame + 3 had-bug ROIs |
| mask coverage % | damage-mask coverage over face flake ROI [383,336,445,398] | same |
| lapvar retention | output-face lapvar / input-face lapvar | face ROI |
| flat-region σ(MAD) | MAD σ on flat corner of bg | [0,0,80,80] |
| near-neutral share | % pixels chroma < 6 | whole frame |
| colourfulness | Hasler–Süsstrunk metric | whole frame |
| halo measure | gradient overshoot at 20 randomly sampled strong edges (max − (neighbour mean)) | sampled near silhouette |
| stereo face similarity | ArcFace (or CLIP if ArcFace weights unavailable at runtime — substitute, note the limitation) | face crop |

The script must be deterministic (fixed RNG seed for any sampling), must run CPU-only, and must skip (record `null`) any metric whose inputs are missing rather than crash.

### 2.4 Baseline recording

Run `compute_fixture_metrics.py` against the **current** outputs (teared output from this thread, if it is still on disk under `temp_results/`; else re-run the pipeline once on teared.jpg only, dump intermediates + metrics as BASELINE-A). Results go in `debug_output/matrix_baseline/metrics_*.json` and numbers are transcribed into §4 gates to have numeric anchors. Do NOT proceed to FIX 1 until BASELINE-A exists.

### 2.5 Test-coverage additions before FIX 1

1. **`tests/test_detect_ragged_borders.py`** — port the synthetic cases used in this session (cream L/R strips on 600×600 mid-tone portrait → crop fires; clean image → inert; interior-flakes-only image → inert) into a committed pytest file with a `PIXREVIVE_RAGGED_BORDER=1` and `=0` case each.
2. **`tests/test_synthetic_flake_field.py`** — synthesize a 512² image with flat background + 12 bright blobs (30–400 px) → asserts the multi-scale detector catches ≥ 70% before LaMa.
3. **`tests/test_cast_inheritance.py`** — synthesize a mildly-sepia L-channel with a *stained* 3-channel input, run DDColor locally with `PIXREVIVE_DDCOLOR_LONLY=1` and `=0`, assert with `=1` that output background b* drops by ≥ 8 units vs `=0`.
4. **`tests/test_nlm_damage_branch_cap.py`** — high-σ fixture with `bg_damaged=True`, assert damage-branch h respects `_nlm_cap`.
5. **`tests/test_grain_reinject_scope.py`** — assert grain injection is bounded by `_cmask` and does not fire in the scanner-margin regions of test margin-robust fixtures.

---

## 3. FIX CARDS (in FINAL EXECUTION ORDER)

---

### **FIX 1 — Multi-scale flake detection tuned on street-level flake loads** | R-DMG | Severity 20.0/25 | teared (worst), image1, image3

1. **Problem:** OBSERVED (audit, teared output): 19 surviving cream/flake blob regions; worst [16,787,274,921] and [430,792,569,911] (background), [337,383,399,445] (face), [569,383,688,518] (neck/collar), [688,207,771,373] (shirt); face flake ROI [383,336,445,398] visibly bright cream in the output. INFERRED: `detect_white_flakes` single-scale local-contrast with a `max_flake_size` upper cap misses the 40–400px class.

2. **Hypotheses:**
   - PRIMARY: detection miss — the multi-scale path (`PIXREVIVE_MS_FLAKES`) exists in the tree but either (i) is not actually reachable in `detect_white_flakes` (wired to a dead code path), (ii) fires but its masks are not OR-ed into the fracture mask that feeds LaMa, or (iii) fires but its size band / threshold parameters are mis-tuned for 40–400px blobs where halftone dot texture also lives.
   - ALT-1: detection is fine, mask feeds LaMa, but LaMa chooses a cream fill that matches the stained paper (paper stain prior).
   - ALT-2: detection correct but Stage 2b/5/9c partially re-lightens/re-injects the very pixels LaMa just cleaned.

3. **FALSIFICATION TEST (before code):** run the harness (§2.2) on teared.jpg ONCE with default flags; inspect `meta_teared.json`’s mask-coverage % on ROI [383,336,445,398] and the `02_after_damage_cluster_smooth` mask overlay.
   - **TRUE (detection miss confirmed):** mask coverage in that ROI < 5% while pixels are visibly bright cream ⇒ proceed with re-tune/enlargement of the multi-scale kernel set + size band.
   - **FALSE-ALT1 (mask present ≥ 25%, LaMa still cream):** shift work to the fill-prior: pass a neutral mid-grey synthetic patch prompt to LaMa, OR post-process the inpaint fill to pull the fill colour from the *surrounding skin/paper* mean rather than the nearest flake edge. The cast fix (FIX 3) may also resolve part of this because the fill inherits the stain.
   - **FALSE-ALT2 (mask present AND `03_after_inpaint` is clean, but the final output shows the blob again):** the damage was *re-introduced* by a later stage; grep the last-stage debug dumps (`05_after_denoise`, `06_after_colorize`, `07_after_face_restore`) and diff each against `03_after_inpaint` to find which stage re-paints the blob.

4. **Proposed change:** `backend/ai/damage/enhanced_detector.py` — in `detect_white_flakes`, ensure:
   - multi-scale tophat kernels are engaged by default (flag `PIXREVIVE_MS_FLAKES`, **default 1**, exists);
   - the upper blob size band admits 40–400px blobs in a 1000×1000-comparable frame; at 1932 long-side, the equivalent band is ~90–1600px² (ASSUMED scaling — parameter is in the code; verify constant);
   - the multi-scale results are **OR-fused** into the main `flake_uint8` mask **without replacing** the single-scale path (behaviour change gated by flag, default on).
   If the flag currently connects to unreachable code the change may be as small as re-wiring one `if` branch. Verify in the falsification phase before assuming.

5. **Alternatives considered / rejected:** none structural needed.

6. **Verification tests to add:**
   - `test_synthetic_flake_field_512.py` — 30–400px synthetic blobs in flat bg → assert ≥70% are masked pre-LaMa (threshold justified: below this, we know the detector did NOT actually fix the fixture class)
   - `test_flake_mask_coverage_teared.py` — fixture-driven: parse `meta_teared.json` metric `mask_coverage_face_flake_ROI`, assert ≥ 0.25.

7. **Acceptance criteria:**
   - flake-blob count (whole frame, metric table §2.3) post-pipeline **= 0** for the 5 named ROIs on teared.jpg. Justification: 100% of these pixels are emulsion loss, a human restorer removes all of them.
   - mask coverage ≥ 25% in every pre-named face/neck/shirt flake ROI.
   - `fixture_clean_photo.png` — flake-blob count must remain **0 before and after** (must-not-fire criterion).
   - No regression in `image3.jpg.jpeg` — its collar cracks may partially be caught by the expanded detector; if it catches > 60% of the *healthy* fabric weave pixels, that is a false-positive retreat (assert via a must-not-change metric: `mask_coverage` on healthy fabric ROI [531,324,984,988] < 5%).

8. **Side-effect risks:** over-detection on halftone dots (teared is a printed replica — the dot lattice is bright and small); over-detection on the image3 healthy fabric; over-detection on the image1 dark vignette edges. Mitigation: the multi-scale fusion keeps the existing single-scale path's gates; the **halo/threshold** of the tophat bounding detuning is exactly the falsifiable-test knob. The harness (§2.2) + must-not-change fixtures are the detection net.

9. **Rollback:** `PIXREVIVE_MS_FLAKES=0` (one-line env change, no code revert).

10. **Dependencies:** after COMMIT HYGIENE + Phase 0 harness; must precede FIX 2's judgement, FIX 3's cast measurement, and FIX 5's amplification measurement.

11. **Effort:** S–M, ~6–12h including falsification test + tests. **Confidence:** high that the flag wiring/threshold tuning fixes the audit's observed blobs *if* the primary hypothesis is right; medium overall (LaMa-fill alternative could dominate).

12. **Definition of done:** all named ROIs' blob count = 0 post-pipeline; synthetic 512² flake-field test green; no must-not-change fixture regressed; gate (§4) passes; one commit.

---

### **FIX 2 — Ragged-border fallback committed + param-verified on all 3 margin classes** | R-DMG/R-GEO | Severity 15.0/25 | teared, image1 (must NOT fire), image3 (marginal), Lena figure (must crop white strip)

1. **Problem:** OBSERVED: teared output retained ragged cream strips [0,0,1000,75] and [0,919,1000,1000] (0-1000 normalized). INFERRED: uniform-margin detector (σ≤6 gate) cannot see them; the `detect_ragged_borders` fallback exists in the working tree but has (i) no committed pytest coverage, (ii) no verification against the three real margin classes in the fixture set (cream ragged, soft dark vignette, baked white frame strip).

2. **Hypotheses:**
   - PRIMARY: the fallback works for cream margins but not yet verified for image1's soft dark vignette (should NOT fire) or Lena's baked white strip (should fire — but the **uniform** detector, not the ragged one, should catch that first).
   - ALT: the coverage threshold (≥55%) or the bright+neutral pixel rule wrongly-classifies the image3 background mould texture (which is bright but *not* low-saturation) as a border — this would wrongly crop content.

3. **FALSIFICATION TEST:** run `detect_ragged_borders` directly on each fixture INPUT at native resolution with default flag; also run the original `detect_and_crop_borders`.
   Expected TRUE/fallback-correct:
   - teared.jpg: strips found at expected positions, crop fires.
   - image1.jpeg: **inert** (its vignette is dark, not bright).
   - image3.jpg.jpeg: inert or minimal (mould flecks must not read as a full-height strip).
   - SigShrink-Lena: the ORIGINAL uniform detector fires (baked white strip is uniform); ragged fallback NOT needed. Verify the ordering in `restoration_pipeline.py`: uniform detector runs first, ragged only when uniform misses.
   If any fixture fails its expectation, then the parameter set needs tuning: `brightness_floor=180` (for cream vs dark separation), `min_strip_frac=0.55`, `margin_frac=0.18`.

4. **Proposed change:** `backend/ai/preprocessing/image_normalizer.py` — **commit it** (already in tree), then:
   - add a cap: `PIXREVIVE_RAGGED_BORDER_MAX_CROP_FRACTION = 0.45` — never crop more than 45% of the frame area (safety valve, as a hard-coded module constant, not an env flag — it is a guard, not a behavioural knob);
   - behavioural env flag `PIXREVIVE_RAGGED_BORDER` (default 1, exists) with one-line rollback.

5. **Alternatives considered / rejected:**
   - *Write a new detector from scratch*: rejected — the working-tree implementation was already synthetic-tested and behaves correctly on the three primary cases (from this session's smoke test results).
   - *Border reconstruction pass (inpaint the border rather than crop)*: rejected for now — crop is safer and cheaper; reconstruction is a later feature, not a bug fix.

6. **Tests to add:** the three synthetic cases from this session's smoke test, committed as `tests/test_detect_ragged_borders.py` (see §2.5 item 1), plus the full-fixture pre-flight check in §3-FIX2-callout-3.

7. **Acceptance criteria:**
   - teared.jpg final output: residual cream margin band wider than 2% of frame width = **0** on each side.
   - image1.jpeg: `border_cropped=False` or crop bbox y_min < 40 (justification: the dark vignette is a genuine exposure gradient, part of the photo; cropping it would be worse than input).
   - image3.jpg.jpeg: `border_cropped=False` (no full-height strips there) — must-not-change criterion.
   - Lena figure: uniform detector fires; a uniform white strip of width ≥ 4% of frame is cropped.

8. **Side-effect risks:** over-crop on images with genuine bright card-mounted subjects; the frame-area cap (§4 of the proposed change) bounds the damage. Also risk: the ragged border crop runs BEFORE the flat-field correction (per fixed pipeline order); if the crop changes the geometry, the downstream flat-field fit shifts. Mitigation: log `border_bbox` and make the crop idempotent (a second run on the cropped image must not crop further) and verify in the harness.

9. **Rollback:** `PIXREVIVE_RAGGED_BORDER=0` — the pipeline reverts to uniform detector + original behaviour.

10. **Dependencies:** after COMMIT HYGIENE; MUST precede FIX 3 cast-judgement (borders surviving confuse the chroma stats); independent of FIX 1 (can run in either order with FIX 1, but fix should resolve amicably).

11. **Effort:** S, 2–4h (tests + param verification + cap). **Confidence:** high (function verified live this session).

12. **Definition of done:** pre-flight check on all 4 fixtures, each matching its expectation; committed pytest file green; harness run green; gate passes; one commit.

---

### **FIX 3 — C-CAST root cause: L-only DDColor input + fail-loud validator + limiter floor** | C-CAST, C-INC, C-SEM | Severity 21.3/25 (top) | teared; touches colour of every sepia/grey fixture

1. **Problem:** OBSERVED (audit, measured on teared output): output background b*≈+48..+53, skin b*≈+40..52 vs input b*≈+19; whole-frame single-hue mustard-olive; no lip chroma; mismatched irises. INFERRED (primary): `colorize()` merges surviving a/b (`cv2.merge([l_smooth, a_in, b_in])`) from the stained canvas into the model input, so DDColor treats the stain as real chroma; Chroma Magnitude Limiter (cap 58, floor 0.50) cannot re-centre. ALT: DDColor misreads the halftone dot lattice as photo texture (the input is a printed replica).

2. **Hypotheses:**
   - PRIMARY: stain inheritance through the merged a/b path.
   - ALT-1: DDColor genuinely predicts yellow for halftone prints.
   - ALT-2: a later stage (ColorDiffusion guided filter, launcher-rebalanced `rebalance_dye_fade`, or color_refiner) neutralises/shifts the model's chroma after prediction.

3. **FALSIFICATION TEST:** DDColor in isolation:
   - Load the model, run it twice on the same teared.jpg face crop [176,357,595,699]: (i) the current 3-channel-prepared input, (ii) an explicit L-only input (a=b=128).
   - Dump both raw outputs + mean a/b per output.
   - **TRUE (stain inheritance):** (ii) shows differentiated chroma — skin b*∈[12,32] and bg b*≤15 — while (i) shows uniformly high b*≈35–50 ⇒ the LONLY flag path is the fix; verify the working-tree implementation reaches the same numbers.
   - **FALSE (both weak):** the model itself fails on this texture; inspect checkpoint integrity (`ddcolor_modelscope.pth`), `input_size`, device routing, and BGR/RGB ordering; NOT an input-prep bug.
   - **FALSE-ALT2:** compare timestamps/logs between `06a_raw_ddcolor` and `06_after_colorize` intermediates — if the model output has good chroma and the FINAL is washed/olive, the neutraliser lives between them (ColorDiffusion / color_refiner); the falsification of FIX 7 (BLEED) overlaps — split as FIX 7's job.

4. **Proposed change:** `backend/ai/colorization/ddcolor_colorizer.py` —
   1. Ensure the L-only path engages by default: `PIXREVIVE_DDCOLOR_LONLY` (default 1, exists in tree); verify it also applies on the **deferred face-restore pass** and grayscale-skip paths, not only the main colorize call.
   2. Chroma validator (`_chroma_validator`, exists at line 207) must fail **loudly** — a WARN log + `colorization_wash_kind` metadata flag, but pipeline continues (do NOT fail the task on an aesthetic risk).
   3. Chroma Magnitude Limiter floor: 0.50 → **0.62** (existence of the floor verified in code; change gated behind `PIXREVIVE_CHROMA_FLOOR` (default 0.62) — one-line rollback).
   4. (Piggy-back, NO) — do NOT stack any ColorDiffusion/refiner change here. One commit, one behavioural change.

5. **Alternatives considered / rejected:** *rewrite the colorizer*: rejected; *raise floor to 0.8*: rejected for now, over-correction risk to grey outputs — tune on fixture data only if 0.62 is insufficient.

6. **Tests to add:** `tests/test_cast_inheritance.py` (§2.5 item 3).

7. **Acceptance criteria (teared.jpg):**
   - skin ROI a* ∈ [6,20] AND b* ∈ [12,32] (warm skin, not sunburnt, not grey).
   - face-vs-background b* separation ≥ 10 units.
   - near-neutral pixel share ≥ 15% (bg paper must be near-neutral; INFERRED welcome range 30–60%; threshold tuned on real data if needed).
   - colourfulness metric ≥ 0.8× of the audit output (colour must NOT collapse).
   - `fixture_colour_photo.png` must-not-change: mean a*/b* shift ≤ 3 units after (colour photos must not be re-tinted).
   - SigShrink-Lena (grayscale): colorizer skipped (`_gray_skip_warranted`); a near-black-white output.
   - image1/image3 (greyscale): chroma stays neutral-bg; skin ROI only mild warm (b* ≤ 20) — no cast appearing where none existed.

8. **Side-effect risks:** raising the limiter floor and L-only input together can over-suppress colour, producing grey-green outputs (the opposite failure). The validator + fixture matrix catches both directions. Also risks: the deferred-face-restore pass re-colorsing ignoring LONLY (verify specifically).

9. **Rollback:** `PIXREVIVE_DDCOLOR_LONLY=0` + `PIXREVIVE_CHROMA_FLOOR=0.50` (two env values, or the flag docs' single-line rollback note).

10. **Dependencies:** after FIX 1+2 (flake-free canvas to measure honest chroma); must precede FIX 4, 5, 6, 7.

11. **Effort:** M, 6–10h including falsification + tests. **Confidence:** medium-high (the mechanism is plausible and matches code reading; halftone-confound alternative not ruled out).

12. **Definition of done:** falsification test artifacts archived; acceptance metrics green on teared + all must-not-change fixtures; validator logs fire on synthetic cast-stub; one commit.

---

### **FIX 4 — Waxy skin: NLM damage-branch cap + grain re-injection** | R-OVS | 15.8/25 | teared primary; any portrait fixture

1. **Problem:** OBSERVED: face skin waxy/plastic [176,357,595,699]; no stubble/pores in [487,492,517,611]; shirt fabric melted [621,109,1000,926]. INFERRED: damage-branch NLM `h_bg = max(11.0, 1.4σ)` unbounded when σ is polluted by damage; grain floor flattened pre-SR; GFPGAN lower-LQ paste dominates skin.

2. **Hypotheses:**
   - PRIMARY: unbounded damage-branch h.
   - ALT-1: GFPGAN paste alpha dominates texture fields (shipped face-aware damping PIXREVIVE_FACE_SHARPEN may be insufficient at lower LQ).
   - ALT-2: pre-SR anti-aliasing (bilateral, hair guard) rounding off fabric weave.

3. **FALSIFICATION TEST:** run harness on teared.jpg; compute **Laplacian variance retention** (output face ROI / input face ROI). Per BASELINE-A:
   - **TRUE (h exceeded):** retention < 0.45 while `damage%` > 3 ⇒ cap is needed and the shipped cap(14) raises retention when violated (retune, not revert).
   - **FALSE (retention ≥ 0.45 but waxiness still observed):** the loss is not from Stage 5; GFPGAN or SR. Diff `05_after_denoise` vs `07_after_face_restore` lapvar on the face ROI to isolate the stage that flattens it.

4. **Proposed change:** `backend/ai/pipeline/stages/finish_stages.py` —
   - ensure `PIXREVIVE_NLM_CAP=14.0` (default on, exists) truly caps the damage branch (verified in tree);
   - grain re-injection `PIXREVIVE_GRAIN` (default on, exists) — verify its injection amplitude produces Lap-var gain of 5–15% only, in flat content zones.
   No additional change proposed here unless falsification shows the cap is NOT engaged/insufficient — in that case raise the cap logic from 14 to a measured value; do NOT widen beyond that without data.

5. **Alternatives considered / rejected:** globally lowering h (breaks the σ-15 noise contract, previously known bad). Texture re-synthesis (diffusion grain) — future feature, not this fix.

6. **Tests to add:** `tests/test_nlm_damage_branch_cap.py` + `test_grain_reinject_scope.py` (§2.5 items 4, 5).

7. **Acceptance criteria (teared):**
   - Lap-var retention (face ROI) ≥ 0.45× (justified: input face lapvar 8205; less than half of that reads waxy on comparable print scans).
   - flat-bg σ(MAD) post-pipeline ∈ [1.5, 4.0] (not snow, not haze).
   - image1 (no damage): retention ≥ 0.9× (denoiser must be near-inert when damage < 3%); σ(MAD) after ≤ 3.5 (must not ADD noise to a clean scan).
   - `case1_noisy_lena` σ(MAD) after ≤ 8.0 (the speckle-path fixture — grain re-inject must not fire on the speckle-removed zone).
   - `fixture_clean_photo` — lapvar retention ≥ 0.8×.

8. **Side-effect risks:** grain injection on a subject's face may look like new noise — bounded by `_cmask`; the amplitude is tuned to 2.2 LAB units at 1932px; if real-skin texture reads unnatural, reduce to 1.5 (tune on real data only if flagged).

9. **Rollback:** `PIXREVIVE_NLM_CAP=0` (uncapped legacy) + `PIXREVIVE_GRAIN=0` (no injection).

10. **Dependencies:** after FIX 1 (grain must not paint over surviving flakes); independent of FIX 2/3 ordering; must precede FIX 5 (SR amplification measurement needs the flattening honoureded).

11. **Effort:** M–L, 8–14h (cap verification + grain amplitude tuning + tests). **Confidence:** medium-high.

12. **Definition of done:** all lapvar/σ thresholds green on the matrix; grain visible but non-intrusive in a 1:1 zoom human check; one commit.

---

### **FIX 5 — Suppress surviving-flake SR/sharpen amplification** | R-SR, R-SHP | 16.0/25 | teared (primary), image1 (dust flecks), image3 (mould flecks)

1. **Problem:** OBSERVED: surviving flakes amplified into crisp artifacts (e.g. [688,207,771,373], [880,556,916,595] on teared output), SR halos on flake borders. INFERRED: SR + LAB sharpen run at the end, amplifying residue the earlier stages missed; also the sharpen stage ignores the mask.

2. **Hypotheses:**
   - PRIMARY: absent post-inpaint flake sweep → SR amplifies residue the earlier stages missed.
   - ALT: SR's Real-ESRGAN model inherently hallucinates texture at surviving flake boundaries, even with a clean inpaint (fill colour vs ink boundary sharpness).

3. **FALSIFICATION TEST:** take the FIX 1..4-clean teared run:
   - **TRUE:** flake-blob count in the final output returned to non-zero => a sweep pass between colorize/SR is warranted; the existing `PIXREVIVE_SPECK_SWEEP` (Stage 10 pre-sharpen, exists in tree) may not be engaged/adequate for the 40–400px class.
   - **FALSE:** post-fix the blob count is 0 => FIX 1–4 did the whole job, and FIX 5's only remaining work is the "ssweep engage/adequacy" audit; shrink scope accordingly.

4. **Proposed change:** `backend/ai/pipeline/stages/finish_stages.py` Stage 10 —
   - raise `PIXREVIVE_SPECK_SWEEP`'s speck cap to include the 40–400px *blob class* (currently keyed for 1–24px specks per its own guard); size band genuinely from fixture data: measure one blob's px area at 1932 on the teared input (ASSUMED parameter name `speck_cap` — verify in code).
   - Optionally (if POST-SR blob remnants occur): add a chroma-and-brightness guard: a bright blob on a dark garment with high brightness + low local texture = artifact → adaptive median/Telea neutralise, gated on `PIXREVIVE_SR_FLAKE_SWEEP` (default 0 initially, then promote to 1 once validated).

5. **Alternatives rejected:** mask-aware sharpening (a mask threading через SR is not trivially forward-feasible); skipping SR entirely on damaged canvases (regress output size).

6. **Tests to add:** post-SR blob count version of `test_synthetic_flake_field`, extended to T-intercept after SR (fixture-driven).

7. **Acceptance criteria (teared):**
   - flake-blob count in final output = 0 (already required by FIX 1, but re-check after this fix introduces the sweep).
   - Lap-var retention does not drop (sweep must not eat fabric weave) — ≥ 0.45× maintained.
   - halo measure ≤ baseline (sharpen must not overshoot more than BASELINE-A).

8. **Side-effect risks:** the sweep eating real specular/fabric highlights — gated by the low-local-texture requirement and by the "must-not-fire on `fixture_clean_photo`" criterion.

9. **Rollback:** `PIXREVIVE_SPECK_SWEEP=0` (existing flag); the new sweep has its own `PIXREVIVE_SR_FLAKE_SWEEP=0` default-off rollout gate.

10. **Dependencies:** after FIX 1–4; enables honest measurement of SR artifacts.

11. **Effort:** S–M, 4–8h. **Confidence:** medium (depends on whether FIX 1–4 fully eliminate blobs before SR).

12. **Definition of done:** post-SR blob count = 0 on teared/image1/image3; halo metric not worse than baseline; one commit.

---

### **FIX 6 — Iris/lip chroma consistency post-GFPGAN** | C-INC, C-SEM | 12.3/25 | teared

1. **Problem:** OBSERVED: left iris [316,440,352,507] brown vs right iris [321,600,352,668] grey-green; lips [484,487,523,621] achromatic. INFERRED: GFPGAN RGB paste discards Stage-6 chroma where paste alpha < 1; shipped J-9b face↔neck nudge is global and may not fix eyes/lips specifically.

2. **Hypotheses:**
   - PRIMARY: paste chroma discarding — GFPGAN's RGB paste replaces both luminance AND chrominance.
   - ALT: Stage 6 colours asymmetrically left/right (the colour model generating brown left, grey-green right).

3. **FALSIFICATION TEST:** inspect the intermediate diffs: `06_after_colorize` (pre-GFPGAN) vs `07_after_face_restore` (post-paste) on the two iris ROIs:
   - **TRUE (paste discarding):** `06` has symmetric-brown irises; `07` shows one grey → the nudge must run POST-paste and target the eye band. (This also means the fix's change should be at the end of Stage 7, not Stage 6.)
   - **FALSE-ALT (asymmetric in `06`):** Stage 6 (or DDColor itself) is generating asymmetry; the J-7 smoothed-guide change in ColorDiffusion, plus the nudger, both target it — but the falsification is on the intermediate, not the nudger.

4. **Proposed change:** `backend/ai/pipeline/stages/finish_stages.py`, Stage 7 post-paste block — extend J-9b
   - measure per-region chroma means in 3 bands: left iris ROI, right iris ROI, lip ROI;
   - if the two iris Δab > 8, gently pull the OFFENDING iris's a/b toward the mean of the two (conservative: 50/50 mean shift, half applied per step, capped at 6 LAB units per step, clamp within the pupil/iris connected component only).
   - if lip a* < surrounding cheek a* + 10, raise the lip's a* by a capped +6 toward the cheek+10 target, within a morphological lips mask (dilated lip landmark hull) — gated `PIXREVIVE_FACE_NECK_FIX` (default 1, exists) so the iris/lip logic joins the same flag (or a sub-flag `PIXREVIVE_FACE_NECK_FIX_LIPSLIKE` for finer rollback granularity).

5. **Alternatives rejected:** re-running GFPGAN with a colorized input (double-beta risk); using the semantic validator to re-run only the eye band (Phase V0's rule A1) — defer to V1 where the "re-imagine" prompt exists; here we nudge rather than re-generate.

6. **Tests to add:** `tests/test_face_restorer.py`-style fixture with two miscoloured irises → assert post-fix Δab ≤ 8 and unchanged summary on a symmetric stub.

7. **Acceptance criteria (teared):**
   - inter-iris Δab ≤ 8 (justification: symmetric eye hue is the visual signature; anything above this reads as a mismatch at normal zoom).
   - lip a* ≥ cheek a* + 10 (lip colour must be visible above surrounding skin).
   - unchanged behaviour on `fixture_colour_photo` (must-not-change).
   - identity: ArcFace cosine similarity input vs output face ≥ 0.85 — SUBSTITUTED (if ArcFace weights unavailable at runtime): CLIP face-crop embedding, plus a human check; state the reduction in claim.

8. **Side-effect risks:** painting over real eye colour (e.g., one blue + one brown eye — heterochromia exists in real people!) — gate the fix on a chroma-difference threshold that is LARGE (> 15 pre-fix) before firing, so random JPEG chroma noise does not trigger.

9. **Rollback:** `PIXREVIVE_FACE_NECK_FIX=0`.

10. **Dependencies:** after FIX 3 (the chroma it normalises doesn't shift under it).

11. **Effort:** M, 4–8h. **Confidence:** medium.

12. **Definition of done:** both iris ROIs Δab ≤ 8 in the harness run; lip a* delta green; identity check done (sim or human confirmation); one commit.

---

### **FIX 7 — Colour bleeding / guided filter guide (the "ragged edge chroma snapping")** | C-BLEED | 10.5/25 | teared; less on image3/image1

1. **Problem:** OBSERVED: posterised 'cut-paper' fringe on hair outline [31,295,362,725]. INFERRED: ColorDiffusion guided filter in Stage 6 uses the *raw* L (damage-ragged) as the guide, snapping chroma to those edges.

2. **Hypotheses:**
   - PRIMARY: rawguide snapping — shipped fix (smoothed-L guide, larger radius/eps, flag `PIXREVIVE_CHROMA_SNAP_RAW=0` default) exists in tree; verify it engages.
   - ALT: the fringe is actually SR aliasing (R-SR class) not chroma snapping — check whether the posterised edge still appears if Stage 6 is disabled entirely.

3. **FALSIFICATION TEST:** harness run with `PIXREVIVE_CHROMA_SNAP_RAW=1` (old behaviour, force legacy guide) vs `=0` (new); compare the L-gradient1330 complement (i.e., `104-CIE hue histogram` cost) in the ROI [31,295,362,725]:
   - TRUE (guide was the cause): new guide shows a substantially smoother hue-transition distribution in that ROI.
   - FALSE: no difference => the fringe is an SR/sharpen artifact; reroute to FIX 5's remit (check for sharp-edge ringing) and STOP changing Stage 6.

4. **Proposed change:** `backend/ai/pipeline/stages/color_stages.py` — commit the already-present smoothed-guide change (`PIXREVIVE_CHROMA_SNAP_RAW` default 0/legacy-off). Verify the guide's smoothing sigma (0.008 × max side) is correct at 447px and 1932px (ASSUMED current scaling; verify).

5. **Alternatives rejected:** replacing guidedFilter with a joint bilateral (same class); disabling ColorDiffusion altogether (regresses the inpaint-zone chroma diffusion, unmeasured cost).

6. **Tests to add:** compare hue-histogram entropy pre/post on a synthetic ragged-edge stub (rasterized国旗-free, purely synthetic L-ragged edges + surrounding flat chroma).

7. **Acceptance criteria (teared):**
   - hue-entropy metric (or Visible-posterisation metric) on [31,295,362,725] improves by ≥ 30% vs the legacy-guide run.
   - must-not-change: `fixture_colour_photo` chroma stats shift ≤ 3 units (the new guide must not wash a clean photo's chroma).
   - image3: the collar-crack region shows NO fringe appearance change if the tear mask is thin (avoid testing on an artifact case; the crack region is low-chroma so bleeding there is minor).

8. **Side-effect risks:** the smoothed guide loses tight chroma boundaries (e.g., dim lip edges blur, less crisp mouth). Compensate: radius=10, eps=60 are moderate; if a REAL photo suffers soft lip edges, reduce eps to 40 for that class (tune on data).

9. **Rollback:** `PIXREVIVE_CHROMA_SNAP_RAW=1`.

10. **Dependencies:** after FIX 1+3. Re-measure before touching this stage if FIX 1–3 have already removed the damage-ragged guides empirically (the leg work may already be dead after FIX 1/3).

11. **Effort:** S, 2–4h. **Confidence:** medium.

12. **Definition of done:** falsification artifacts archived; acceptance green; one commit.

---

### **FIX 8 — Hair scratch lines** | R-DMG (linear) | 6.8/25 | teared; same class as image3's collar cracks

1. **Problem:** OBSERVED: surviving hair scratch lines [129,497,176,561] on teared output. INFERRED: fracture masking catches cross-hairline scratches but not within-hair linear damage.

2. **Hypotheses:**
   - PRIMARY: the fracture mask's linear detector has a width/length gate above hairline-scale scratches.
   - ALT: detected but "pecking" discarded because they sit inside the hair region's variance gate.

3. **FALSIFICATION TEST:** dump the raw fracture mask as an overlay on teared input; overlay on the scratch lines ROI. Coverage < 10% on the scratch lines ⇒ detection gap (fixable); ≥ 10% with inpaint still cream ⇒ skip to ALTERNATIVE (eval LaMa fill in hair).

4. **Proposed change:** WORK WITH FIX 1 (the multi-scale tophat is already linear-friendly at the 3px kernel); if the layer in the falsification shows mask coverage gap, add the 3px tophat to the fracture-mask OR by default — no new stage.

5. **Alternatives rejected:** dedicated "scratch detector" (the multi-scale tophat IS the right mechanism).

6. **Tests to add:** one synthetic narrow scratch in hair-like texture asserted caught by `detect_white_flakes` multi-scale path.

7. **Acceptance criteria (teared):** scratch-line pixel count (defined as bright pixels in [129,497,176,561] above local+22) post-pipeline ≤ 5% of ROI area. (Justification: hair detail is the fixture's most detail-sensitive region.)

8. **Side-effect risks:** the 3px tophat firing on real hair highlights — measure false-positive on the same ROI in `fixture_clean_photo`.

9. **Rollback:** covered by FIX 1's flag.

10. **Dependencies:** after FIX 1 (same mechanism); may be a no-op if FIX 1's falsification already shows hair-band coverage.

11. **Effort:** S, 2–4h. **Confidence:** medium.

12. **Definition of done:** scratch ROI pixel count green; no false positives on clean fixtures; one commit.

---

## 4. INTEGRATION GATES

Between EVERY fix:
1. **Full test suite:** `pytest tests/ -q` — must return the same or better pass/fail count as the last gate (Baseline 435 collected; the pre-existing `test_colony_on_dark_substrate_is_masked` failure is a known separate issue, not introduced by these fixes).
2. **Full fixture matrix:** run `run_fixture_matrix.py` on all 6 fixtures.
3. **Metric diff:** `compute_fixture_metrics.py`, then diff against the previous gate's run AND against BASELINE-A.
4. **Go/no-go rule:** a fix PASSES the gate if:
   - no fixture's flake-blob count increases;
   - no must-not-change fixture ([`fixture_clean_photo`, `fixture_colour_photo`, `image1` vignette band, `image3` fabric band]) changes by more than its per-fix must-not-change threshold (defined per fix card);
   - identity (ArcFace/CLIP face similarity, or a manual + documented human check where weights are unavailable) does not drop below 0.85 on any face-bearing fixture;
   - colourfulness does not drop by more than 15% on any non-cast-fix fixture;
   - lapvar retention does not drop > 10% relative on any fixture.
   - **FAIL OF GATE ⇒** either revert the fix commit (git revert) or, if the failing metric is in a must-not-change fixture whose tolerance seems wrong, split the fix into a tighter scope and re-gate. NEVER bypass a gate by weakening an assertion.
5. **One commit per fix.** Commit-message convention: `FIX<N>(<defect-codes>): <one-line why>. Flags: PIXREVIVE_<...>=<value>. Gate: green on <date>`.

---

## 5. SPECIAL-CASE HANDLING MINI-PLANS

### (a) image3's thin facial tear
- The tear crosses hairline→brow→nose→upper-lip [177,324,575,620]. Mask must be **thin (< 15px)** or LaMa rebuilds nose/mouth geometry (trust-risk/identity).
- Detection: at this fixture's scale (678×452), a 10px-wide tear ≈ 2.2% of frame width; the fracture detector's linear path must catch it; verify overlay pre-LaMa. If the detector MISSES because the tear is **dark-on-skin**, extend the black-hat side (there IS a black-hat path already — `detect_dark_spots()` in damage_stages.py — exists in tree, no new stage).
- Inpaint guard: write a face-wipeout circuit-breaker post-inpaint: if the nose-mouth landmark region's Sobel-energy collapses relative to pre-inpaint AND ArcFace identity similarity falls below 0.85 vs the pre-inpaint face crop, revert the inpaint inside the face hull and log `inpaint_face_geometry_guard_triggered`. ASSUMED helper exists — `damage_stages.py` already carries a `_face_wipeout_circuit_breaker` (verified in tree during this session) — extend/parameterize rather than write new.
- Acceptance: post-inpaint ArcFace sim ≥ 0.85 (or human check); tear-region blob count = 0; no nose rebuild; garment cracks caught.

### (b) image1's hair flecks vs dark vignette
- Flecks [593,544,998,983] are small bright dots inside dark hair — the multi-scale tophat (FIX 1) should catch them; verify mask overlay. If it misses because the flakes are bright-but-not-white (L<242), the brightness threshold may need the existing multiscale kernel's per-scale thresholds, NOT a global drop.
- must NOT crop: the `detect_ragged_borders` bright rule correctly ignores a dark vignette per its synthetic tests; verify on the real image1.jpeg in the harness pre-flight (FIX 2 falsification covers it).
- must NOT route to denoise-first: σ(MAD)=0.00 measured; the noise router's grain path must stay inert — gate: `noise_score` for image1 must read < 2.5 in `meta_image1.json` (verify this is what the analyzer produces; if the analyzer reads 0 as "route to denoise" due to a fallback, that is a separate small fix — mark for falsification).

### (c) Greyscale / already-denoised inputs (Lena figure class)
- Colorizer skip: `_gray_skip_warranted` exists; verify it fires on the Lena figure (small greyscale figure with the white noise pathogenic). If it does not, adjust thresholds (ASSUMED current thresholds unknown — inspect `color_stages.py`).
- Denoise inert: σ(MAD)=0.00 on a denoised figure should not trigger a heavy NLM; verify damage cluster smoother's variance gate (35.0 threshold — this is a real number only if icrosoft the analyzer pivots on `local_variance > 35.0`, verify).
- Border strip: the ORIGINAL uniform detector must crop Lena's baked white right-edge strip; verify again in the pipeline (it currently is the fallback's job ONLY IF the uniform detector misses).
- Acceptance: colorizer skip log line present; no SR blowup; output lapvar retention ≥ 0.8×.

### (d) Printed-halftone sources
- teared.jpg carries a visible halftone dot lattice (OBSERVED). Two risks: DDColor reading dot texture (bad semantic confound) and the flake detector reading bright dots as flakes (over-detection).
- DECISION: **no up-front halftone-descreen pre-step** — it softens real detail on non-halftone fixtures and is a separate stage. Instead:
  - measure: dot-detection probe (analyse the input's high-frequency periodic energy with an FFT peak test on the suspect crops) — a cheap descriptor only, no stage change;
  - verify: if the DDColor falsification (FIX 3) shows ALT-1 TRUE (model misreads dots), revisit with a targeted low-pass pre-DDColor bounded by the periodic-descriptor (never global softening);
  - the multi-scale tophat's halo must be tuned so its smallest kernel (3px) does NOT fire on dot-lattice texture (measured dot spacing is larger; verify with a synthetic halftone stub test).

---

## 6. RISK REGISTER

| Risk | Likelihood | Impact | Early-warning signal | Mitigation |
|--|--|--|--|--|
| Cast fix masks a flake bug (FIX 3 colours right because blobs are now gone) | Medium | High | flake-blob count non-zero in final but colour metric now green | keep FIX 1's blob-count metric as its own gate; never let the cast metric replace it |
| Over-correction → grey/neutral outputs | Medium | High | near-neutral share jumps > 60% post-fix on a portrait | colourfulness metric + validator WARN; roll back the floor |
| Over-smoothing returns (grain injection too weak / NLM unchanged) | Medium | Medium | lapvar retention < 0.45 on a portrait | keep lapvar metric; tune amplitude on the darkest fixture first |
| Identity drift from GFPGAN or inpainting | Low | Critical | ArcFace/CLIP sim < 0.85 on a face fixture | §4 gate hard-threshold; image3's tear circuit-breaker |
| VRAM/RAM blow-up on other hardware | Medium | High | cudnn OOM in logs on a fixture | all fixes here are CPU/NLM/ OpenCV-path; SR stays behind Super.restore's 4096-cap; no fix adds a big model |
| Flag sprawl (7+ PIXREVIVE_* flags hard to reason) | High | Medium | new flag needed in the next fix | §0.3 centralise after the plan's core fixes land |
| Uncommitted-fix data loss touching the current tree | Low | Critical | `git status` still shows 4 modified fix files | §0.2: commit both batches FIRST |
| Halftone dot texture defeating flake detector (over-firing) | Medium | Medium | blob count on `fixture_clean_photo` > 0 after FIX 1 | synthetic halftone stub test; smallest-kernel halo tuning |

---

## 7. TIMELINE (half-day blocks, ~6 days total)

| Block | Work | Review artifact |
|--|--|--|
| D1-AM | COMMIT HYGIENE both batches + Phase 0 harness scripts written + fixtures prepared + test-coverage additions (§2.5 items 1–5) | commit hashes + harness runs on all 6 fixtures |
| D1-PM | BASELINE-A metrics recorded for all fixtures | `metrics_*.json` baseline dir + §1.2 numbers re-anchored |
| D2-AM | FIX 1 falsification test + tune + commit | blob-count metric = 0 on teared's 5 ROIs; mask overlay artifacts |
| D2-PM | FIX 2 pre-flight + cap + commit + tests | border-crop assertions on all 4 margin-class fixtures |
| D3-AM | FIX 3 falsification test (DDColor isolated A/B) + assumption check | raw-DDColor dump artifacts + a/b tables |
| D3-PM | FIX 3 change + tests + commit + gate | cast metrics green on teared; colour-photo fixture unchanged |
| D4-AM | FIX 4 falsification + cap verification + grain amplitude tuning + commit | lapvar retention + σ(MAD) on face/bg |
| D4-PM | FIX 5 falsification + sweep-parameter work + commit | post-SR blob count; halo metric at/below baseline |
| D5-AM | FIX 6 (iris/lip) falsification + nudge + commit | inter-iris Δab; lip a* delta; identity check |
| D5-PM | FIX 7 (guided filter guide) re-falsify + commit (or archive as already-covered) | hue-fringe metric diff vs legacy guide |
| D6-AM | FIX 8 (hair scratches) falsification + commit (or fold into FIX 1's already-committed work) | scratch ROI pixel count |
| D6-PM | §5 special-cases: image3 tear-guard + image1 vignette/denoise-routing + Lena grey-skip + halftone stub test | fixture-specific acceptance rows all green |
| D7 (buffer) | full matrix re-run; final before/after table; regression watchlist sign-off; FLAG docs | §8's final table |

---

## 8. DEFINITION OF DONE (whole effort)

1. **teared.jpg:** Restoration ≥ 6/10, Colorization ≥ 6/10 by the audit rubric, using these measurable proxies:
   - flake-blob count = 0 in all 5 named ROIs;
   - residual margin width ≤ 2% per side;
   - skin ROI a* ∈ [6,20], b* ∈ [12,32]; face-vs-bg b* separation ≥ 10;
   - inter-iris Δab ≤ 8; lip a* ≥ cheek a* + 10;
   - lapvar retention ≥ 0.45×; flat σ(MAD) ∈ [1.5, 4.0];
   - face identity similarity ≥ 0.85 (ArcFace, or CLIP + human check if weights unavailable).
2. **image1.jpeg:** blob count = 0 in the fleck ROI; vignette NOT cropped; no denoise-first routing; output ≥ input.
3. **image3.jpg.jpeg:** tear region inpainted with identity similarity ≥ 0.85; nose/mouth geometry preserved (no rebuild); collar cracks and mould flecks removed; no border crop.
4. **SigShrink-Lena figure:** white strip cropped; colorizer skipped; denoiser inert.
5. **`case1_noisy_lena`:** σ(MAD) after ≤ 8; no other-fixture regressions.
6. **`fixture_clean_photo` / `fixture_colour_photo`:** no mask, no border crop, no grain inject, chroma shift ≤ 3 units.
7. **Every fix is behind a flag with a one-line rollback documented in `docs/FLAGS.md`** and the flag defaults recorded in §0 table.
8. **The regression watchlist** (sepia/olive cast, surviving flakes, LaMa hallucination on noise, colour bleeding, over-smoothing, identity change, SR halos) gets a formal per-fixture PRESENT/ABSENT sign-off, and the entries that remain PRESENT are explicitly listed as known-remaining with their metric values.
9. **435-collected test suite** passing at the end, no weakened assertion, no skipped test introduced by this effort.

---

## 9. WHAT YOU NEED FROM ME

Ordered by how much each unblocks:

1. **Decision: commit batching (§0.2)** — confirm Batch A/B contents; one session to complete. Unblocks ALL.
2. **Fixture set confirmation:** point me at `case1_noisy_lena.jpg`'s real path (ASSUMED present per dataset); confirm the SigShrink-Lena figure file stays in the matrix as a border/grey-skip fixture (or supply a real speckle-noisy Lena).
3. **The raw DDColor pre-ColorDiffusion intermediate dump** (`06a_raw_ddcolor`) — needed for FIX 3's cast falsification *if* the harness extension is not in place yet.
4. **The damage-mask overlay artifact for teared.jpg pre-LaMa** — needed for FIX 1's falsification if the harness dump is not wired yet.
5. **GPU/CLIP/ArcFace decision:** whether ArcFace w600k weights can be fetched (`scripts/fetch_faceid_weights.py` exists in the repo per file references, ASSUMED) — this decides whether identity is metric-backed or human-check-backed in the acceptance criteria.
6. **A green-light on fixing policy for `image3`'s tear** — accept that the identity-similarity guard is a "revert-inpaint, log, still-deliver" and the missing thi should ship with `inpaint_face_geometry_guard_triggered` metadat flagged in the response to the caller; or specify a different policy.
7. **Confirmation of the exact fixture demonstrator I should use for `fixture_colour_photo.png`** — any well-coloured portrait you own; a synthetic will do but a real one is stricter.

---

*Prepared from live-tree verification (git status, flag greps, function greps) and the QA audit of the teared.jpg pair; every ASSUMED name is marked; every fix begins with a falsification test and carries numeric acceptance criteria; the execution order respects the dependency rationale in §1.3/§1.4.*
