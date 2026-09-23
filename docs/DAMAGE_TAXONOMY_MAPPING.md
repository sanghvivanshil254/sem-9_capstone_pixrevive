# PixRevive — Damage Taxonomy → Pipeline Coverage Map

Maps the full damage/noise taxonomy (`photo_restoration_damage_taxonomy.md`) onto what
PixRevive's 14-stage pipeline actually does today, and assigns an action ID for every gap.
Cross-references: `D#` = docs/DEAD_CODE_REMOVAL.md · `U/Q/A/O` = docs/IMPROVEMENT_SUGGESTIONS.md ·
`T#` = this document (new work items).

**Verdicts:**
- ✅ **Covered** — a wired stage handles it; no action.
- 🟡 **Partial** — handled for common cases, weak for severe ones.
- ❌ **Gap** — no wired handling; action item defined.
- 🚫 **Beyond scope** — genuinely unsolvable in principle; needs *flagging*, not filling (see §Honesty).

Evidence checked against code on 2026-09-21 (grep + file reads; NoiseRouter declares 30 noise
types in `SUPPORTED_NOISE_TYPES`).

---

## 1. Noise & Sensor-Level Artifacts

| Type | Verdict | Where / What's missing |
|---|---|---|
| Gaussian noise | ✅ | NoiseRouter random family → edge-aware NLM; Stage 5 3-zone NLM scaled to measured σ (MAD estimator) |
| Salt & pepper / impulse | ✅ | Stage 0 ImpulseDenoiser (ROAD test) + NoiseRouter `_reduce_impulse`; density-invariance regression-tested |
| Speckle | ✅ | Stage 0 SpeckSuppressor (autocorr + edge co-location gating); regression-tested on case2/case3 |
| Poisson / shot | 🟡 | Lumped into "random" family; signal-dependent σ (worse in darks) not modeled. **Accept** — LAB MAD estimate is a good proxy; no action |
| Film grain | ✅ | Deliberately *preserved* on clean backgrounds (Stage 5 bg guard: high σ alone does not trigger aggressive smoothing) — matches taxonomy's "preserve vs remove is a choice" |
| Chroma noise | ✅ | NoiseRouter `_reduce_chroma` (luma/chroma separated); Stage 8 pre-SR chroma bilateral |
| Banding / posterization | ✅ | **T-6 (4B.4)**: `ai/postprocessing/deband.py` — float luminance smoothing + triangular-PDF dither, applied pre-SR inside Stage 8, gated on banding_score ≥ 0.6 (corpus max 0.503 stays bitwise inert; synthetic posterization reads 1.0) |
| JPEG blocking / ringing | ✅ | `_blocking_score` detection + `_reduce_compression` deblock/deringe stage |

## 2. Physical / Chemical Degradation

| Type | Verdict | Where / What's missing |
|---|---|---|
| Surface scratches | ✅ | `detect_scratches` (directional morphological) + U-Net DefectSegmenter gate → LaMa |
| Dust & debris specks | ✅ | Stage 0 impulse path + `detect_dark_spots`… **but see Foxing below — the dark-spot detector itself is dead code** |
| Creases / folds | ✅ | `detect_wide_cracks` multi-scale black-hat + LaMa; seam smoothing |
| Small tears (no content loss) | ✅ | Same path; edge-continuity via LaMa |
| Large missing pieces | 🟡 | LaMa fills plausibly — but **no flag that result is fabrication**. → §Honesty (T-2) |
| Fading (overall) | ✅ | Stage 10b CLAHE + filmic curve; ColorSmoother |
| Differential dye fading | ✅ | **T-7 (4B.5)**: `ColorArtifactReducer.rebalance_dye_fade` (D17 salvage) wired input-side as `cast_fix` stage — color-classified prints only beyond deviation 8.0; sepia/grayscale excluded BY TYPE (historic character, not a defect) |
| Yellowing / sepia shift | ✅ | Sepia detection + neutralization before colorization; white-balance via ColorArtifactReducer |
| **Foxing / mold spots** | ✅ | **T-1 (Phase 2)**: `detect_dark_spots` wired into Stage 2 with percentile-adaptive detection; router escalates measured dark-spot evidence to repair routes (4A) |
| Silver mirroring | 🚫/🟡 | Rare; localized tone correction possible but niche. **Accept — document only** |
| Water stains / mold texture | 🟡 | Stains register as flakes/dark spots (both wired); T-7's input-side cast rebalance addresses the tonal component. Region-level stain recolor remains out of scope. **Accept** |
| Emulsion cracking / crazing | ✅ | `detect_thin_crack_network` (Hessian + multi-scale black-hat) — a pipeline strength |
| Curling / warping | 🟡 | Full dewarp not attempted. **T-4 (4B.1)** delivers the geometry half: Hough dominant-angle deskew (+ affine correction) straightens rotated scans; perspective/curl remains. **Accept residual** |

## 3. Optical / Capture Issues

| Type | Verdict | Where / What's missing |
|---|---|---|
| Motion blur | ✅ | **T-3 (4B.6)**: Stage 4.5 Restormer deblur (vendored, official checkpoint) — fires on H/V gradient anisotropy ≥ 2.2 (motion path B); corpus max 1.69 stays inert |
| Out-of-focus blur | ✅ | **T-3 (4B.6)**, path A: edge-spread acutance ≥ 0.47 with the PSF-widening signature (corpus max 0.456; sigma ≥ 1.5 measures 0.48+); sharp-but-low-detail photos measured inert |
| Chromatic aberration | ❌/🟡 | Channel realignment absent; rare in scans. **T-8 (P3)** |
| Vignetting | ✅ | **T-5 (4B.2)**: flat-field correction via quadratic-field gain (max 2×), gated on SYMMETRIC corner deficit ≥ 20% (spread ≤ 12%); runs after border-crop so frames cannot fake a lamp defect; one-sided falloff documented false-negative |
| Lens flare / light leaks | 🟡 | Bright-region detectors partially catch it (flake detector guard rails prevent overreach); dedicated handling out of scope. **Accept — document** |
| Double exposure / ghosting | 🚫 | Under-determined; must be flagged, not processed. → §Honesty (T-2) |

## 4. Missing / Illegible Content

| Type | Verdict | Where / What's missing |
|---|---|---|
| Small occlusions | ✅ | LaMa path — high-confidence fills |
| Medium missing regions | ✅ | LaMa path |
| Damaged faces | 🟡 | GFPGAN + inpaint circuit breaker + deferred post-SR pass — good engineering, but identity accuracy is fundamentally generative → flag (T-2) |
| Fully destroyed sections | 🟡 | LaMa will fill *anything* — silent fabrication today. **T-2 must catch coverage ≥ threshold and label output** |
| Illegible text | 🚫 | No OCR path; out of scope → flag only (T-2) |

## 5. Color & Tone Issues

| Type | Verdict | Where / What's missing |
|---|---|---|
| Color cast | ✅ | Stage 6c ColorArtifactReducer cast reduction |
| Color bleeding | ✅ | EdgeAwareColorBleedFixer + Stage 6 chrominance diffusion with border protection — sophisticated |
| B&W → color | ✅ + honesty | DDColor (opt-in only — router never auto-colorizes). Ambiguity disclaimer → T-2 wording |
| Cross-processing shift | 🟡 | Curve remapping partially via ColorSmoother. Accept |

## 6. Resolution & Scanning Artifacts

| Type | Verdict | Where / What's missing |
|---|---|---|
| Low resolution / soft | ✅ | Real-ESRGAN ×4 tiled, 2K normalize, halftone pre-filter; Q-5 fixes the fallback cap |
| Moiré / halftone | ✅ | **T-9 (4B.3)**: HalftoneSuppressor (D5 salvage) wired inline after despeckle — FFT spectral-peak detection + Gaussian notch, self-gated with MAE safety valve; probe-verified inert on all 17 real photos (incl. case2 macro edges); SR bilateral pre-filter remains the faint-screen backstop |
| Scan skew / rotation | ✅ | **T-4 (4B.1)**: `ImageNormalizer.deskew` — exhaustive HoughLines dominant-angle vote (deterministic), 0.3° inertness gate, ±8° range, runs pre-border-crop so rotation wedges get cropped; registered as the first registry T-stage |
| Uneven scan illumination | ✅ | **T-5 (4B.2)** — see Vignetting above (same stage; gate discriminates lamp defects from content via corner-falloff symmetry, chosen by probe after channel-correlation and R² candidates failed) |
| Interlacing | 🚫 | Video captures out of scope for photo restoration. **Accept — document** |

---

## §Honesty Layer (T-2) — the taxonomy's core lesson

The taxonomy's bottom line: *"any AI output [for unsolvable categories] is a plausible
reconstruction, not a recovered fact — which is exactly what a forensic validator /
damage-coverage gating system is designed to enforce."*

PixRevive has the ingredients but never connects them:
- `SynthesisManifest` provenance tracking exists (dormant, orchestrator-only)
- `ForensicValidator` exists with 6 hard-constraint audits (wired only into `scripts/forensic_restore_validator.py`)
- Stage 2 computes per-detector masks (flake/wide/scratch/tear/3d-tear) with coverage % — discarded after inpainting
- Stage timings already collected in `context.stage_timings`

**T-2 builds a `RestorationReport` (backend/ai/report/):**
1. **Coverage accounting** — persist per-detector mask coverage + inpaint coverage from Stage 2/3 (already computed, just keep it).
2. **Classification** — map damage evidence to taxonomy classes: *recovered* (noise/scratches/flakes), *reconstructed* (inpaint area, colorization), *unrecoverable* (inpaint coverage ≥ 25% of frame, or face-region inpaint ≥ 40% with GFPGAN applied, or border/geometry total loss).
3. **Verdict strings** — machine-readable: `{repaired: [...], reconstructed: [...], flagged: [...]}` attached to metadata + rendered as a Restoration Report card in the UI (collapsible, per-stage timings included per O-3).
4. **Hard flags** — `plausible_reconstruction` marker set in metadata; PNG stays PNG; the *UI* carries the honesty, not the pixels.

This is the capstone differentiator: it converts the taxonomy from documentation into a product feature, and it reuses D6 (QualityService slot), D7 (IdentityValidator debug check), and D16 (route manifest) instead of deleting them.

---

## New T# work items (added to EXECUTION_PLAN Phase 4)

| ID | Item | Priority | Effort |
|---|---|---|---|
| T-1 | Wire `detect_dark_spots` into Stage 2 (foxing/mold/dust); add foxing regression fixture | P1 | XS |
| T-2 | Restoration Report: coverage accounting + recovered/reconstructed/unrecoverable classification + UI card | P1 | M |
| T-3 | Deblur stage 4.5 (Restormer ~65MB, gated on blur_score < threshold & directional-variance test); sits after denoise, before colorize | P1 | M |
| T-4 | Auto-deskew: Hough-line dominant rotation estimate + affine correction in ImageNormalizer (before border crop) | P2 | S |
| T-5 | Flat-field illumination correction (background estimate + divide), gated on measured gradient | P2 | S |
| T-6 | Banding/posterization pass: 16-bit intermediate + ordered-dither re-quantization on smooth gradients | P2 | S |
| T-7 | Channel-curve color-cast rebalancing (revive `correct_color_cast` logic from deleted Enhancer into ColorArtifactReducer) | P2 | S |
| T-8 | Chromatic aberration channel realignment (P3, rare) | P3 | S |
| T-9 | Wire HalftoneSuppressor (D5) into Stage 1 as FFT moiré specialist ahead of SR bilateral pre-filter | P2 | S |

**Deliberate non-goals** (documented in README limitations section): silver mirroring, lens
flare, interlacing, ghosting/double-exposure separation, OCR text recovery.
