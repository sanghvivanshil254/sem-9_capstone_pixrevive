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
| Banding / posterization | 🟡 | Detected (banding_score) and `_reduce_compression` touches it gently; dedicated bit-depth-expansion + dithering pass would do better. **T-6 (P2, cheap)** |
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
| Differential dye fading | 🟡 | Color cast correction exists in `Enhancer.correct_color_cast` — which is dead code (D17). Channel-curve rebalancing not wired. **T-7 (P2)** |
| Yellowing / sepia shift | ✅ | Sepia detection + neutralization before colorization; white-balance via ColorArtifactReducer |
| **Foxing / mold spots** | ❌ | `EnhancedDamageDetector.detect_dark_spots()` **exists but is never called** by Stage 2 — only 5 detectors + tearmark are wired. One-line wiring + regression case. **T-1 (P1, cheapest high-value fix in the plan)** |
| Silver mirroring | 🚫/🟡 | Rare; localized tone correction possible but niche. **Accept — document only** |
| Water stains / mold texture | 🟡 | Covered only insofar as stains register as flakes/dark spots; region color correction absent. **T-7 partially addresses** |
| Emulsion cracking / crazing | ✅ | `detect_thin_crack_network` (Hessian + multi-scale black-hat) — a pipeline strength |
| Curling / warping | ❌ | No dewarp. Estimable from border-line geometry (Hough). **T-4 (P2)** |

## 3. Optical / Capture Issues

| Type | Verdict | Where / What's missing |
|---|---|---|
| Motion blur | ❌ | No deblur stage; Stage 10 USM sharpens but cannot recover blur. **T-3 (P1) — Restormer/NAFNet gated stage** |
| Out-of-focus blur | ❌ | Same gap as motion blur → T-3 (same stage handles both) |
| Chromatic aberration | ❌/🟡 | Channel realignment absent; rare in scans. **T-8 (P3)** |
| Vignetting | ❌ | No flat-field correction. **T-5 (P2, classical, cheap)** |
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
| Moiré / halftone | 🟡 | `_suppress_halftone` bilateral pre-filter in SR runs, but the stronger FFT-notch `HalftoneSuppressor` (D5) is dead code. **Decision: wire HalftoneSuppressor into Stage 1 as the moiré specialist OR delete it and accept bilateral-only. Recommend wire (T-9, P2)** |
| Scan skew / rotation | ❌ | No deskew. `DeskewCropAction` (orchestrator, dormant) only crops solid borders — misnamed, no Hough rotation. **T-4 (P2)** — watermark-style horizontal-line estimate + affine correction in `ImageNormalizer` |
| Uneven scan illumination | ❌ | No flat-field. **T-5 (P2)** — large-kernel background estimate (morphological closing / big Gaussian) + division correction, gated on gradient magnitude |
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
