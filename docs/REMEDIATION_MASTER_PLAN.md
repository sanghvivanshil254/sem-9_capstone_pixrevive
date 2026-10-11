# Remediation Master Plan — from critiqued outputs to semantically honest restoration

**Status: planned (2026-09-29). Supersedes the Phase V0.2 section of
[EXECUTION_PLAN.md](EXECUTION_PLAN.md) as the single execution reference; absorbs and
extends the V1 addendum in [SEMANTIC_VALIDATION_PLAN.md](SEMANTIC_VALIDATION_PLAN.md).**

Every item below is grounded in a measured number from this codebase's own evidence —
none of it is a guess about what might be wrong.

---

## 0. Executive summary

Three external critiques (speckle-Lena figure 4/10, teared portrait 6/10, plus a
user-spotted defect list) decompose into exactly six failure modes. Each is
stage-attributed, each has a workstream here, and the workstreams are organized as
three layers so the plan reads as a system, not a list:

| Layer | Question it answers | Workstreams |
|---|---|---|
| **L1 Prevent** | How do we stop *manufacturing* defects? | W1 paste guard, W2 anatomy halo, ✔ V0.1 grain floor (shipped `38d7de1`) |
| **L2 Detect** | How do we *notice* defects that still occur? | B3/A3/C1 rules (V0, shipped `8ad9822`), W7 seam/HF observability, W8 regression pins |
| **L3 Repair** | How do we *honestly rebuild* what is lost? | W3=V1a mask compiler, W4=V1b LaMa refill, W5=V1c diffusion refill, W6=V2 garment rules |

The through-line: V0.1 stopped feeding noise to the generators; V0.2 stops the
generators from mangling structure they already had; V1–V2 make the pipeline able to
*put back* what is genuinely lost — with identity gates, evidence discipline, and
disclosure at every step.

---

## 1. Evidence base (all numbers measured this codebase, 2026-09-29)

| # | Fact | Source |
|---|---|---|
| E1 | Speckle figure: grain_floor routed, NLM accepted (edge_loss 0.5%, residual 6.06→5.84→5.69); pre-SR armed at noise 3.75; background HF energy 4.96→4.24, face top-hat patches 2744→912 | offline debug run + contact sheet |
| E2 | Teared: Stage 5 ran for the first time on this class (σ 7.41→1.48 — the dispatch bug had killed it since the A-2 refactor, `denoising: 0.0`); background HF 8.63→3.16 (−63%), face patches 5268→1187 | offline debug run + contact sheet |
| E3 | Speckle eye square = GFPGAN paste rect **255×167 px** on the 435×342 frame; the detected eye landmark sits **outside** it; LaMa mask covers only **7.04%** of the eye box | `debug_output_speckle/` before/after-GFPGAN delta |
| E4 | Teared ear: **34.9%** of the ear region was in the LaMa damage mask; ear gone at `03_after_inpaint`; ear lies outside the face bbox, so MaskRefiner's anatomy protection (**102 px**) never reached it | `debug_output_teared/` stage row + log |
| E5 | Teared collar right: **27.5%** masked; fold line inside the flake cluster → LaMa filled plain fabric | `debug_output_teared/` stage row |
| E6 | Live teared Stage 12 (whole-frame hybrid) **skipped** — frame 32.96% < 35% bar, worst face 25.15% < 55% bar; A3 hairline candidate persists on output; B3 fires live (teared 169 patches / 0.35% / evidence_lost 25%; speckle 52 / 0.17%) | V0 live run + V0.1 runs |
| E7 | Corpus separators: clean scans residual_mad 1.39–1.88; grain class 3.46–4.0; heavy 13.67+. SFace on 60%-destroyed reference measured 0.095 → absolute identity gates impossible; delta-gating required | corpus probes + H4 measurements |

Evidence artifacts: `test_artifacts/contact_sheet/defect_forensics.html`,
`contact_sheet.html`, probes `.freebuff/probe_defect_forensics.py`,
`.freebuff/probe_stage2_speckle.py`, debug archives `debug_output_speckle/`,
`debug_output_teared/`.

---

## 2. Workstreams (each independently shippable, with tests + acceptance)

### W1 — GFPGAN paste-alignment guard *(L1; ~1 session; files: `ai/restoration/face_restorer.py`, new `tests/test_face_paste_guard.py`)*

Kills defect E3. The aligned-crop composite in `_restore_single_face` feathers
correctly, but nothing verifies the paste footprint is anatomically aligned before
`paste_faces_to_input_image` (line ~385).

1. **Footprint warp**: warp each face's crop `alpha_mask` to canvas coordinates via
   `affine_mat`; composite all faces into one `paste_footprint` mask (uint8 0/255).
2. **Containment check**: map `helper.all_landmarks_5[i]` to canvas coords; both eye
   landmarks must fall inside the footprint's connected component. Failure →
   `image_metadata["gfpgan_paste_misaligned"]` + WARN log, then:
   - landmark-to-footprint-centroid distance ≤ 0.25 × face width → re-blend that face
     with `fidelity_weight` 0.85 → 0.5 (keeps most of the restore, shrinks the delta);
   - beyond that → skip the paste for that face entirely (original pixels win).
   Record which action fired per face in the Restoration Report.
3. **Seam metric (advisory)**: post-paste |before − after| inside the footprint; if
   the outer 8 px ring's mean delta > 3× the interior's → feather the ring once and
   re-composite.
4. **Tests**: offset synthetic paste → guard fires and falls back; aligned paste →
   silent; footprint equals the analytically warped alpha; synthetic hard-edge paste →
   seam feather triggers.
5. **Acceptance**: speckle offline rerun — no eye landmark outside the |delta| rect;
   no rectangle seam in the contact-sheet crop; suite green; Gate A still 0 FP.

### W2 — Anatomy-aware damage protection *(L1; ~1 session; files: `ai/damage/mask_refiner.py`)*

Kills defects E4/E5's *cause* (mask authoring), independent of any generator work.

1. **Head-region halo** built from landmarks + bbox: ear discs (r ≈ 0.22 × face width,
   centered at bbox left/right edges at eye height), hair-band ring (bbox top −
   0.35 × face height), neck/collar band (bbox bottom + 0.25 × face height, full bbox
   width). Subtract the halo from the final damage mask; flakes leaving the mask are
   cleaned by Stage 4b/5 NLM instead of LaMa.
2. **Cap discipline**: the existing 8% face-ceiling stays for the bbox interior; the
   halo is additional and logs `protected_px` per region — E4/E5's 34.9%/27.5% become
   tracked regressions, not folklore.
3. **Regression fixture**: teared.jpg keeps ≥60% of ear-region flake pixels out of the
   LaMa mask (measured baseline: 65.1% would be unmasked under the current halo
   geometry; assert ≥60% masked→unmasked flip), while the output crop's HF energy must
   not regress vs the V0.1 sheet (NLM still cleans what LaMa no longer inpaints).
4. **Acceptance**: teared offline rerun — `03_after_inpaint` frame shows an ear
   silhouette (noise acceptable); collar fold partially survives; B3/A3 verdicts
   unchanged or better; suite green.

### W3 — V1a findings→mask compiler *(L3; ~1 session; files: `ai/validation/semantic_validator.py` (emit only), new `ai/validation/semantic_actions.py`)*

Zero generation code. Compiles report findings into refill masks:

- Sources: A3 regeneration candidates (hair band), **B3 skin-patch clusters merged
  into contiguous zones** (E6's 169/52 patches become few region masks), and **W2's
  halo-deferred ear/collar zones** (the flake fields LaMa no longer eats become
  explicit "lost-detail candidates" instead of being silently dropped).
- Geometry: finding band ∩ evidence>0, dilated, clipped to the face-bbox neighborhood;
  emitted as `image_metadata["semantic_actionable"] = [{rule, region, mask_ref,
  evidence_state, priority}]`.
- Unit tests assert mask geometry against bbox anchors on fixtures.
- **Acceptance**: masks verified on teared + case9; Gate A rerun still 0 FP (the
  compiler must no-op on pristine images — same bar as V0).

### W4 — V1b LaMa region completion *(L3; ~1–2 sessions; files: new `ai/restoration/region_refiller.py`, Stage 13 hook)*

The core honest-repair loop, using the existing LaMa path — no new weights:

- When candidates exist **and** the `reimagine` opt-in is set: re-fill **only** the
  candidate masks; re-assert evidence bytewise outside the mask.
- **SFace delta-gate** (E7): post-refill cosine ≥ pre-refill cosine − margin (delta,
  not absolute — absolute gates are unachievable at heavy destruction).
- **SSIM revert guard**: if SSIM inside the band got worse, revert the region and
  disclose `action: attempted-rolled-back`; success discloses `regenerated` with both
  cosines. `candidate-reported` otherwise.
- E6 pivot honored: this runs on the LaMa canvas — it does **not** assume whole-frame
  Stage 12 ran (teared proves it won't, on this class, at any threshold we can defend).
- Bounded loop: one pass; if the finding persists with unchanged evidence state,
  disclose "attempted, unresolved" (prime directive forbids retry wars with absent
  evidence).
- **Acceptance**: teared live run with hair band + ear/collar candidates attempted;
  identity delta passes on ≥2 real runs; Gate A 0 FP; report discloses every action.

### W5 — V1c diffusion region completion *(L3; ~2 sessions; files: Stage-12 engine region routing)*

Only after W4's gates prove the loop; only when the resource tier ≥ MINIMAL:

- Candidates route through the SD1.5-inpaint + FaceID engine **region-cropped**
  (H4's identity conditioning; reroll ladder as budget allows). LaMa refill (W4)
  remains the always-available floor.
- Identity gate here may use absolute thresholds *plus* rerolls (headroom exists via
  reroll ladder), but the delta-gate stays the primary honest comparator.
- **Acceptance**: case9 + teared live runs; SFace verdicts recorded per reroll;
  disclosure parity with W4.

### W6 — V2 garment & continuity activation *(L3; ~2 sessions)*

The collar class (E5) ultimately wants generation, not just protection:

- C1/C2 + B1/B2 go active; surviving-side collar crop becomes the IP reference for
  mirror-constrained continuation (existing adapter path, zero new weights).
- `paper_neutrality` (B2) additionally retires the Δab-27 border flag class seen in
  the speckle live run's report.
- **Acceptance**: collar/button continuation demonstrated on a fixture with a
  destroyed torso zone; Gate A 0 FP.

### W7 — Observability: make stage decisions visible *(L2; ~½ session; files: `routers/image_router.py`, `ai/pipeline/debug_logger.py`)*

This session's defects were found only because PIXREVIVE_DEBUG archives existed; the
live SSE path hid everything (`denoising: 0.0` masked a crash; Stage 4b lines never
surfaced).

- Append a processing-log tail (last ~25 lines incl. PipelineRouter/PasteGuard/
  NoiseRouter/Semantic lines) to the SSE `complete` event payload.
- DebugLogger: always dump `paste_footprint`, `anatomy_halo`, and `semantic_actionable`
  masks into `debug_masks/`; log `protected_px` + guard actions per run.
- **Acceptance**: a live run's SSE alone suffices to answer "which stage decided
  what" for all three defect classes.

### W8 — V3 calibration & regression pins *(L2; ~1–2 sessions, continuous)*

- Threshold sweeps per rule (H3 pattern: fixed seed, fresh process, one variable).
  Known calibration debt: A1 AMBIGUOUS fired 17× with `texture_std=-1` (structure
  proxy N/A) — never let it trigger generation until calibrated; add a **textured
  headwear fixture** before W5 (A3 can mimic its signature; corpus has none today).
- Regression pins as tests: teared background HF energy < 4.5 (E2 baseline);
  ear-mask protected share ≥ 60% (W2); Gate A 0 FP as a corpus-gate CI run;
  paste-containment check green on the pristine corpus (W1 must never fire there).
- Docs closeout: DESIGN §13 addendum, EXECUTION_PLAN, CHANGELOG per phase.

---

## 3. Sequencing & dependency graph

```
V0.1 (shipped) ──► W1 ──► W3 ──► W4 ──► W5
                    │            ▲
                    └──► W2 ─────┘        W6 (after W4 proves the loop)
W7 anytime after W1 (it documents W1's decisions)
W8 runs continuously; its pins gate every phase's exit
```

- W1 → W2: no coupling (disjoint files), either order works; W1 first because it is
  smallest and its evidence (footprint masks) feeds W7's debug dumps.
- W3 needs W2's halo-deferred zones to exist; W4 needs W3's masks.
- W5 and W6 both branch off a proven W4 loop; W5 first if identity quality dominates,
  W6 if garment complaints dominate user feedback.
- Effort total: ~9–12 sessions; every phase ships user-visible evidence (contact
  sheet or report disclosure) — no phase is "internal only".

---

## 4. Unified test strategy

| Level | What | Where |
|---|---|---|
| Unit | guard logic, halo geometry, mask compiler, delta-gate math, seam metric | `tests/test_face_paste_guard.py`, `tests/test_anatomy_halo.py`, `tests/test_semantic_actions.py`, `tests/test_region_refiller.py` |
| Fixture | teared.jpg + case9 + headwear fixture through offline pipeline with assertions on masks/frames | extends `tests/test_restoration_regressions.py` pattern |
| Corpus | `scripts/corpus_semantic_gate.py` Gate A 0 FP after every phase that touches masks or rules | existing gate, rerun each phase |
| Live | HTTP SSE runs of speckle + teared; contact-sheet before/after as the acceptance artifact | `/api/process-stream` + probes |
| Discipline | full suite only with backend paused (torch coexistence segfault); runtime rules per AGENTS.md | documented, enforced by habit + hooks |

---

## 5. Risk register & pre-registered no-go triggers

| Risk | Mitigation / trigger |
|---|---|
| W1 guard skips too many faces (over-firing) | containment failure rate logged; pristine corpus must show ~0 guard actions (pin in W8) — if it fires on clean faces, geometry is wrong, not the threshold |
| W2 halo starves LaMa (real damage left unmasked) | halo capped at head-region bands; `protected_px` logged; SSIM/HF checks on output crops; B3/A3 verdicts must not degrade |
| W4 region refill worse than LaMa's whole-frame fill inside a band | SSIM revert guard + honest `attempted-rolled-back`; **no-go** if corpus-majority refills revert |
| SFace delta-gate fails to separate refill quality across ≥3 real runs | **no-go** before W5; recalibrate V3-style (fixed seed, one variable), re-gate |
| Gate A regression from mask dilation (W3) | **no-go**: any pristine FP from compiler dilation → tighten clip/dilation, re-gate |
| A3 mimicked by textured headwear → bad refill | headwear fixture required **before** W5 |
| Identity collapse at heavy destruction | delta-gating per E7; absolute thresholds only in W5 with reroll headroom |
| Observability rot (W7 drifts from reality) | SSE tail asserted in `tests/test_image_router.py` style tests |

## 6. Definition of done (the whole plan)

1. Fresh before/after contact sheet (same protocol as V0.1's) where: no eye square
   (W1), ear + collar survive with detail (W2, W4/W6), background grain class stays
   suppressed (V0.1 pin), and the speckle figure's score-driving artifacts are
   disclosure-listed rather than silent.
2. Gate A 0 FP maintained through every phase; prime directive asserts in tests:
   no surviving-evidence pixel ever modified by guard, halo, or refill paths.
3. Every automated repair action (regenerated / attempted-rolled-back / skipped-paste)
   disclosed in the Restoration Report with its SFace cosines / containment verdict.
4. All workstreams' thresholds carry measured justification (W8 sweep), and the
   regression pins from §4 run in the standard suite.

## 7. Explicitly out of scope (unchanged from the V plan)

SAM-class segmentation; VLM world-knowledge checks (won't fit the 4 GB / MINIMAL
floor); modifying surviving evidence under any circumstance; auto-applying semantic
re-imagination without the `reimagine` opt-in + preset gates.
