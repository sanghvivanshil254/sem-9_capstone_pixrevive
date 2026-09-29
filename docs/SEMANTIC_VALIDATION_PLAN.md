# Phase V — Evidence-Conditioned Semantic Validation & Targeted Re-Imagination

**Status:** PLAN (approved direction, not yet implemented)
**Date:** 2026-09-28
**Origin:** user core concept — *"every image must pass an AI validator: a shirt has a
collar on both sides; if hair is lost to flakes, regenerate it; a human has two eyes;
teeth are white; buttons contrast with the cloth"* — promoted to an engineering plan.
**Prerequisite:** Phase H (hybrid tier) — shipped, v1.1.0.

---

## 0. Prime Directive (the one rule everything follows)

> **The validator may never add what the evidence cannot support.
> It may only complete what the evidence proves was there.**

Every rule in this document is *evidence-conditioned*: enforced only where the
evidence mask (`original vs re-imagined`, Phase H) can adjudicate the claim.
Where no evidence survives, completion is permitted — and **disclosed as an
invention**. Where evidence survives, it is ground truth and outranks any
generic world prior. A one-eyed man in the evidence stays one-eyed.

This directive is what separates *restoration* from *falsification*, and it is
non-negotiable. Any rule that cannot be evidence-conditioned is deferred or
rejected.

---

## 1. Concept: the missing validation layer

The current pipeline validates three things:

| Layer | Stage | Validates | Misses |
|---|---|---|---|
| Pixels | Stage 11 output_verify | residue, color coherence, texture | meaning |
| Identity | SFace gate (H1) | same person | structure/anatomy of the scene |
| **Semantics** | **nothing — Phase V** | **"is this a coherent depiction of a plausible world?"** | — |

Phase V adds the semantic layer, and closes the loop: **validator findings become
generation targets.** A finding ("hair region implausible") is not just reported —
it compiles into a targeted mask, drives a bounded, disclosed re-imagination of
exactly that region, and the result is re-validated. Validate → mask → re-imagine →
re-validate. The hybrid tier's machinery (evidence composite, SFace rollback,
resource tiers, evidence overlay, report disclosure) is reused unchanged.

### The rule taxonomy (from the user's core idea, generalized)

| Class | Examples (user's) | Generalized rule | Enforcement |
|---|---|---|---|
| **A. Anatomy inventory** | "hooman should have 2 eyes (if not blind/injured)", ears, teeth | Face-region parts exist / are symmetric / are complete — **only where the evidence mask says the region was destroyed or ambiguous** | detect via YuNet landmarks + evidence mask; violations become re-imagine targets |
| **B. Anatomy appearance** | "teeth must be white" | Surviving-evidence appearance is truth; destroyed-region appearance follows photo-world priors (era, lighting) | CLIP/region priors, evidence-conditioned |
| **C. Garment continuity** | "shirt has a collar on both sides", "buttons according to cloth contrast" | Bilateral symmetry + continuation of structures across the evidence boundary: a surviving collar implies its mirror; surviving button implies matching set | symmetry/continuity checks across the evidence mask |
| **D. Accessory coherence** | "same for other accessories" | Objects present in evidence continue plausibly; new objects never appear | conservative: only continuation, no invention |

**Universal veto:** a rule that fires on *surviving evidence* pixels is a validator
bug, not a finding. Findings may only target re-imagine-capable zones
(`evidence_mask == destroyed`) or the boundary band where continuation is
uncertain — and any finding that would alter evidence pixels aborts.

---

## 2. Architecture (fits the existing pipeline — no new guardrails needed)

```
Stage 18 output_verify (pixel layer, existing)
        │
        ▼
Stage 19 semantic_validate   ← NEW (V-phase core)
   ├─ runs AFTER all standard repair, BEFORE/alongside Stage 12 gate
   ├─ inputs: working image, evidence mask (pre/post-repair Destroy zones),
   │          faces+landmarks (existing), verifier flags (existing),
   │          metadata (noise_score, damage summary)
   ├─ runs rule battery (per class A–D), each rule emits:
   │     {rule_id, region_mask, severity, evidence_state, finding}
   ├─ findings with evidence_state == "destroyed|ambiguous" → compile
   │     targeted masks → feed Stage 12 re-imagine as EXTRA masks
   └─ all findings → report block (disclosure, additive JSON)
        │
        ▼
Stage 12 re-imagine (existing engine, unchanged)
   ├─ evidence composite (bytewise) — unchanged
   ├─ SFace identity gate + rollback — unchanged
   └─ evidence overlay: now marks semantic zones distinctly (e.g. cyan =
        "validator-completed region") — small overlay upgrade
        │
        ▼
Re-validate (bounded loop, max 2 passes — reuse the reroll pattern)
```

Key wiring facts (already true in the codebase):
- Stage 12's engine accepts masks; semantic masks merge with the coverage mask.
- The report builder is additive; `semantic` block slots beside `reimagine`.
- The evidence overlay is already persisted and served; one more color class.
- Rule findings are metadata → zero risk to the canvas when rules run in
  report-only mode (V1 ships report-only first — see rollout).

---

## 3. The rule battery (each rule = detector + evidence-conditioner + target-masker)

### Class A — Anatomy inventory (face regions)

| Rule | Detector | Evidence conditioning |
|---|---|---|
| `A1.eye_presence` | YuNet 5-pt landmarks: eye centers inside face bbox; symmetry (interocular line angle vs roll); CLIP crop-score "a close-up photo of an eye" vs "skin" | fires only if the eye quadrant is ≥X% destroyed/ambiguous in evidence; if evidence shows the eye → PASS forever |
| `A2.eye_pair_symmetry` | both eye quadrants same evidence-state; if one eye visible + other destroyed → mirror-constrained completion (mirror the surviving eye's geometry as diffusion conditioning hint) | mirror-hint only, pixels still generated by SD + SFace-gated |
| `A3.hairline_continuity` | hair-region mask (above face oval, evidence-state split); continuity of intensity/texture across evidence boundary | fires only where hair region destroyed; survives-implies-bald rule: if evidence hair region is thin/absent AND surviving → bald is truth, no regeneration |
| `A4.mouth_teeth` | landmark mouth box; "teeth must be white" = brightness prior **only if** mouth region was destroyed and smile evidence exists elsewhere (e.g. surviving lower lip); never whitens surviving teeth | brightness prior is a *hint weight*, never a pixel override |

### Class B — Appearance priors (era-aware, hint-level)

| Rule | Detector | Conditioning |
|---|---|---|
| `B1.skin_tone_continuity` | face/hands vs surviving skin patches; LAB ΔE across evidence boundary | continuity to evidence; no global "beautification" |
| `B2.paper_neutrality` | destroyed border zones follow the photo's surviving paper tone | prevents the "DDColor invents blue border" flag from recurring (teared.jpg measured Δab 27) |

### Class C — Garment continuity (the "collar" rules)

| Rule | Detector | Conditioning |
|---|---|---|
| `C1.bilateral_structure` | detect collar/lapel/button-line candidates on the surviving side (Hough/contour + CLIP crop verification); mirror-check across the torso axis **where the mirror zone is destroyed** | fires only on destroyed mirror zone; surviving asymmetry (one-sided collar styles exist) → PASS with note |
| `C2.button_contrast_set` | detect surviving button (blob + ΔL vs local cloth ≥ threshold); predict expected count/spacing from surviving run; check destroyed continuation | contrast rule applied only between *evidence* button and *evidence* cloth; generated buttons inherit from the surviving set |

### Class D — Accessory coherence (conservative continuation only)

| Rule | Detector | Conditioning |
|---|---|---|
| `D1.object_continuation` | surviving object at evidence boundary (glasses arm, necklace, uniform strap) continues into destroyed zone | strictly continuation; a rule may never introduce an object not present in evidence |

### Honest scope limits (recorded, not hidden)

- **No rule checks world facts the pixels can't adjudicate** (military insignia
  correctness, historical garment accuracy) — deferred to a future VLM tier on
  capable hosts; recorded as out of scope for MINIMAL.
- **Symmetry is a prior, not a law** — one-sided collar styles, turned heads,
  profiles. All Class-C rules PASS-with-note when surviving evidence shows
  asymmetry. The battery never "corrects" surviving asymmetry.
- **Faces are special**: any A-class re-imagination goes through the existing
  SFace identity gate with the *same* thresholds; the gate can always roll back.

---

## 4. Phased rollout (each phase independently shippable, report-only first)

### V0 — Semantic report-only battery (no generation) `~2–3 sessions`
- Implement rule runner + rules A1/A3/C1 + evidence-state classification.
- **Everything report-only**: findings appear in the report as a `semantic`
  block (`findings: [{rule, region, state, verdict, action:"none"}]`).
- **Gate**: run across `tests/restoration_dataset/` (all fixtures) +
  teared.jpg + case9; measure false-positive rate on pristine set — must be 0
  false "regenerate" findings on pristine images.
- Unit tests: rule logic with synthetic evidence states; report block schema.
- *Exit criteria: ≤1% FP on pristine fixtures, every finding human-auditable.*

### V1 — Face anatomy validator → targeted re-imagine (the core loop) `~3–4 sessions`
- A1–A4 go active: findings compile into masks; merge into Stage 12 input;
  bounded 2-pass re-validate loop; overlay marks semantic zones (cyan).
- Gate: case9 (face 60%-destroyed synthetic) and teared.jpg live runs; SFace
  verdicts recorded; no evidence pixel ever altered (assert in tests).
- *Exit criteria: live hybrid run where a validator finding causes a targeted
  completion that passes the identity gate and is disclosed in the report.*

#### V1 addendum — design from measured V0 data (2026-09-29)

V0's numbers reshape V1. What the corpus and live runs actually showed:

| Measured signal | Count | V1 consequence |
|---|---|---|
| Gate A pristine FP | **0 / 17** | Precondition for activating any rule is met |
| A3 regeneration candidates | 2/17 damaged fixtures + teared live (texture_std 47.2 raw / 30.3 post-fill) | The only finding class wired to an action in V1 |
| A1 DESTROYED w/ residual structure | 4 (case3/case9 eyes, raw battery) | Stays report-only in V1 — residual structure is evidence to preserve, not overwrite |
| A1 AMBIGUOUS | 17 (all `texture_std=-1`, structure proxy N/A) | Never triggers generation until calibrated (V3 sweep); pure notes |
| **Live teared.jpg Stage 12** | **skipped** — frame 32.96% < 35% bar, worst face 25.15% < 55% bar; LaMa filled the 33% | **The pivot**: V1 cannot assume Stage 12 ran. Semantic completion must apply to the LaMa canvas too |

The decisive live-path fact: teared.jpg's hair band is genuinely lost evidence that
the whole-frame Stage-12 gate will *never* touch on this class of image, yet Stage 13
still classifies it DESTROYED and A3 still calls it a regeneration candidate on the
output canvas. Region-scoped completion is the only mechanism that can act on it.

Design (per slice, each independently shippable):

- **V1a — findings→mask plumbing (no generation).** Compiler turns
  action="regeneration candidate" findings into region masks: finding band ∩
  evidence>0, dilated, clipped to the face bbox neighborhood. Stage 13 emits a
  `semantic_actionable` list alongside the report block. Unit tests assert mask
  geometry against the bbox anchors on fixtures. *Exit: masks verified, zero
  generation code, Gate A still 0 FP.*
- **V1b — LaMa region completion (no diffusion).** When candidates exist and the
  `reimagine` opt-in is set, re-fill **only** the candidate masks with the existing
  LaMa path (seconds, no new weights, no resource-tier drama), then:
  evidence composite re-asserted bytewise outside the mask; SFaceIdentityGate
  pre/post cosine — **delta-gated** (post ≥ pre − margin, not absolute 0.42, because
  H4 measured that destroyed-face references can be noise-dominated);
  pixel-quality guard reverts the region if SSIM inside the band got worse.
  Report `semantic` findings gain `action: regenerated | attempted-rolled-back |
  candidate-reported` with SFace cosines. Overlay marks semantic zones cyan.
  *Exit: teared.jpg live run with the hair band completed, identity delta pass,
  Gate A rerun still 0 FP.*
- **V1c — diffusion region completion.** Route candidates through the Stage-12
  engine region-cropped (SD1.5 inpaint + FaceID conditioning + reroll ladder) only
  when the resource tier ≥ MINIMAL; LaMa result is the always-available floor.
  *Exit per original plan: case9 + teared live, SFace verdicts recorded.*

Bounded loop: one semantic pass in V1 (the plan's 2-pass loop is a V3 calibration
question). If the finding persists with unchanged evidence state after the pass,
accept and disclose "attempted, unresolved" — the prime directive forbids endless
retrying against evidence that isn't there.

Risks specific to the measured data: LaMa's frame-wide fill may beat a region-only
refill inside the band (context argument) — hence the SSIM revert guard and honest
`attempted-rolled-back` disclosure; textured headwear above a face can mimic the A3
signature (corpus has none — add a headwear fixture to the V3 sweep before V1c);
SFace absolute thresholds may be unachievable at heavy face destruction (H4 measured
0.095 on 60%-destroyed references) — hence delta-gating in V1b, absolute gates
revisited only in V1c with reroll headroom.

Pre-registered no-go triggers: any Gate A regression from mask dilation; SFace
delta-gate failing to separate refill quality across ≥3 real runs; region refill
SSIM-worse than the pre-pass canvas on the corpus majority. Any of these → stop
before V1c, recalibrate V3-style (fixed seed, one variable), re-gate.

### V2 — Garment & continuity rules `~2–3 sessions`
- C1/C2 + B1/B2 active; mirror-constrained conditioning for C1 (surviving-side
  crop as IP reference — the existing adapter path, no new weights).
- *Exit criteria: collar/button continuation demonstrated on a fixture with a
  destroyed torso zone; paper_neutrality kills the Δab-27 border flag class.*

### V3 — Calibration, thresholds, docs `~1–2 sessions`
- Per-rule thresholds swept on the fixture corpus (the H3 sweep harness
  pattern: fixed seed, fresh process, one variable).
- DESIGN doc §13 → full measured addendum; EXECUTION_PLAN + CHANGELOG.
- *Exit criteria: thresholds have measured justification; no untested priors.*

**Explicitly out of scope (recorded):** SAM-class segmentation and VLM
world-knowledge checking (won't fit 4 GB / MINIMAL floor); modifying surviving
evidence under any circumstance; auto-applying rules to non-opted-in users —
semantic re-imagination stays behind the same `reimagine` opt-in + preset as H.

---

## 5. Engineering constraints & reuse map

| Need | Already exists | Reuse path |
|---|---|---|
| "Is this region destroyed?" | evidence mask (H composite) | pass `mask==0` zones + verifier flags into the validator |
| Face parts + landmarks | YuNet in gate + embedder | same detector, same weights, zero additions |
| Plausibility scoring | CLIP ViT-H encoder on disk (IP-Adapter's, 2.5 GB) | load once in the validator; zero-shot crop prompts; no downloads |
| Targeted generation | Stage 12 engine (masks, evidence composite, SFace, overlay) | merge semantic masks; engine unchanged |
| Bounded retries | reroll ladder pattern | 2-pass validate loop |
| Disclosure | report builder + frontend card | additive `semantic` block + card section |
| Opt-in safety | `reimagine` flag / `reimagine_beta` preset | same gates |

**Hardware honesty:** CLIP crop-scoring at 224px on CPU ≈ tens of ms/crop —
fine. The binding constraint stays Stage-12 generation time (~3 min/25-step on
this box); V-rules *reduce* wasted generations by targeting only implausible
zones instead of whole-frame masks.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| False-positive rule "regenerates" healthy anatomy | V0 report-only gate with 0-FP exit criterion on pristine corpus; findings only ever target destroyed/ambiguous zones |
| Rules fight each other (hair vs collar overlap) | rule executor merges masks with priority A > C > B; conflicting findings downgrade to report-only |
| Identity drift from anatomical completion | existing SFace gate with unchanged thresholds; A-class completions get max rerolls, rollback = standard result ships |
| Validator over-trust (automation bias) | every finding carries evidence_state + rule_id + region preview in the report; user can always see *why* |
| Scope creep into beautification | Prime Directive in code as an assertion: findings touching `evidence_mask==original` pixels raise and abort |

## 7. Definition of done (for the whole phase)

1. A photo with a destroyed eye region gets a *disclosed*, identity-gated
   completion; a photo with a surviving one-eyed face is never touched.
2. Pristine corpus: zero semantic findings with action != none.
3. teared.jpg class (border-wash, hair gaps) resolved or disclosed by rule B2/A3.
4. All thresholds measured (V3), docs updated, full suite green, live-verified
   through the API per the H-phase discipline.
