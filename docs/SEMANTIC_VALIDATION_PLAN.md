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
