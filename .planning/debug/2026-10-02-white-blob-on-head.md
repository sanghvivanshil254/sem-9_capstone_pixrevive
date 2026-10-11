# Debug Session: White blob on head (teared portrait, run 2026-10-02 00:28)

**Status:** investigating
**Started:** 2026-10-02 ~02:00 (user local)
**Mode:** debug (find_and_fix; no spawn_agent available in session — inline per skill fallback)
**User evidence:** `C:\Users\sangh\Downloads\restored_teared.png` (1341x1341) —
sha256 `ae0bfa9c…` ≡ `backend/temp_results/2ad13cff0652.png` (01:30 run, provenance verified).
Second attached screenshot = frontend view of the same output (report: 10.5% inpainted,
face damage 7.46% ≡ debug_output/08_metadata.json). Same run family as the 00:28
debug_output snapshots (identical input 447x447, identical log numbers).

## Symptoms
- Expected: clean restoration, no white patch on the head.
- Actual: a large white blob on the top of the head; overall quality below par.
- Repro: restore the teared sepia portrait through the backend pipeline.

## Current Focus
- hypothesis: The blob is un-repaired dense dust/emulsion field on the head — no stage
  claims it (Stage 0 blob cap too small, Stage 2 grain gate discards dust with grain,
  tearmark field-mode keeps only the amplitude tail, Stage 11 residue detector only sees
  small compact specks and excludes the face box; blob sits above the face bbox y=204).
- test: .freebuff/probe_white_blob*.py region traces across debug_output snapshots.
- expecting: bright pixels survive in final inside AND outside the inpaint mask.
- next_action: implement source fix (Stage 2 dust-cluster mask enrichment) + Stage 11
  mid-scale dust dissolve outside faces; verify by pipeline re-run + suite.

## Evidence (2026-10-02, probes .freebuff/probe_white_blob{,2,3}.py)
- Blob region (1341-frame x555-790, y35-235 — above face bbox top y=204):
  input bright(>170) share 61.8%; final bright share 24.4% (11,486 px, mean ~210).
- Of final bright px: 51.2% are INSIDE the Stage-2 inpaint mask (LaMa fill, p99=230 —
  context-brightened synthesis), 48.8% are OUTSIDE the mask (survived dust, mean 217).
- Stage-wise masked-subset mean: pre-inpaint 190.4 -> LaMa fill 137.1 -> final 129.6
  (flat downstream; no stage brightens — the blob is preservation, not addition).
- Mask coverage in blob region: tearmark 41.97%, final 60.36% — ~40% of the dust field
  was never masked. Whole-frame mask 10.46% (log: "LaMa: inpainting 21.6% of image"
  refers to dilated LaMa-internal padding; refined mask is authoritative).
- Stage 0 log: speck field 19.6%->15.7% (amplitude 15.9->6.2), impulse 1.57%,
  BlobInpaint removed only 0.27% (538 blobs; per-blob cap ~120px @447 — head dust
  patches are far larger, so they pass through by design).
- Stage 2 log: GrainField gate discarded noise-derived masks (content sigma 7.4 > 6.0,
  damage 3.0% < 6.0) — correct per RC-1, but it also discards object-level dust
  clusters; TearmarkDetector field-mode re-admitted only the amplitude tail (7.28%).
- Stage 11 log: residue 1.21% -> 0.28% (detector: bright vs 11px local mean, small
  compact components, capped at 2.5% frame; faces excluded) — structurally blind to a
  dense bright dust field whose local mean is itself bright.
- Final grade NIQE 5.34; quality_audit flags: hf_retention_low, contour_lost, eye_asymmetry.

## Root Cause (confirmed)
**RC-B (white blob):** dense mid-scale dust clusters on the head belong to no repair
class: Stage 0 BlobInpaint caps blobs at ~0.06% frame area; Stage 2's grain gate
(design intent: protect frames from RC-1) discards the whole noise-derived field and
field-mode TearmarkDetector keeps only the speck amplitude tail; LaMa fills the masked
core with context-brightened fill; Stage 11's residue detector measures vs an 11px
local mean and excludes faces, so a 200x235px bright dust field outside the face bbox
survives to the output. Similar images with dust fields over hair/background hit the
same gap.

## Plan (worst-first, reusable)
1. Fix 1 (source): `EnhancedDamageDetector.detect_dust_clusters()` — mid-scale bright
   compact clusters (top-hat at dust scale, compactness + structure-touch + coverage
   caps, env PIXREVIVE_DUST_CLUSTERS=0 reverts) OR-ed into the Stage-2 mask after the
   grain gate, so LaMa re-synthesizes the region once, at source.
2. Fix 2 (closed loop): output_verifier `detect_dust_field()` — mid-scale dust pass on
   the FINAL canvas, outside faces, capped; dissolves whatever survives any upstream
   path so the class is dead even when detectors disagree (PIXREVIVE_DUST_FIELD=0).
3. Unit tests (tests/test_dust_field_fixes.py): dust cluster masked; strand/collar NOT
   flagged; clean frame no-op; revert flags honored.
4. Verify: full pipeline re-run on the same input via backend API; probe re-measure
   (blob-region bright %, mask coverage, NIQE); pytest suite stays green.

## Eliminated
- hypothesis: a late stage (colorize/GFPGAN/SR/sharpen/grade) ADDS the blob —
  eliminated: masked-subset and whole-region means are flat 137->130 downstream, and
  bright px counts never grow before SR (9x scale factor only).
- hypothesis: LaMa fill is white — eliminated: fill mean 137 vs survived dust 217;
  LaMa IMPROVED the masked core (190->137). The blob is 49% survived dust.
- hypothesis: stale backend/snapshots mismatch — eliminated: sha256 provenance + log
  numbers match the frontend report.
