# Concerns & Watch Items

**Analysis Date:** 2026-09-27 (full refresh; previous map 2026-09-19)

Status framing: the 2026-09-19 concerns (false-positive crack masking, jawline shadow,
over-denoising, scanner margins, concurrency) remain **resolved** and are still pinned by
`tests/test_restoration_regressions.py` — not repeated here. This pass adds what changed
since, with current evidence.

## Active Watch Items

**1. Host ceiling is the architecture's hard constraint (structural, managed).**
7.3 GB RAM + GTX 1650 4 GB: fp16 diffusion is NaN-broken on this torch/GPU (documented),
SDXL cannot run, and the 2.2 GB MINIMAL RAM floor means the backend must be paused for
full-suite runs (torch/pytest coexistence segfaults). Everything is tiered/measured
around it (`resource_profiler.py`), but any feature that adds resident models must go
through the Stage-12-style **release/reload bridge** or it will never clear the floors.

**2. Identity on hopeless coverage is unsolved by design (honest negatives recorded).**
Measured: seed variance (±0.17 SFace cosine) dominates IP-Adapter scale effects (±0.03),
and FaceID conditioning — while mechanically verified (ArcFace self-retrieval 0.683) —
scored *worse* on the 60%-destroy fixture (0.095/0.098 vs CLIP 0.233) because the
reference embedding is computed from destroyed pixels. The gate's rollback + disclosure
is the correct behavior; FaceID's value case (moderate damage, meaningful reference
face) is plausible but **not yet benchmarked** — see "Open questions".

**3. Detector taxonomy coverage vs synthetic damage (disclosed limitation).**
The trained detectors legitimately ignore damage outside their taxonomy: a 98%-altered
synthetic blotch probe measured `frame_pct 0.0` at Stage 12 (blotches read as
highlights, not damage). The skip reason now says so honestly. Implication: the hybrid
beta fires on *taxonomy* damage (tears, flakes, foxing, mold), not arbitrary destruction.
If wider recall is wanted, it's a detector-training scope decision, not a Stage-12 bug.

**4. Two CWD/anchor bug classes have bitten twice — keep the pattern.**
CWD-relative paths broke weights resolution and the gate inside the uvicorn server
(cwd=`backend/`), and post-offload attention-processor swaps stranded weights on CPU.
Any new module that touches weights must anchor to `__file__`; any new conditioning must
be fitted before `enable_sequential_cpu_offload()` or explicitly re-hooked.

**5. Commit hygiene is good but the tree is ahead of the remote (ops).**
`main` = 6d2d479 (+ tag v1.1.0), **no remote configured** — single-machine risk on
~7.5k lines of unpushed history. Also: `upcoming.txt` (user notes) intentionally
untracked; `e2e/` Playwright suite cannot download browsers (CDP-attached Chrome only).

## Open Questions (deliberately not scheduled)

- Does FaceID beat CLIP on *moderate* damage (meaningful reference face)? A benchmark
  across the fixture set (FaceID on/off) would validate the H4 value case.
- SDXL path is code-complete but unverifiable here — needs a STANDARD/AMPLE host.
- ControlNet-inpaint (STANDARD/AMPLE) and AMPLE batch mode remain deferred by design.

## Failure Modes to Watch in Production

- Stage-12 error branch: now logs traceback + surfaces `status:"error"` in the report —
  if a report ever shows an empty reimagine block with `ran:true`, that's a regression.
- Model reload after the VRAM bridge: a failed reload degrades the *next* request and
  shows in `/api/health` `load_errors` — monitor that field after hybrid runs.
- SSE consumers must treat only `complete`/`error` as terminal (worker emits progress
  via `call_soon_threadsafe`; ordering is otherwise strict).
