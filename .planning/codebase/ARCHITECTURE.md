# Architecture

**Analysis Date:** 2026-09-27 (full refresh; previous map 2026-09-19)

## System Shape

Three-process local app: **React 19 SPA** (Vite, :5173) → **FastAPI** (:8000, REST + SSE)
→ **16-stage in-process restoration pipeline** with GPU/CPU model management. No message
broker, no DB — state is per-request `PipelineContext`; admin config is file-backed.

## Request Flow

```
Client → image_router (POST /api/process | /api/process-stream, multipart form)
       → ImageService.process_image()          (task id, semaphore acquire)
       → RestorationPipeline.restore()/process() (PipelineContext)
       → registry-dispatched stages in fixed order
       → restoration_report built from metadata → SSE `complete` / PNG bytes
```

GPU concurrency: one `asyncio.Semaphore(1)` in `image_router` serializes inference per
process; **stages must never re-acquire it** (nested acquire = deadlock; verified none).

## Stage Graph (`backend/ai/pipeline/stage_registry.py`, order is contractual)

| # | id | pct | Purpose / owner module |
|---|---|---|---|
| 0 | loading | 0 | model preload (skipped when warm) |
| 1 | deskew | 1 | auto skew correction |
| 2 | flat_field | 2 | illumination flattening |
| 3 | despeckle | 3 | impulse-noise suppression |
| 4 | moire | 4 | halftone/moiré suppression |
| 5 | analyze | 10 | degradation profile → `image_metadata` |
| 6 | face_detect | 11 | RetinaFace/YuNet → `metadata["faces"]` |
| 7 | cast_fix | 12 | color-cast rebalance |
| 8 | damage_detect | 20 | taxonomy detectors → `damage_masks["final"]` |
| 9 | inpaint | 30 | LaMa + seam smoothing + face-wipeout circuit breaker |
| 10 | damage_cluster_smooth | 35 | mask refinement post-inpaint |
| 11 | refresh_faces | 0 | re-detect post-inpaint |
| 12 | denoise | 50 | NLM with edge/variance gating (T-3 deblur gate: `deblur` stage fires only when blur detected) |
| 13 | colorize | 65 | DDColor (records `colorization_applied` evidence flag) |
| 14 | face_restore | 78 | GFPGAN + chin-fade anatomy mask |
| 15 | pre_sr → super_res | 85–90 | Real-ESRGAN ×4 |
| 16 | sharpen | 95 | border enhancement |
| 17 | projector_grade | 98 | exhibition grading |
| 18 | output_verify | 99 | closed-loop residue/color/texture checks |
| 19 | **reimagine** | 99 | **Stage 12 — hybrid tier (opt-in)** |

Stage methods live in `backend/ai/pipeline/stages/` (prep/color/damage/finish stage
mixins); dispatch is data-driven from the registry (`route_gate` falls through to the
user's pipeline flag — `enable_reimagine`).

## The Hybrid Re-Imagine Tier (Stage 12 — new since baseline)

`backend/ai/pipeline/stages/finish_stages.py::_stage_12_reimagine`:

1. **Coverage gate** — hopeless = ≥35% frame or ≥55% worst-face Destroy coverage in
   `damage_masks["final"]`. Distinguishes "no unrecoverable damage detected" (detector
   found nothing — disclosed honestly) from "below the bar" (LaMa already healed it).
2. **VRAM bridge** — releases standard models (LaMa/GFPGAN/RealESRGAN/DDColor/RetinaFace/
   U-Net via `unload()` + `_reload_models_on`) before the engine profiles resources;
   restores them in `finally`. This is what makes Stage 12 *possible in-process*
   (0.94 → 2.88+ GB free, measured).
3. **Engine** (`backend/ai/restoration/reimaginer.py`) — resource-tiered SD inpaint with:
   evidence composite (bytewise-asserted — every surviving pixel returns identical),
   feathered tiling, NaN self-heal, SFace identity gate with bounded reroll ladder.
4. **Evidence overlay** — on success, a colorized zone map
   (`{task_id}_evidence.png`, green = original / magenta = re-imagined) is persisted and
   served at `/api/process/{task_id}/evidence` ("Show what's real", design §7).
5. **Disclosure** — full record into `metadata["reimagine"]` → report block → UI card.

## Identity Conditioning (H4)

Priority chain inside the re-imagine engine: **FaceID** (ArcFace 512-d embedding of the
surviving face, projected to 4 cross-attention tokens concatenated to text embeds) →
**CLIP IP-Adapter** (image-crop reference) → none (unconditioned). FaceID is fitted once
at pipeline load, *before* the CPU-offload snapshot (`encoder_hid_dim_type` marker cleared
— the CLIP loader's image-embeds requirement otherwise crashes the FaceID path); faceless
images run null identity tokens (`proj(zeros)`), never adapter surgery at runtime.
The SFace gate (`backend/ai/validation/face_identity_gate.py`, worst-cosine across ALL
detected faces) remains the border control regardless of conditioning signal.

## Invariants (unchanged, enforced by tests)

- Evidence pixels are **bytewise identical** after a hybrid run (`_composite_evidence`
  assertion + post-composite re-check in the stage).
- The gate can only *reject* hybrid output, never silently pass it: undetectable
  similarity → flag (kept, disclosed), < flag floor → rollback → standard result ships.
- Any adapter state is two-way: adapter installed ⇔ matching embeds supplied
  (the `'NoneType' is not iterable` crash class, measured 2026-09-27).
- Stage 12 errors are logged with traceback AND surfaced as `status: "error"` in the
  report — nothing swallows.
