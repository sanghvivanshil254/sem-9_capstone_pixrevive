# Hybrid Evidence-Preserving Diffusion — Design v2

**Status:** design only — nothing here is implemented yet.
**v2 changes:** (1) resource budgeting is now **dynamic and tiered** — the fixed
"4 GB GTX 1650" assumption is replaced by runtime profiling (`resource_profiler.py`),
closing the loophole where the stage's precheck and offload strategy were hardcoded
to one dev box. (2) The model stack is upgraded to **SDXL inpainting + IP-Adapter
(SDXL) + ControlNet-inpaint + SFace identity metric**, made feasible on small cards
via **chunked (tiled) latent generation**.

---

## 1. Mission and the core contract (unchanged)

For photos whose damage is so extensive that any "repair" is fabrication anyway,
offer a diffusion-based **re-imagination of the destroyed regions only** — while
the output remains a *restoration*, not an illustration.

**The contract:** no surviving pixel is ever substituted.

- **Evidence (E)** — original pixels that survived → composited back
  *unconditionally* at full resolution.
- **Destroy (M)** — regions the detectors proved are gone → the only place
  diffusion may paint.

`out[E] == original[E]` is enforced **bytewise** and asserted by Stage 11
(`output_verifier`) on every hybrid run. A failed assertion is a hard error, not
a warning. This is what separates the hybrid from a full redraw (rated 4/10 as a
replacement — beautiful output, wrong person, no evidentiary value).

## 2. Pipeline position and gate conditions (unchanged)

```
... damage_detect → LaMa inpaint → seams → ★ 3d REIMAGINE (opt-in) → denoise → colorize → ...
```

Stage 3d fires only when **all three** hold:

1. User opted in (`enable_reimagine`, a new toggle; "Re-imagine (beta)" preset).
2. Coverage is hopeless: **≥ 35 %** of the frame, or **≥ 55 %** of the detected
   face, is Destroy-class damage (tiers the damage detectors already compute).
3. The resource profiler selects a runnable tier (§4) and weights are present.

Otherwise behavior is **byte-identical to today**, and the report carries an
offer line: *"output is largely reconstructed — the optional re-imagining tier
may produce a more coherent result."* No GPU is never a crash: it is the same
graceful skip, just tier-aware (§4.3).

## 3. Model stack — SDXL by chunking (v2)

| Role | Pick | fp16 size | Why |
|---|---|---|---|
| Generator | **SDXL 1.0 inpainting** (`diffusers StableDiffusionXLInpaintPipeline`) | ~7.0 GB (UNet 5.2 + 2 text encoders 1.6 + VAE 0.2) | Real fidelity jump over SD1.5; native mask-aware UNet; fills 1024-px canvases |
| Identity conditioning | **IP-Adapter (SDXL variant, CLIP image-emb)** | encoder ~2.0 GB; adapter ~0.1 GB | Conditions generation on the input's surviving face; no insightface dependency. Run the encoder **once**, cache the embedding, unload the encoder before generation (§5) |
| Structure control | **ControlNet-inpaint (SDXL)** | ~5.1 GB | Conditions directly on the masked image — closer conceptual fit than an edge-map ControlNet and cheaper than stacking two. Loaded only on STANDARD/AMPLE (§4); on MINIMAL its VRAM floor exceeds what chunking can guarantee, so it is skipped by tier, not by luck |
| Identity metric | **SFace via `cv2.FaceRecognizerSF`** (+ YuNet detector) | 38 MB ONNX | Meaningfully more accurate than CLIP-cosine — and SDXL drifts identity further than SD1.5 did, so the metric must be stronger. Runs on OpenCV DNN (no onnxruntime). **Zero new installs:** `opencv-contrib-python==4.12.0.88` is already pinned and ships `cv2.FaceRecognizerSF` |

**Weights** download via `urllib` into `backend/ai/weights/`
(`sdxl_inpaint/`, `ip_adapter/`, `sface/`) through `download_weights.py`,
per project rules (never a hub cache). Wrapper module
`ai/restoration/reimaginer.py` follows the existing lazy `deblurrer.py` pattern.
License note: SDXL base is OpenRAIL-M — recorded in the weights README.

## 4. Dynamic resource profiling (v2 — the loophole fix)

### 4.1 New module `ai/restoration/resource_profiler.py`

```python
detect_gpu_vram_free_gb() -> tuple[float, float] | None   # (free, total); None if no CUDA
detect_system_ram_free_gb() -> float                      # psutil.virtual_memory().available
select_resource_tier(vram_free_gb, ram_free_gb) -> ResourceTier
```

- `torch.cuda.mem_get_info()` — reports what CUDA actually sees **free right
  now**, not device spec; "total minus current usage" is what matters on
  shared/variable hosts. Profiling happens **at first gate-fire** (lazy-load
  time), *not* at import time — specs change between process start and first
  use. Re-profiled per invocation (not cached across runs).
- `psutil` (new dependency, pinned `psutil==6.1.x`) — CPU-offloaded weights
  live in host RAM, so VRAM alone cannot predict feasibility.
- Both detectors are wrapped so a probe failure degrades to "skip gracefully",
  never to a crash.

### 4.2 Tier table (framed as **minimums at gate-fire time**, not "the" config)

| Tier | Free VRAM | Free RAM (min) | Generation config | SDXL feasibility |
|---|---|---|---|---|
| `NO_GPU` | none | — | Stage never runs; report: *"re-imagine unavailable: no GPU detected"* | — |
| `MINIMAL` | ≥ 3.0 GB | ≥ 8 GB | **Full chunking kit:** sequential CPU-offload + attention-slicing + tiled VAE + **tiled latent inpainting (704-px tiles, 128-px feathered overlap)** + IP-Adapter (embed-once-then-unload). ControlNet-inpaint **off** (its 5.1 GB floor exceeds what chunking can guarantee) | ~2.6–3.2 GB peak per tile-pass. **The current 4 GB box lands here** — it is the floor, not the assumption |
| `STANDARD` | ≥ 8.0 GB | ≥ 10 GB | Model-level CPU-offload (one component resident at a time), attention-slicing off, tiled VAE optional, tiles ≤ 1024 px, **ControlNet-inpaint on** | ~6.5–7.5 GB peak |
| `AMPLE` | ≥ 12 GB | ≥ 8 GB | No offload, full-res 1024-px single pass, batch-friendly | ~9–10 GB peak |

**RAM step-down rule:** if VRAM qualifies for a tier but free RAM fails that
tier's minimum, **step down a tier** (host-side OOM is a crash; slower is not).
A `MINIMAL` tier that still fails its own minimums (VRAM < 3 GB or RAM < 8 GB)
is not runnable → graceful skip.

**Manual override:** `REIMAGINE_FORCE_TIER = minimal | standard | ample | off`.
Pins a host explicitly (shared/variable hosts, reproducible benchmarks).
Invalid values log a warning and fall back to autodetect; `off` always skips.
Override does **not** waive the RAM minimums — a forced tier that fails its own
minimum still steps down/skips, with the reason recorded.

### 4.3 Wiring into `reimaginer.py`

- The static "`mem_get_info() >= 3 GB` free" precheck is **replaced** by
  `select_resource_tier()` at lazy-load time.
- `NO_GPU` and any non-runnable tier → the existing graceful skip, same report
  line, now tier-aware: *"re-imagine unavailable: <tier + detected numbers>"*.
- The SD pipeline (offload mode, attention slicing, tiled VAE, tile size) is
  configured **from the selected tier's settings**, not a fixed setup.
- Unchanged discipline: lazy-load on first gate-fire, **explicit unload after
  the stage** (DDColor/GFPGAN later in the pipeline must not fight it), the GPU
  semaphore serialization stays exactly as is.

### 4.4 Report disclosure (v2 addition)

```json
"resource": {
  "tier": "minimal",
  "vram_free_gb": 3.6, "vram_total_gb": 4.0,
  "ram_free_gb": 11.2,
  "offload": "sequential", "attention_slicing": true,
  "tiled_vae": true, "tiled_latent_px": 704,
  "override": null
}
```

A hosted run's behavior is auditable after the fact: which tier ran, on what
detected headroom, and whether an override was in force. Skips record the same
block with `"tier": "no_gpu"` (or the failed minimum) plus the offer line.

### 4.5 Measured on the reference box (2026-09-26) — supersedes the estimates above

The benchmark harness (`scripts/bench_reimagine.py`, results in
`test_artifacts/bench_reimagine.json` + `bench_probe.json`) replaced the
estimated ledger above with measurements on the GTX 1650 / torch 2.13.0+cu126
reference box. Three findings change the plan materially:

1. **fp16 diffusion is NaN-broken on this torch/GPU class.** SD1.5-inpaint
   emits an all-black frame under fp16 in *every* configuration probed —
   sequential/model/no offload, PNDM and DDIM schedulers, and with
   `allow_fp16_reduced_precision_reduction=False`. fp32 is clean. The probe
   matrix that isolated this is preserved in `bench_probe.json`.
2. **Sequential CPU-offload is both the smallest AND the fastest fp32
   config** (0.76 GB peak, 9.0 s/step) — fully-resident fp32 needs 4.76 GB,
   spills to shared memory, and runs 4× slower (36.7 s/step). The design's
   offload bet is confirmed by measurement.
3. **SDXL cannot run on THIS box** (fp16 unhealthy per finding 1; fp32
   weights need ~7 GB host RAM against ~5.3 GB free) — the documented SD1.5
   fallback (section 5) becomes the measured default for MINIMAL. Whether
   fp16 SDXL is healthy on modern cards is an H3 calibration item.

| Config (25 steps @ 512px) | Time | Peak VRAM | Output |
|---|---|---|---|
| fp16, sequential offload | 144 s | 0.61 GB | ❌ 100% black (NaN) |
| **fp32, sequential offload** | **165 s** | **0.76 GB** | ✅ clean |
| fp32, fully resident | 367 s | 4.76 GB (spills) | ✅ slow |

Tier table amendments: `MINIMAL` now pins `dtype="fp32"`, `use_sdxl=False`
(encoded in `ResourceTier`; `REIMAGINE_FORCE_TIER` override semantics
unchanged). Runtime NaN self-healing in `reimaginer.py` stays as the safety
net for unmeasured hosts — a poisoned frame retries at seed+1, degrades to
SD1.5 on SDXL tiers, and reports a skip rather than shipping black pixels.

## 5. Chunked generation on small cards (how SDXL fits 4 GB)

1. **Embed-once:** run the IP-Adapter image encoder on the surviving face crop,
   cache the embedding tensor (a few KB), **unload the 2 GB encoder** before
   any generation pass.
2. **Tiled latent inpainting:** split M into 704-px tiles (128-px overlap,
   feathered alpha composite in pixel space after decode). Each tile-pass only
   pays activations for its own tile; the UNet's weights live in RAM via
   sequential offload and stream in layer-by-layer.
3. **Tiled VAE** decode (the standard 4 GB escape hatch) to avoid the 1024-px
   VAE activation spike.
4. **Seam discipline:** evidence pixels are composited back at full resolution
   *after* generation, so tiles can never corrupt E; tiles only compete with
   each other inside M, and the overlap-feather hides latent-space seams.
5. **Cost:** est. 4–8 min/image on a GTX 1650-class card (offload trades speed
   for VRAM). Disclosed in the UI tooltip; unacceptable for batch, which is why
   the tier is opt-in beta. Seed fixed from task id → reproducible.

**Honest limits of chunking:** tiles can drift in global tone; the identity
gate (§6) and Stage 11 verifier check the **composite**, never trust a tile.
If chunked SDXL proves too slow or too seamy on MINIMAL in practice, the
documented fallback is SD1.5-inpaint on MINIMAL only (config already planned,
zero code cost — it is the same pipeline object with a different weights dir).

## 6. Identity gate — SFace (v2)

`cv2.FaceRecognizerSF` cosine between the input's surviving face crop and the
output's face crop (YuNet detects both; same alignment path):

- **≥ 0.42** → pass, recorded in the report
- **0.36–0.42** → keep but flag *"identity drifted (0.38)"*
- **< 0.36** → **rollback Stage 3d entirely** (canvas returns to the LaMa
  result) after at most 2 re-rolls with seed+1

Thresholds are **calibrated on the case8–12 fixtures during H1** before they
freeze (SFace same-person cosines typically land 0.4–0.7; 0.42 is the starting
prior, not a dogma). The existing landmark/SSIM `IdentityValidator` keeps
running as today; SFace is the new active border control.

## 7. Restoration Report — evidence disclosure (unchanged from v1, plus §4.4)

```json
"mode": "hybrid-restoration",
"evidence": {
  "real_pixels_pct": 61.4, "generated_pixels_pct": 38.6,
  "evidence_preserved": true,
  "synthesis": {"model": "sdxl-inpaint+ip-adapter", "resolution_px": 1024,
                "tiles_px": 704, "seed": 824731},
  "generated_regions": [{"bbox": [0, 0, 0, 0], "area_pct": 31.2,
                         "kind": "diffusion_reimagination"}],
  "identity_similarity": 0.52, "identity_action": "kept"
}
```

Card copy: *"**Hybrid restoration.** 61 % original evidence, 39 % AI
re-imagined (identity similarity 0.52)"* plus a **[Show what's real]** overlay —
the evidence mask is persisted to `debug_output/`, so the real/generated claim
is auditable, not asserted. Rollbacks keep today's honest wording; a hybrid run
never hides an identity failure.

## 8. Unchanged invariants (recap)

Evidence-preservation contract (§1) · gate conditions (§2) · lazy-load/unload +
semaphore (§4.3) · bounded rollback ≤ 2 re-rolls (§6) · report honesty and the
offer line (§4.4, §7) · zero behavior change when the toggle is off.

## 9. Implementation plan (v2)

| Phase | Scope | Est. |
|---|---|---|
| **H1 — core** | `resource_profiler.py` + tier-matrix tests; `reimaginer.py` (SDXL-inpaint + IP-Adapter, tier-driven config); evidence composite + bytewise assertion; SFace gate (YuNet+SFace ONNX → weights dir) + threshold calibration on case fixtures; registry stage; report `evidence` + `resource` blocks; requirements (+`psutil`, `diffusers`, `accelerate`, `transformers`, pinned); weights-manifest entries; tests: tier matrix runs **unconditionally**, e2e weights-gated skips per `test_deblur.py` | ~2.5 d |
| **H2 — UI** | Re-imagine toggle + beta preset (reuses `/api/presets`), report evidence/resource section, Show-what's-real overlay | ~1 d |
| **H3 — optional** | ControlNet-inpaint on STANDARD/AMPLE, tiled-ControlNet exploration for MINIMAL, AMPLE batch mode, SFace threshold re-calibration sweep, SD1.5 fallback bench if chunking underdelivers | ~1–2 d |

Total ≈ 4–5 days; fully revertible; **zero behavior change** with the flag off.

## 10. Risks

| Risk | Mitigation |
|---|---|
| 4 GB stays tight even chunked | Tier floor + graceful skip + SD1.5 fallback (§5); never an OOM |
| Chunked tiles drift in tone/texture | Feathered overlap, composite asserted, verifier + identity gate judge the composite |
| SFace thresholds miscalibrated | Calibrate on case8–12 in H1 before freezing; report the raw score either way |
| IP-Adapter identity weaker than FaceID-class | SFace metric catches drift and rolls back; FaceID upgrade is H3 material |
| Slow on MINIMAL (4–8 min) | Opt-in beta, disclosed in tooltip; STANDARD/AMPLE tiers exist precisely for hosts that can pay |
| New deps (diffusers/accelerate/psutil) | Pinned minimal set; psutil is the only one imported at module scope, diffusers stays lazy inside `reimaginer.py` |
| Licensing | OpenRAIL-M recorded in weights README |

## 11. Measured addendum — H3 calibration (2026-09-27)

The §6 thresholds were exercised on the reference box with IP-Adapter conditioning live.
Result: a documented **negative** — the calibration question "which scale passes?" has no
answer on hopeless coverage, and the gate's refusal is the designed behavior.

* 8 E2E points on case9's synthetic 60%-destroy mask (scale × seed). SFace worst-cosine
  spread across seeds at fixed scale ≈ 0.17 (0.21–0.38); scale effect at fixed seed
  ≈ 0.03 (0.5→1.0, means 0.22–0.30). **Seed variance dominates.**
* No (scale, seed) point reached PASS 0.42; several cleared FLAG 0.36 and shipped as
  disclosed drift; the rest rolled back. Unconditioned baseline remains −0.099 —
  conditioning still earns its keep, it just cannot certify identity here.
* Verdict: `IP_ADAPTER_SCALE` stays **0.5**. Higher scales bought nothing and over-bake
  the *damaged* reference face. The structural fix is FaceID-class identity conditioning
  (H4-class item); `REIMAGINE_IP_SCALE` / `REIMAGINE_PASS_COSINE` / `REIMAGINE_FLAG_COSINE`
  remain available for future re-calibration, and `scripts/sweep_ip_scale.py` (fresh
  subprocess per point — fp32 weights never return to the OS from a live interpreter;
  fixed seed across points) is the harness to re-run when that lands.
* Live end-to-end check through the API on `teared.jpg` (reimagine=true): Stage 12 measured
  frame coverage 32.96% < 35% bar (post-LaMa healing) → `not_applied` with the reason and
  coverage numbers disclosed in the report — the honesty path working as specified.
* H3 gate hardening: `compare(max_faces=0)` verifies every YuNet face (worst cosine
  governs); per-face scores land in the report block; `reimagine(rerolls=N)` bounds the
  ladder per call.
