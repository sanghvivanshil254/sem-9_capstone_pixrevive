# PixRevive — Execution Plan (v2 — taxonomy-aware)

> **STATUS (2026-09-25): COMPLETE.** All phases 0–6 implemented, including the
> decision gates (5.2 archive, 5.4 prune) and frontend items (5.5–5.7).
> See [CHANGELOG.md](../CHANGELOG.md).

Turns `docs/DEAD_CODE_REMOVAL.md` (D#), `docs/IMPROVEMENT_SUGGESTIONS.md`
(U-#/Q-#/A-#/O-#), and `docs/DAMAGE_TAXONOMY_MAPPING.md` (T#) into an ordered, verifiable
work sequence. Each phase ends with a **green gate** — do not advance until it passes.

**Defaults chosen** (questionnaire dismissed — all reversible, flagged where a decision
gate exists): keep the waterfall engine and archive the orchestrator · build the
Restoration Report honesty layer · add deblurring only (Restormer), no CodeFormer ·
no deadline compression assumed.

**Ground rules (from CLAUDE.md — enforced):**
- Interpreter: `"C:/Program Files/Python312/python.exe"` · Pytest:
  `"C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe"` with
  `PYTHONPATH=backend` · **no venvs ever**.
- Weights stay in `backend/ai/weights/`; DDColor stays vendored; never
  `pip install` DDColor or upgrade basicsr.
- Backend has no `--reload`: after each backend change, restart and wait for
  `/api/health` → `{"ready": true}` before manual verification.
- Progress contract: after touching stage ids, run
  `"C:/Program Files/Python312/python.exe" scripts/check_progress_stream.py`.

**Verification used throughout:**
```bash
PYTHONPATH=backend "C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe" tests/ -v
"C:/Program Files/Python312/python.exe" scripts/check_progress_stream.py
```

---

## Phase 0 — Safety net (½ day)

| # | Task | Ref | Output |
|---|---|---|---|
| 0.1 | `git init` + initial commit (no repo exists in this checkout — deletions must be revertible before anything else) | — | clean baseline commit |
| 0.2 | Extend `.gitignore`: `debug_output/`, `test_artifacts/`, `test-results/`, `backend/temp_results/`, `__pycache__/`, `*.pyc`, `backend/training_data/bw|color/` | D20 | git status clean of artifacts |
| 0.3 | Run full pytest suite; record pass/fail baseline + durations in `docs/baseline_test_run.md` | — | known-good starting point |
| 0.4 | Run `check_progress_stream.py` against a live backend once; record emitted stage sequence | U-7 baseline | stage-order reference |

**Gate:** pytest baseline recorded; artifacts untracked.

---

## Phase 1 — Junkyard purge (1 day) — all 🟢 zero-risk removals

| # | Task | Ref |
|---|---|---|
| 1.1 | Salvage `benchmark_runner.py` + `full_dataset_sanity_sweep.py` → `scripts/diagnostics/`; fix imports if needed | D21 |
| 1.2 | Delete `test_artifacts/*.py` (~80 files); `git rm -r --cached test_artifacts debug_output test-results` | D21/D22 |
| 1.3 | Delete `backend/tests/run_pipeline_tests.py`; move `backend/tests/metrics.py` → `tests/metrics.py` | D23 |
| 1.4 | Remove `realesr-general-x4v3.pth` entry from `download_weights.py`; remove `zeroscratches` (+ comment block) from `backend/requirements.txt` | D9/D10 |
| 1.5 | Delete unreachable benchmark body in `admin_router.py`; keep `/training/*` endpoints returning honest "not available" JSON, or remove endpoints + WeightsAdmin training panel together — **choose one, do it completely** | D15 |
| 1.6 | Drop unused `_realesrgan_tile_forward` from `restoration_pipeline.py` import | D13 |
| 1.7 | Fix `restoration_pipeline.py` header docstring: remove Stage 5b/6b phantom snapshot lines | D1 |
| 1.8 | **Do NOT delete** `HalftoneSuppressor` (D5) — Phase 4B wires it in as the moiré specialist (T-9) | T-9 |

**Gate:** pytest green (same set as baseline), backend boots, `check_progress_stream.py` passes.

---

## Phase 2 — Truth restoration + cheapest coverage win (2 days)

| # | Task | Ref |
|---|---|---|
| 2.1 | Rewrite `PROJECT_DOCUMENTATION.md` against current code: DDColor, real stage table, real routes (BW_RESTORE repairs-damage-first), real endpoints, weights table incl. `defect_detect/FT_Epoch_latest.pt`; add **capabilities & limitations section built from the taxonomy mapping** (what's recovered vs reconstructed vs out of scope) | D2 + T-doc |
| 2.2 | Correct `CODEBASE_MAP.md`: add `orchestrator/`, `admin_router.py`, ImpulseDenoiser/SpeckSuppressor, `detect_and_crop_borders`, `DefectSegmenter`; remove nonexistent frontend components; fix "segmentation absent" claim | D4/I1 |
| 2.3 | Delete `docs/current_pipeline.md` (pre-v2 relic) | D3 |
| 2.4 | Write root `README.md` (O-5): architecture, stage table, run/test commands (mandated absolute paths), weights setup, env vars, and the taxonomy-based capability matrix | O-5/U-10 |
| 2.5 | `main.py`: `/api/health` delegates to `pipeline.health_status()`; remove duplicate `ddcolor` key | U-2 |
| 2.6 | Per-model load tracking (U-1): `initialise_models` records successes/failures; health reports them; Navbar shows degraded-mode pill | U-1 |
| 2.7 | **Salvage-then-delete** `Enhancer` class: move `correct_color_cast` logic into `ColorArtifactReducer` (feeds T-7), then delete the class; update `postprocessing/__init__.py` | D17 + T-7 |
| 2.8 | **T-1 foxing/mold wiring:** call `EnhancedDamageDetector.detect_dark_spots()` in `_stage_2_fracture_masking` (OR into `combined`), sized to avoid shadow false-positives; add a foxing-spotted fixture to `tests/restoration_dataset/damaged/` + regression test asserting spots land in the final mask | T-1 |
| 2.9 | **Adopt the 4 user benchmark cases as regression fixtures** (probe evidence 2026-09-21): `teared.jpg` (35.7% refined mask — heavy-regen path), `SigShrink…speckle…png` (NoiseRouter misclassifies speckle as jpeg/blocking family — add a test asserting the denoise path, not deblock, handles it post-Stage-0), `image1.jpeg` (spiderweb creases, 14.6% mask, mouth-adjacent damage), `image3.jpg.jpeg` (5.1% minor creases). Copy into `tests/restoration_dataset/damaged/` with per-case assertions; they become the 4B acceptance set alongside case1–6 | T-fix |

**Gate:** pytest green incl. new T-1 test; docs grep-clean of ECCV16/zeroscratches/venv;
`/api/health` shows per-model truth; foxing fixture gets repaired end-to-end.

---

## Phase 3 — Structural fixes (3–4 days)

| # | Task | Ref |
|---|---|---|
| 3.1 | **Stage registry (A-2):** `backend/ai/pipeline/stage_registry.py` — ordered `StageSpec(id, name, method, progress_pct, debug_snapshot, route_gate)`. Refactor `restore()` to iterate it; export to `frontend/src/generated/stages.json` (`scripts/export_stage_list.py`); update `check_progress_stream.py` to validate against the registry. **Design the registry to accept the new T-stages (deskew/flat-field/deblur) as first-class entries** | A-2/D14/I3 |
| 3.2 | Prune ProgressBar dead stage ids (`smart_denoise`, `upscale`, `harmonize`, `enhance`, `final_refine`, `identity_check`, `encode`) | D14 |
| 3.3 | Implement `device_preference` end-to-end (U-5): thread to model `.load(device)`; lazy-reload on mismatch; tests via `torch.cuda.is_available` monkeypatch | U-5/D12 |
| 3.4 | Split `_stage_3_lama_inpaint` into `_seam_smooth`, `_protect_facial_anatomy`, `_face_wipeout_circuit_breaker` + unit tests (feathered eye/mouth masks, breaker thresholds 0.03<ratio≤0.20, grad ratio <0.05) | U-3 |
| 3.5 | Split `restoration_pipeline.py` into `pipeline/stages/{preprocess,damage,color,finish}_stages.py`; `RestorationPipeline` = thin orchestrator. Do **after** 3.1 so the registry moves with it | A-3 |
| 3.6 | Task-store pruning on `get_status` + startup timer (U-9) | U-9 |
| 3.7 | SR fallback respects 2K target (Q-5) + unit test with 3000px input | Q-5 |

**Gate:** pytest green incl. new unit tests; `check_progress_stream.py` passes;
50 sequential requests keep `temp_results/` pruned.

---

## Phase 4A — Quality engine: routing + measurement (3 days)

| # | Task | Ref |
|---|---|---|
| 4A.1 | **Router authority (Q-1):** `restore()` consumes route flags ANDed with user flags; remove ad-hoc `noise_score > 6.0` scatter; unit tests for all 6 routes + fallback | Q-1/D16 |
| 4A.2 | **Benchmark harness (Q-6):** `scripts/benchmark_quality.py` — case1–6 through pipeline, PSNR/SSIM (+ optional LPIPS/NIQE), writes `test_artifacts/benchmark_report.json` + markdown diff vs previous run; wire `QualityService` to compute metrics at pipeline end behind `PIXREVIVE_DEBUG=1`. **This is the acceptance ruler for every 4B item** | Q-6/D6 |
| 4A.3 | Colorization hallucination rollback (Q-3): hue-histogram guard post-DDColor → fallback to L-preserved grayscale + warning; synthetic pathological test | Q-3 |
| 4A.4 | Guided-filter seam smoothing at native resolution (Q-4), reusing 3.4 helper | Q-4 |
| 4A.5 | Stage-2b face-safe zone from Stage-1b convex hull instead of center rectangle (U-6); no-regression check on case5 | U-6 |

**Gate:** benchmark report at parity on all 6 cases (no metric regression >2%).

---

## Phase 4B — Taxonomy coverage: new detection & correction stages (1 week)

New stages register in the stage registry (3.1) and land in the taxonomy matrix verdicts.
Order within the phase is by dependency, not priority.

| # | Task | Ref |
|---|---|---|
| 4B.1 | **T-4 Auto-deskew:** Hough-line dominant rotation estimate + affine correction in `ImageNormalizer`, before border crop; gated (skip when \|angle\| < 0.3°); test with rotated fixture | T-4 |
| 4B.2 | **T-5 Flat-field illumination correction:** large-kernel background estimate (big Gaussian / morphological closing) + divide; gated on measured illumination gradient; test with synthetic gradient overlay | T-5 |
| 4B.3 | **T-9 Moiré specialist:** wire `HalftoneSuppressor` (FFT notch) into Stage 1, ahead of the SR bilateral pre-filter; verify no regression on case2 landscape (macro edges must not trip it — the autocorr lesson from Stage 0 applies) | T-9/D5 |
| 4B.4 | **T-6 Banding pass:** 16-bit intermediate + smooth re-quantization/dither on gradients; apply pre-SR when banding_score high | T-6 |
| 4B.5 | **T-7 Color-cast rebalancing:** wire the salvaged `correct_color_cast` (2.7) into ColorArtifactReducer for differential dye fading; gray-world + channel-curve; test with sepia-shifted fixture | T-7 |
| 4B.6 | **T-3 Deblur stage 4.5 (the big one):** add Restormer/NAFNet real-blur weights (~65MB) to `download_weights.py` manifest (GitHub release source, per weight rules); vendored minimal inference module `backend/ai/restoration/deblurrer.py` following the LaMa TorchScript pattern; **gate strictly**: run only when `blur_score < threshold` AND directional-variance test says optical blur (not just soft detail); placed after denoise, before colorize; 2K output cap; benchmark case1–6 must show no regression on sharp inputs (gate must never fire on them) | T-3 |
| 4B.7 | Re-run taxonomy matrix; update verdicts ✅/🟡 in `docs/DAMAGE_TAXONOMY_MAPPING.md` to reflect new coverage | T-doc |

**Gate:** benchmark parity or improvement on all 6 cases; each new stage has a unit test +
a gating test proving it stays inert on inputs that don't need it.

---

## Phase 5 — Honesty layer + decision gates + frontend (1 week)

| # | Task | Ref |
|---|---|---|
| 5.1 | **T-2 Restoration Report (flagship):** create `backend/ai/report/` — persist per-detector mask coverage from Stage 2 (already computed), classify evidence into **recovered / reconstructed / unrecoverable** (inpaint coverage ≥25% of frame; face-region inpaint ≥40% with GFPGAN applied; flagged taxonomy classes), emit machine-readable `{repaired, reconstructed, flagged}` + `stage_timings` into metadata; **UI: collapsible Restoration Report card** on the result panel; colorization results carry the "plausible colors, not original colors" disclaimer | T-2/O-3 |
| 5.2 | **Decision gate — orchestrator (A-1):** default = move `backend/ai/orchestrator/` → `experimental/orchestrator/` with README; remove env hook from `restore()`; drop orchestrator-only state from PipelineContext (D19). *Reversible: promote later if parity is ever proven via 4A.2 harness* | A-1/D8/D19 |
| 5.3 | **Decision gate — IdentityValidator (D7):** wire it as a debug-only post-SR check feeding the Restoration Report (identity-risk warning), or delete module + its 2 test files. Default: wire — it feeds 5.1's report | D7 |
| 5.4 | **Decision gate — wavelet denoiser stack (D18):** keep `AdaptiveDenoiser`/`SmartDenoiser` only if NoiseRouter gains a dispatch path to them; else delete + prune their tests. Default: delete (Stage 5 NLM + NoiseRouter cover the space) | D18 |
| 5.5 | Frontend API client + `VITE_API_BASE` (A-6); replace 12 hardcoded URLs | A-6 |
| 5.6 | Split `ImageUploader.jsx` (A-7): `useRestorationJob` hook + `ResultCard` + `OptionsPanel` (Report card from 5.1 lands in `ResultCard`) | A-7 |
| 5.7 | Presets (Q-8): backend `PRESETS` dict → `/api/presets` → UI selector mapping to the four toggles | Q-8 |

**Gate:** pytest green; e2e (`npx playwright test`) green via local Chrome CDP; report card
renders for a case6 (severe tear) upload showing unrecoverable-content flag.

---

## Phase 6 — CI & hardening (2 days)

| # | Task | Ref |
|---|---|---|
| 6.1 | Add pytest job to CI (Windows runner, Python 3.12, absolute-path pytest, `tests/ -k "not slow"`, pip cache) | O-4 |
| 6.2 | Admin weight upload: extend to `big-lama.pt`, DDColor, and the new deblur weights (O-1) with same atomic staging | O-1 |
| 6.3 | Pyright basic mode on `backend/{main.py,routers,services,ai/pipeline,ai/report}` in CI (A-4) | A-4 |
| 6.4 | Optional VRAM-aware concurrency (O-2) — only if a real need exists; otherwise document the semaphore | O-2 |
| 6.5 | Tag `v1.0`; CHANGELOG.md from here | — |

**Gate:** CI green; docs current; fresh-clone smoke test following README verbatim succeeds.

---

## Dependency notes

- 3.1 (registry) → 3.5 (split) → 4B (new stages must register in it) → 5.1 (report reads
  stage metadata). The registry is the spine of the whole plan.
- 4A.2 (metrics) → 4B (no new stage merges without benchmark evidence).
- 2.7 (salvage) → 4B.5 (T-7 uses the salvaged logic).
- 5.1 (report) consumes 3.1 metadata, 4B coverage data, and 5.3's identity check — do it
  after those, but its **metadata-collection half** can start any time after Phase 3.
- Decision gates 5.2/5.3/5.4: present to the project owner with the 4A.2 benchmark report.

## Effort summary

| Phase | Duration | Character |
|---|---|---|
| 0 Safety net | ½ day | git + baselines |
| 1 Junkyard purge | 1 day | deletion, zero-risk |
| 2 Truth + foxing win | 2 days | docs, honest health, T-1 |
| 3 Structural fixes | 3–4 days | registry, splits, device |
| 4A Routing + metrics | 3 days | router authority + benchmark ruler |
| 4B Taxonomy coverage | 1 week | deskew, flat-field, moiré, banding, cast, deblur |
| 5 Honesty + decisions | 1 week | Restoration Report, engine decision, frontend |
| 6 CI & hardening | 2 days | automation, tag v1.0 |
| **Total** | **~5 weeks** | |

## If time compresses (minimum viable path)

0 → 1 → 2 → 4A.2 → 5.1 → 6.1. That is: clean the junkyard, tell the truth in docs and
health, build the measuring ruler, ship the Restoration Report, add CI. The taxonomy
coverage items (4B) and structural refactors (3) are what make PixRevive *better*; this
path is what makes it *credible*.

---

## Phase H1/H2 — Hybrid Evidence-Preserving Diffusion (post-1.0, status as of 2026-09-27)

Design: [HYBRID_DIFFUSION_DESIGN.md](HYBRID_DIFFUSION_DESIGN.md) v2. Goal: when damage is so
extensive that any "repair" is fabrication anyway, re-imagine ONLY the destroyed regions with a
diffusion inpainting model while every surviving pixel is composited back bytewise
(`out[E] == original[E]`), gated by an identity check that can roll the whole thing back.

**Status: H1 + H2 complete (code, tests, UI); H3 (IP_ADAPTER_SCALE calibration) open.**

| Item | State |
|---|---|
| Resource tiering (`resource_profiler.py`) | ✅ 4 tiers from measured VRAM/RAM; MINIMAL pins fp32 + sequential CPU offload (fp16 diffusion is NaN-broken on this host — see design §4.5); `REIMAGINE_FORCE_TIER` override never waives floors. 37 tests |
| Re-imagine engine (`reimaginer.py`) | ✅ SD1.5 (DDIM) / SDXL (unhealthy on 4 GB — pinned off), IP-Adapter identity conditioning, evidence composite with bytewise assertion, feathered 1024→4×704 tile grid, NaN self-heal (black>60% → seed+1 → SD1.5 fallback), SFace rollback ladder (≥0.42 pass / 0.36–0.42 flag / <0.36 rollback, ≤2 re-rolls). 23 tests |
| Identity gate (`face_identity_gate.py`) | ✅ YuNet + SFace, thresholds verified on case9 self-similarity (>0.95). 11 tests |
| Stage 12 wiring | ✅ `stage_registry` (pct 99, route gate `enable_reimagine`), coverage gate (≥35% frame or ≥55% worst-face), report disclosure block (hybrid / not_applied + skip reasons), `reimagine` Form flag on both endpoints, `reimagine_beta` preset. 18 tests |
| Frontend (H2) | ✅ OptionsPanel beta checkbox + tooltip, `useRestorationJob` flag & preset mapping, report hybrid card + skip-note, `/api/presets` refetch loop fixed |
| "Show what's real" overlay (§7) | ✅ hybrid runs persist the colorized evidence-zone map (`temp_results/{task_id}_evidence.png`, served at `/api/process/{id}/evidence`); the restored pane gains a "Show what's real" toggle stacking the pixel-aligned overlay over the output |
| Full suite | ✅ 312 passed, 1 skipped, 105 subtests |

**Integration findings (measured, all fixed):** IP-Adapter folder must mirror the h94 repo
layout (`models/image_encoder` inside `models/`, not beside it); the encoder ships without
`config.json` → transformers guesses ViT-B and dies on size mismatch (config vendored in
`reimaginer.py` and self-healed onto disk); `enable_attention_slicing()` must run BEFORE
`load_ip_adapter()` or SlicedAttnProcessor wipes the IP-Adapter processors and the UNet dies
on `'tuple' object has no attribute 'shape'`; diffusers 0.40 has no `pipe.ip_adapter`
attribute — load success is tracked in engine state instead.

**E2E on case9 (60% synthetic mask):** unconditioned generation correctly rolls back 3/3
attempts (best SFace cosine −0.099). With IP-Adapter conditioning active (scale 0.5), the
best cosine rises to **0.340** — a huge identity gain but still inside the 0.36–0.42 flag
band's wrong side, so the gate honestly rolls back 3/3 attempts and records
`skipped_degraded` with `evidence_preserved: true` (572 s). This is the measured H1-tail
finding that motivates H3: `IP_ADAPTER_SCALE` 0.5 is not enough for a 60%-destroyed face;
calibration (and possibly threshold re-justification) is the next task.
See `test_artifacts/reimagine_e2e4.log` / `test_artifacts/reimagine_e2e/case9_record.json`.

**H3 (next):** calibrate `IP_ADAPTER_SCALE` (0.5 default) on case9 + real damage; decide
whether the pass threshold can hold at 0.42 with IP conditioning; consider full-image
SFace verification instead of largest-face only.

---

## Phase H3 — identity hardening + calibration (2026-09-27)

| Item | State |
|---|---|
| All-faces identity verification | ✅ `SFaceIdentityGate.compare(max_faces=0)` verifies EVERY YuNet face; worst per-face cosine governs (drift on any face, not just the biggest, can roll back). Stage 12 now gates on it. `per_face` scores disclosed in the report block |
| Env-tunable ladder | ✅ `REIMAGINE_PASS_COSINE` / `REIMAGINE_FLAG_COSINE` / `REIMAGINE_IP_SCALE` overrides (never widen the gate silently — they are calibration knobs, not waivers); `reimagine(rerolls=N)` param bounds the ladder per call |
| Calibration sweep | ❌ **Negative result (measured, honest).** 8 E2E points on case9 (60%-destroy mask): seed variance (SFace cosine 0.21–0.38 across seeds) dominates any scale effect (0.5→1.0 spread ≈0.03 on a fixed seed; per-scale means 0.22–0.30). No IP-Adapter scale clears the 0.36 rollback floor on hopeless coverage — with 3 attempts per run the gate correctly refuses, and the bytewise evidence composite means the fallback (standard restoration) ships instead |
| Verdict | `IP_ADAPTER_SCALE` stays **0.5** (diffusers prior; higher scales showed no identity gain and over-bake the damaged reference). Conditioning still matters: unconditioned cosine −0.099 vs conditioned 0.21–0.38. The structural fix is FaceID-class identity conditioning (H4-class), not more scale tuning |

Harness: `scripts/sweep_ip_scale.py` (fresh python subprocess per point — the fp32 weights
never return to the OS from a live interpreter, so in-process sweeps self-refuse at the RAM
floor; fixed seed across points so scale is the only variable). Raw data:
`test_artifacts/reimagine_sweep/`.

### Integration audit (2026-09-27) — 5 live-path bugs found and fixed

The full chain was exercised end-to-end through the real API (heavy-damage probe with
`reimagine=true`); each failure below was reproduced live before being fixed:

1. **Report colorization lie** — `colorize=false` on sepia input still emitted "colors
   added by AI" (the report inferred from `color_type` alone). Fix: `colorization_applied`
   is seeded by `restore()` at flag time and flipped by Stage 6 on success; the claim now
   requires positive evidence, and a disabled-colorization run gets an explicit
   "colorization was disabled — monochrome kept" disclosure.
2. **Stage 12 could never run inside the server** — the standard models held ~2.4 GB so
   only 0.94 GB stayed free; the resource gate refused every live attempt (standalone E2E
   never saw this). Fix: Stage 12's `_release_standard_models()` frees LaMa/GFPGAN/
   Real-ESRGAN/DDColor/RetinaFace/U-Net before the engine profiles (0.94 → 2.88+ GB) and
   `_restore_standard_models()` reloads via the canonical `_reload_models_on` path in a
   finally block. Measured live: 2.88 GB was still refused → with detectors released the
   gate passes and the hybrid run completed (`mode: hybrid-restoration, done`).
3. **CWD-relative weights paths** — `reimaginer.WEIGHTS` and the gate's `WeightsDirDefault`
   resolved from the process CWD; the server runs with `cwd=backend/`, so every live run
   reported "sd15 weights missing" while root-launched scripts worked. Fix: both anchored
   to their module files.
4. **Faceless probe crashed generation** — a loaded IP-Adapter makes the UNet require
   embeds; when YuNet finds no face in the input, nothing conditions ON, and generation
   crashed (`'NoneType' is not iterable`). Fix: two-way adapter sync in `_generate`
   (unload when the image has no face, reload when one appears). Previously only the
   one-directional case was handled.
5. **Stage-level errors were invisible** — the stage's exception branch recorded an
   `error` key the report never surfaced. Fix: full traceback logged via `logger.exception`
   and the report block now emits `status: "error"` with the reason.

Also: a stale skip reason no longer lingers on a success record. Full suite after the
audit: **320 passed, 1 skipped, 105 subtests**. Known host quirk (documented, not a code
bug): running the pytest suite while the backend server holds torch weights can segfault
the runner (access violation inside torch model init under memory pressure) — pause the
server for suite runs, as was done for the green run above.
