# Changelog

All notable changes to PixRevive are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed
- **Compaction (2026-09-29)** — removed stale planning docs (`plan.md`,
  `photo_restoration_pipeline.md`, `pixrevive_orchestrator_plan.md`, the Ralph PRD,
  captured pytest/stream outputs, retired audit/inventory docs), the superseded
  `start_dev.bat`/`restart_all.bat` launchers (superseded by `.claude/launch.json`),
  one-off probe/diagnostic scripts, the Vite-template `App.css`/`hero.png`, and
  `e2e/example.spec.js`. 44 MB of untracked `debug_output_*` run artifacts deleted and
  the pipeline's debug snapshot dirs added to `.gitignore`.

### Added
- **Phase V0.1 — grain-floor routing + skin-residual validation (beta).** Closes the two
  gaps the 4/10 (speckle Lena) and 6/10 (teared) live critiques identified. ① The
  4/10 image's real failure was never the specks (Stage 2 provably confines them to the
  border ring — probe `.freebuff/probe_stage2_speckle.py`) but the **unremovable grain
  floor** riding through pre-SR into Real-ESRGAN and out through DDColor as watercolor
  background. `NoiseRouter` gains a `grain_floor` family (fires at residual_mad ≥ 2.6 +
  edge structure + no impulse signature — a band no existing family could see: the
  corpus's grain class sits at residual_mad 3.5–4.0 while clean scans are 1.4–1.9, and
  the target's residual profile *declines* with luminance, so speckle scoring is blind
  to it) plus a `grain_floor_nlm` reducer (L-channel NLM h=3+3·strength, Canny
  edge-protect blend, chroma untouched). Pre-SR's sharpness guard drops 6.0 → **2.5** so
  grain-class images get the anti-amplitude pass (clean scans 1.7–2.4 keep sharpness).
  Also fixed a latent Stage 5 dispatch bug: the registry loop passed `on_progress`
  positionally into Stage 5's `faces` slot, so spatially-gated denoising crashed with
  "'function' object is not iterable" on every routed run since the A-2 refactor —
  invisible because `denoising: 0.0` looked like a skip and `add_log` never reaches the
  SSE stream. Verified live over HTTP: `denoising` 0.0 → 0.488 s, Stage 4b accepted
  grain_floor_nlm (edge_loss 0.5%), Stage 8 pre-SR smoothing armed at noise 3.75.
  ② Semantic rule **B3 `skin_residual_damage`**: face-zone white top-hat (k=5, L>12),
  eye/mouth zones excluded, 1–40 px components, gated on lost evidence (≥5% face-zone
  loss AND ≥30% co-location) so repaired skin never trips it. Live: fires on teared.jpg
  output (169 patches, 0.35% of face zone → "V1 refill target") exactly where the
  critique saw undissolved damage, and on the speckle figure (52 patches), silent on
  Gate A's pristine corpus. 24 B3+grain tests (367 total, backend paused).
- **Phase V0 — semantic validation, report-only (beta).** First slice of
  [docs/SEMANTIC_VALIDATION_PLAN.md](docs/SEMANTIC_VALIDATION_PLAN.md): an evidence-state
  classifier plus three read-only semantic rules that run after Stage 12 and can never
  touch a pixel. `ai/validation/semantic_validator.py` classifies each face-anchored
  region's evidence mask (SURVIVING ≤15% lost, DESTROYED ≥60%, AMBIGUOUS between) and
  evaluates A1 eye-presence (dark-blob structure proxy, destroyed/ambiguous quadrants
  only), A3 hairline continuity (destroyed + textured surviving adjacent hair →
  "regeneration candidate (V1)"; destroyed + flat → "possible bald" note), and C1
  bilateral garment symmetry (Canny run once on the full image, then masked —
  masked-Canny manufactures fake boundary edges, measured). Prime directive enforced:
  findings target only lost-evidence regions; surviving evidence (bald/blind/injured)
  is truth and is never proposed for modification. Wired as internal Stage 13
  (`semantic_validate`, hidden from the frontend stage list) and surfaces as an additive
  `semantic` block in the Restoration Report. Corpus gate (`scripts/corpus_semantic_gate.py`):
  Gate A — 17 pristine images with empty evidence masks → 0 false positives; damaged
  corpus → 23 plausible findings; live teared.jpg run → exactly the expected hairline
  regeneration candidate. 18 new unit tests (359 total).
- **Phase H4 — FaceID-class identity conditioning (beta).** The H3 sweep's structural fix,
  now implemented: generation is conditioned on a trained *identity* embedding instead of a
  CLIP embedding of a damaged crop. `ai/validation/faceid_embedder.py` computes 512-d
  ArcFace embeddings (YuNet landmarks → 112×112 similarity alignment → buffalo_l's
  `w600k_r50.onnx` via cv2.dnn/onnxruntime — zero new pip deps, insightface deliberately
  avoided); `ai/restoration/faceid_adapter.py` is a vendored reimplementation of
  tencent-ailab/IP-Adapter's FaceID loader (`ip-adapter-faceid_sd15.bin`, LoRA rank 128,
  4 identity tokens concatenated to the text embeddings, SDPA processors matched to
  diffusers 0.40). FaceID is fitted once at pipeline load, BEFORE the CPU-offload snapshot
  (post-offload processor swaps crash — measured); faceless images generate with null
  identity tokens (`proj(zeros)`), which retires the historical no-face crash class.
  Report discloses `identity_conditioning` ∈ {faceid, clip, none}. Weights:
  `scripts/fetch_faceid_weights.py` (buffalo_l recognizer mirrored from
  `immich-app/buffalo_l`, adapter from `h94/IP-Adapter-FaceID`; ~306 MiB total).
  Smoke-verified on the real SD1.5 inpaint pipeline: install strict-loads, generation
  runs clean through the LoRA/SDPA path, evidence composite stays bytewise intact,
  faceless pass works, ArcFace self-retrieval 0.68 at 8 steps (`scripts/smoke_faceid.py`).
- **Hybrid Evidence-Preserving Diffusion, Phase H1/H2 (beta)** — the re-imagine tier from
  [docs/HYBRID_DIFFUSION_DESIGN.md](docs/HYBRID_DIFFUSION_DESIGN.md) v2. When ≥35% of the
  frame (or ≥55% of the worst face) is classified unrecoverable, Stage 12 re-imagines ONLY
  the lost regions with SD inpainting while every surviving pixel is composited back and
  asserted bytewise. Identity is protected by a YuNet+SFace gate (cosine ≥0.42 pass,
  0.36–0.42 flag, <0.36 rollback, ≤2 re-rolls) that reverts to the LaMa canvas on failure.
  Off by default: `enable_reimagine` flag / "Re-imagine lost regions (beta)" UI checkbox /
  `reimagine_beta` preset.
- **Dynamic resource tiering** — `backend/ai/restoration/resource_profiler.py` measures VRAM
  and RAM at gate-fire and selects MINIMAL/STANDARD/AMPLE (REFUSED below floors); MINIMAL
  pins fp32 + sequential CPU offload because fp16 diffusion is NaN-broken on GTX 1650
  (measured, design §4.5). `REIMAGINE_FORCE_TIER` env override never waives the floors.
- **Report disclosure** — the Restoration Report gains an additive `reimagine` block: mode
  (hybrid-restoration / not_applied), evidence split, identity verdict, resource tier, and
  an explicit skip reason when the stage did not fire.
- **"Show what's real" evidence overlay (design §7)** — hybrid runs persist a colorized
  zone map (green = surviving original pixels, magenta = AI-re-imagined) to
  `temp_results/{task_id}_evidence.png`, served at `GET /api/process/{id}/evidence`, and
  the restored-output pane gains a "Show what's real" toggle that stacks the aligned
  overlay over the result — the real/generated claim is auditable, not asserted.
- **Weights fetcher** — `scripts/fetch_reimagine_weights.py` (SD-1.5 inpainting fp16,
  SDXL optional, h94/IP-Adapter, SFace/YuNet) with `--skip-sdxl` / `--ip-adapter-only`.

### Added
- **Phase H3 — identity hardening + calibration.** The SFace gate now verifies **every**
  detected face (worst per-face cosine governs) instead of only the largest one, and
  discloses per-face scores in the report. `REIMAGINE_PASS_COSINE` / `REIMAGINE_FLAG_COSINE`
  / `REIMAGINE_IP_SCALE` env overrides plus a `rerolls` parameter make the gate tunable for
  calibration. Calibration sweep (`scripts/sweep_ip_scale.py`, 8 measured E2E points on
  case9) is a documented **negative result**: seed variance dominates any IP-Adapter scale
  effect, no scale reliably clears identity on hopeless coverage — the gate refusing there
  is by design, so `IP_ADAPTER_SCALE` stays 0.5 and FaceID-class conditioning is the
  recorded path forward.

### Fixed
- **Live-path integration audit (Stage 12 / hybrid beta).** Five bugs found by driving the
  real API end-to-end and fixed: the Restoration Report claimed "colors added by AI" even
  with colorization disabled (claim now requires the `colorization_applied` evidence flag,
  and disabled runs disclose "monochrome kept"); the re-imagine stage could never clear its
  VRAM floor inside the live server because the standard pipeline models stayed resident
  (Stage 12 now releases them before the engine runs and reloads them after); weight paths
  were CWD-relative and broke when the server ran from `backend/`; generating for a faceless
  input crashed while the IP-Adapter was loaded (two-way adapter sync in `_generate`); and
  stage-level re-imagine errors were swallowed instead of disclosed (`status: "error"` in
  the report, full traceback logged).
- `/api/presets` infinite refetch loop in the frontend (presets were re-applied every poll
  cycle, silently resetting user toggles such as the re-imagine checkbox).
- IP-Adapter loading: the image encoder must sit inside `models/` (h94 repo layout);
  diffusers resolves `image_encoder_folder` relative to `subfolder`. The encoder's missing
  `config.json` is vendored and self-healed (transformers otherwise guesses ViT-B and fails
  on a size mismatch). `enable_attention_slicing()` now runs before `load_ip_adapter()` —
  slicing used to replace the IP-Adapter attention processors and crash generation.

## [1.0.0] — 2026-09-25

First tagged release: the restoration-v2 execution plan (Phases 0–6) is complete.

### Added
- **Restoration Report (T-2, plan 5.1)** — the honesty layer. `backend/ai/report/`
  classifies pipeline evidence into repaired / reconstructed / flagged categories and
  emits a machine-readable report with stage timings. Exposed on `/api/process-stream`
  (`complete` SSE event) and `/api/status/{task_id}`, rendered as a collapsible card in
  the result panel. Colorized results carry the "plausible colors, not original colors"
  disclaimer.
- **Taxonomy stage coverage (Phase 4B)** — auto-deskew (T-4), flat-field illumination
  correction (T-5), moiré/halftone suppression (T-9), banding pass (T-6), dye-fade
  color-cast rebalancing (T-7), and gated Restormer deblurring (T-3), each registered
  in the stage registry with gating tests proving they stay inert on inputs that
  don't need them.
- **Router authority (Q-1)** — `restore()` consumes router route flags ANDed with user
  flags; ad-hoc noise-score scatter removed.
- **Quality benchmark ruler (Q-6)** — `scripts/benchmark_quality.py` measures NIQE/SSIM
  per case and diffs against the previous run; the acceptance gate for every stage.
- **Colorization hallucination rollback (Q-3)** — hue-histogram guard detects the
  single-hue DDColor wash, rolls back to L-preserved grayscale, and discloses it in the
  Restoration Report.
- **Foxing/mold repair (T-1)** — dark-spot detection OR'd into the Stage 2 mask with a
  shadow false-positive cap.
- **IdentityValidator (plan 5.3)** — wired as a debug-only post-SR identity-risk check
  feeding the report.
- **CI (plan 6.1/6.3)** — GitHub Actions workflow running the pytest suite on
  windows-latest / Python 3.12 (CPU torch, pip cache, `-k "not slow"`) plus pyright
  basic over the backend core (`pyrightconfig.json`).
- **Named presets (Q-8, plan 5.7)** — `PRESETS` dict served at `/api/presets`
  (default / fast / repair-only / colorize-only / denoise-only), rendered as a
  one-click selector in the UI that sets the pipeline toggles.
- **Frontend API client (A-6, plan 5.5)** — `frontend/src/services/api.js` derives
  every backend URL from `VITE_API_BASE` (see `.env.example`); all 17 hardcoded
  localhost URLs removed.
- **Frontend component split (A-7, plan 5.6)** — `ImageUploader.jsx` reduced to a
  composition root over `useRestorationJob` (job state + SSE streaming),
  `ResultCard` (before/after + report) and `OptionsPanel` (presets + toggles).
- **Admin weight upload for every model (plan 6.2)** — `/api/admin/weights/upload` now
  accepts `lama` (big-lama.pt), `colorizer` (DDColor) and `deblur` (Restormer) targets
  in addition to `realesrgan`/`gfpgan`, with the same atomic staging and checkpoint
  validation; `/api/admin/weights/status` reports the deblur weights too.

### Fixed
- **T-9 moiré stage was silently dead** — `HalftoneSuppressor` was called but never
  imported, so the stage no-op'd into its `except` on every run. Now genuinely wired
  (and verified gated-off on real photos).
- **Degenerate-input noise detection** — ratio-based noise detectors divided by ~zero
  on textureless content, making a flat scanned figure fire seven spurious noise
  families. A texture-variance guard suppresses them; real photos are unaffected.
- **Six never-green tests re-anchored** — `tests/test_margin_robust_noise.py` failed
  since the initial commit (recorded as the accepted baseline); probing showed the
  assertions encoded pre-4A hypotheses contradicting shipped behaviour. Re-anchored to
  measured reality with rationale, including a true thin-margin dilution control.

### Changed
- Pipeline split into `pipeline/stages/{preprocess,damage,color,finish}_stages.py`
  behind a first-class stage registry (A-2/A-3); device preference threaded end-to-end
  (U-5); junkyard purge of dead code (Phase 1); documentation truth-pass (Phase 2).
- GPU semaphore documented as the deliberate concurrency model (O-2): a 4 GB card
  serves one restoration at a time.
- **Two-Tier PPCAV orchestrator archived (decision gate 5.2)** — moved intact to
  `experimental/orchestrator/` with its tests and a provenance README; the
  `PIXREVIVE_USE_ORCHESTRATOR` hook and `restore(use_orchestrator=…)` parameter are
  gone, as is the orchestrator-only `PipelineContext` state (synthesis manifest,
  hyperparameters, transaction helpers). Reversible: promotion requires benchmark
  parity evidence.
- **Wavelet denoiser stack removed (decision gate 5.4)** — `WaveletDenoiser`,
  `AdaptiveDenoiser`, `SmartDenoiser` and `ArtifactDetector` had no dispatch path
  (Stage 5 spatially-gated NLM via NoiseRouter covers that space); deleted with their
  orphaned test.

### Notes
- Tagging: `v1.0` from this commit.
- Benchmarks at release time: parity on all user cases (max Δ NIQE +0.08, floor 0.5);
  the six historical failures are gone — full suite green modulo environment-specific
  model-weight skips.
