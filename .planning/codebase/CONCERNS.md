# Codebase Concerns & Resolutions

**Analysis & Remediation Date:** 2026-09-19  
**Status:** All critical bugs, concurrency hazards, security vectors, and test coverage deficiencies are resolved (28/28 tests passing). The primary active architectural challenge is transitioning from the static 14-stage waterfall to the v2.0 Two-Tier PPCAV Closed-Loop Orchestrator.

---

## Resolved Tech Debt & Defect Cascades

**1. False Positive Crack Masking on Specular Highlights:**
- **Status:** RESOLVED
- **Root Cause:** In `backend/ai/damage/enhanced_detector.py`, `detect_white_flakes` had a low threshold ($V > 165$, $L > \text{mean\_L} + 20$) and lacked boundary gradient checks, mistaking smooth specular highlights on hat ribbons/brims for peeling emulsion. `MaskRefiner` dilated these into solid masks, causing LaMa to erase hat brims with forehead skin.
- **Fix:** Raised monochrome thresholds ($V > 215$, $L > \max(\text{mean\_L} + 35, 205)$) and added Sobel boundary gradient verification (`mean_bnd_grad >= 35.0`) to reject smooth highlights while keeping step-edge emulsion chips.

**2. Submental Jawline Shadow Erasure by GFPGAN:**
- **Status:** RESOLVED
- **Root Cause:** GFPGAN's face restoration crop blended down to $0.96 \times \text{crop\_h}$, replacing the natural dark shadow between chin and shoulder with flat synthesized skin.
- **Fix:** Implemented an anatomical `chin_fade` mask in `backend/ai/restoration/face_restorer.py` starting at $0.84 \times \text{crop\_h}$ and feathering to $0.0$ by $0.92 \times \text{crop\_h}$.

**3. Foggy Hair and Feather Textures from Over-Denoising:**
- **Status:** RESOLVED
- **Root Cause:** Stage 5 NLM denoising lacked luminance edge gating and did not compare local variance against the noise floor ($\sigma_{noise}^2$), smoothing fine hair strands.
- **Fix:** Added noise-floor edge gating (`edge_thresh = max(30.0, 2.5 * sigma_noise)`) and variance protection (`noise_floor_var = max(16.0, sigma_noise ** 2)`) in `restoration_pipeline.py`.

**4. Scanner Canvas Margins Driving Neural Coordinate Drift:**
- **Status:** RESOLVED
- **Fix:** Implemented `ImageNormalizer.detect_and_crop_borders()` in `image_normalizer.py`, automatically detecting and trimming scanner canvas borders before neural stages.

**5. Color Bleeding Across Clothing and Headwear:**
- **Status:** RESOLVED
- **Fix:** Wired `EdgeAwareColorBleedFixer.fix_bleeding(strength=0.75)` into `ColorArtifactReducer.reduce_artifacts()` in `color_refiner.py` and expanded `subject_mask` upward to protect headwear.

---

## Active Architectural Concerns & Roadmap (v2.0 Orchestrator Migration)

**1. Static Waterfall Brittleness:**
- **Concern:** The existing 1,309-line `restoration_pipeline.py` executes stages sequentially in a one-way feedforward direction. If an upstream stage creates an artifact or fails to clean a defect, downstream stages cannot self-correct.
- **Mitigation:** Migrating to the **Two-Tier PPCAV Loop** specified in `pixrevive_orchestrator_plan.md`.

**2. Spatial Scale Inversion Across Super-Resolution:**
- **Concern:** Real-ESRGAN upscales images by $4\times$. Looping back into inpainting or colorization after super-resolution requires $16\times$ more VRAM and compute, causing out-of-memory crashes and coordinate drift.
- **Mitigation:** The v2.0 plan strictly decouples Tier 1 (native resolution iterative loop) from Tier 2 (terminal $4\times$ super-resolution and exhibition finish).

**3. Detail Synthesis Without Semantic Material Segmentation:**
- **Concern:** PixRevive lacks a semantic material parser (e.g., skin vs. fabric vs. hair). Attempting cross-photo patch borrowing risks pasting cloth weave onto skin.
- **Mitigation:** The v2.0 plan restricts detail synthesis to **Anisotropic Structure-Tensor Grain Synthesizer** and high-frequency bandpass residual re-injection, avoiding ungrounded cross-region borrowing.

**4. Latency Budgeting:**
- **Concern:** Unbounded iterations of heavy deep models (LaMa, DDColor, GFPGAN) could cause restoration latency to spike from 30 seconds to several minutes.
- **Mitigation:** Enforcing an **Action Cost Point Budget (20 CP Max)** in `LoopController` to guarantee all restorations finish within 45 seconds.
