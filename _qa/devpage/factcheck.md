# Fact-check: /developer/weights page content

Every number and "fundamental limit" claim rendered by the page, with its source.
Verified 2026-10-20 against disk state (branch feat/dev-weights-page).

## Pipeline structure

| Claim on page | Source | Status |
|---|---|---|
| 22 stages registered; 20 shown; 2 internal (refresh_faces, semantic_validate) | `backend/ai/pipeline/stage_registry.py` count; export in `frontend/src/generated/stages.json` | ✅ |
| Stage order & % badges | `frontend/src/generated/stages.json` (generated from stage_registry.py via scripts/export_stage_list.py) | ✅ |
| Conditional gates (enable_repair, enable_denoise, enable_colorization, enable_face_restore, enable_sr, enable_reimagine) | `route_gate` field in stages.json | ✅ |
| Re-imagine β: ≥35% frame or ≥55% face destroyed | `stage_registry.py` reimagine comment; `docs/HYBRID_DIFFUSION_DESIGN.md` | ✅ |

## Models & sizes (du / Python os.path.getsize on backend/ai/weights, 2026-10-20)

| Model | Page shows | Measured | Source |
|---|---|---|---|
| Big-LaMa | 197 MB, eager | big-lama.pt = 196.1 MB | os.path.getsize |
| DDColor | 870 MB (dir), eager own thread | ddcolor/ = 870 MB du | du |
| GFPGAN v1.4 | 333 MB, eager | GFPGANv1.4.pth = 332.5 MB | os.path.getsize |
| Real-ESRGAN x4plus | 64 MB, eager | RealESRGAN_x4plus.pth = 63.9 MB | os.path.getsize |
| realesr-general-x4v3 | 4.7 MB | 4.7 MB | os.path.getsize |
| Defect segmenter dir | 431 MB | defect_detect/ = 431 MB du | du |
| Restormer | 100 MB, lazy | restormer/ = 100 MB du; lazy per stage_registry comment | du + registry |
| facelib+CodeFormer+faceid | ~979 MB combined | 186+360+~433 MB du | du |
| sdxl+sd15 inpaint | 8500 MB, lazy | 6.5G + 2.0G du | du |
| Weights total on disk | 14 GB | 14G du | du |

Load strategy (eager/lazy) source: `backend/ai/pipeline/restoration_pipeline.py initialise_models()`.

## Metrics (all medians from docs/baselines/pre_fix.json, runs=3)

| Metric | Page shows | pre_fix.json |
|---|---|---|
| seam_step | teared 15.32 | teared.seam_step 15.32 ✅ (lena 34.28 not shown) |
| hf_retention | teared 0.059, lena 1.071 | ✅ exact |
| speck_count | teared 398 (input 351), lena 162 (input 333) | exact ✅ (input labels corrected in content.js on 2026-10-20; lena input is 333, teared is 351) |
| eye_symmetry | teared 0.752 | teared.eye_sharp_ratio 0.752 ✅ |
| chroma_bleed | teared 1.41%, lena 0.87% | 0.0141 / 0.0087 ✅ |
| contour_retention | teared 0.429, lena 1.041 | ✅ exact |

## Measured effects cited in stage cards

| Claim | Source |
|---|---|
| Despeckle: removes ~94% speck amplitude for ~33% HF-energy cost | `speck_suppressor.py` header comment |
| Damage-mask coverage 16.1% → 7.4% on bw.jpg | `mask_refiner.py` header comments (2026-10-20 run) |
| Defect segmenter coverage 5.49% on bw.jpg speck field | Stage 2 log |
| Color bleed fixer corrected 52.6% of bleeding edges on teared | measured probe 2026-10 session |
| Restormer 100 MB VRAM when it fires | stage_registry.py comment |
| SR vs Lanczos lapvar 353 vs 13 on degraded B&W | .freebuff A/B log |
| GPU var: teared speck_count 224 → 411 across runs | PIXREVIVE_START_GUIDE.txt §7 |
| VRAM peak 3405/4096 MB (GTX 1650) | .freebuff/vram_lena.csv (max row 3405 of 4096) |
| CPU lena ~5.8 s | run logs (approx; run.bat timing) |
| EdgeAwareColorBleedFixer / SceneGate / G1 35% texture floor / HF-rollback 0.40 | finish_stages.py, mask_refiner.py comments |

## "Fundamentally limited" claims

| Claim | Basis |
|---|---|
| Missing pixels ⇒ plausible synthesis, reconstructed ≠ recovered | RestorationReport uses recovered/reconstructed categories (backend report schema + docs/HYBRID_DIFFUSION_DESIGN.md) |
| Colorization ambiguous, many colours → same gray | Image-processing first principle; page keeps the L channel intact (color_stages.py) |
| No ground truth → no-GT metrics (NIQE, seam_step, HF, speck) | metrics definitions in quality_analyzer.py / docs/baselines |
| speck-count rewards deleting detail | stated on page and consistent with its definition (tophat count) |
| Faces can drift identity | GFPGAN prior-based reconstruction; wipeout breaker in damage_stages.py |

## Assets added to bundle

| Dir | Size |
|---|---|
| frontend/public/devpage_assets (15 jpg: before/after pairs, case4, teared AB sheet, lena step crops) | 520 KB total — tiny vs bundle; all local, no CDN |
| generated stages.json | from stage_registry.py (correct by construction) |

## Caveats

- `GFPGAN paper` labelled CVPR 2021 (repo convention); CodeFormer NeurIPS 2022 — both as printed in repo docs; links intentionally omitted.
- "Defect segmenter paper lineage" marked UNVERIFIED on the page (see content.js note).
- speck_count input labels: corrected in content.js on 2026-10-20 (lena input 333, teared input 351) after cross-checking pre_fix.json.
