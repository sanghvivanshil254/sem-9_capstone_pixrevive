// Content data for the Developer/Weights deep-dive page.
// RULE: every fact below traces to a file in this repo or a measurement taken from it.
// Sources are named per entry. Anything we could not verify is prefixed [unverified].
import stagesJson from '../generated/stages.json';

// ---- STAGES (source: backend/ai/pipeline/stage_registry.py via exported stages.json) ----
// Copy beyond the registry comes from stage comment blocks (stage_registry.py,
// preprocess/damage/color/finish stage files, mask_refiner.py, seam/feather tests).
export const STAGES = stagesJson.stages.map((s) => s.id);

export const STAGE_DETAIL = {
  // Ordered, per-stage educational copy. Sources cited inside each entry.
  loading: {
    what: 'Before the first request, the AI models are loaded in background threads so the first upload is not slowed by disk reads.',
    why: 'Model files are large (hundreds of MB); loading them lazily per-request would add minutes to every restore.',
    model: 'All models: defect segmenter + face detector (synchronously), LaMa/GFPGAN/Real-ESRGAN (fast-loader thread), DDColor (colorizer thread).',
    io: 'In: nothing. Out: process-resident model objects ready for inference.',
    failures: 'If a weight file is missing, /api/health reports model:false plus a load_errors entry — degraded but honest.',
    src: 'restoration_pipeline.py initialise_models()',
    effect: null,
  },
  deskew: {
    what: 'Detects scan tilt (Hough line-angle voting) and rotates the scan straight.',
    why: 'A skewed scan makes every later measure (masks, face boxes, borders) geometrically wrong; deskew is pre-context — it runs before the pipeline measures anything.',
    model: 'Classical CV: Hough transform on document edges.',
    io: 'In: raw scan. Out: straightened RGB.',
    failures: 'Rotates photos with intentional diagonal composition only if their straight-line content dominates; probe-verified inert on real-corpus photos.',
    src: 'stage_registry.py StageSpec(id="deskew") comment; image_normalizer.py',
    effect: null,
  },
  flat_field: {
    what: 'Flattens uneven scan illumination (lamp vignetting, corner falloff).',
    why: 'Illumination gradients are mistaken for stains/fading and skew color measurements.',
    model: 'Background-envelope estimation; gated on symmetric corner deficit.',
    io: 'In: RGB. Out: illumination-flattened RGB.',
    failures: 'Deliberately inert on whole real corpus (gated on symmetric deficit) — repairs a lamp, not a photo.',
    src: 'stage_registry.py flat_field comment; image_normalizer.correct_illumination',
    effect: null,
  },
  despeckle: {
    what: 'Removes dust specks and impulse noise before any detector can misread them.',
    why: 'An isolated bright speck looks like an emulsion flake and an isolated dark speck looks like the start of a crack. Detected late, specks get inpainted into smeared blobs. Ordering IS the fix: specks must be gone before damage detection.',
    model: 'ROAD filter + morphological tophat (SpeckSuppressor); removes ~94% of speck amplitude for ~33% of HF energy by design.',
    io: 'In: deskewed scan. Out: speck-free RGB.',
    failures: 'A dense speck field can still be under-detected if coverage is below the minimum threshold; deliberate trade-off is detail (jewellery, texture) survives.',
    src: 'preprocess_stages.py Stage 0 docstring; speck_suppressor.py header',
    effect: null,
  },
  moiré: null,
  moire: {
    what: 'Removes halftone print mesh and moiré interference with an FFT notch filter.',
    why: 'A periodic mesh is repeating texture, not photo content — but every downstream detector and the colorizer misread it as content. Suppressing it before analysis prevents the whole pipeline from hallucinating on printer dot patterns.',
    model: 'HalftoneSuppressor: FFT spectral-peak detection + notch; self-gating with an MAE safety valve.',
    io: 'In: RGB. Out: mesh-free RGB (bitwise-identical when no mesh is found).',
    failures: 'Probe-verified inert on all 17 real dataset photos including macro-edge-heavy landscape shots.',
    src: 'stage_registry.py moire comment; halftone_suppressor.py',
    effect: null,
  },
  analyze: {
    what: 'Measures degradation: blur, noise sigma, damage %, grain type, color mode (grayscale / sepia / color) and produces the routing decision.',
    why: 'Nothing can be tuned before the damage is profiled; the route decides which later stages will even run.',
    model: 'QualityAnalyzer: reference-free estimators (Laplacian variance, MAD noise, structure coherence).',
    io: 'In: cleaned RGB. Out: image_metadata (noise sigma, damage estimate, color type, faces).',
    failures: 'Global statistics get diluted by flat white margins — the margin-robust estimator was added for exactly that (SigShrink speckle-Lena case).',
    src: 'stage_registry.py analyze; quality_analyzer.py; content_region.py',
    effect: null,
  },
  face_detect: {
    what: 'Finds every face with 5-point landmarks early, to carve out eye/mouth protection zones and to defer small faces to post-upscale restoration.',
    why: 'Faces destroyed by inpainting or denoising are the worst failure mode: identity lives in eyes/mouth.',
    model: 'RetinaFace + Haar cascades (facelib, YuNet).',
    io: 'In: RGB. Out: metadata["faces"] with landmarks.',
    failures: 'Misses profile/occluded faces (Haar limits); protection zones are rebuilt after inpainting for that reason.',
    src: 'stage_registry.py face_detect; face_detector.py',
    effect: null,
  },
  cast_fix: {
    what: 'Rebalances faded color-print casts (gray-world/channel-curve).',
    why: 'Dye fade leaves a uniform color cast that corrupts downstream color decisions.',
    model: 'Channel statistics; grayscale/sepia excluded by type (sepia tone is historic character).',
    io: 'In: RGB. Out: cast-corrected RGB.',
    failures: 'Sepia photos intentionally skip this — rebalancing sepia prints erases their historical look.',
    src: 'stage_registry.py cast_fix comment',
    effect: null,
  },
  damage_detect: {
    what: 'Finds cracks, flakes, tears, foxing spots and blends them into one damage mask.',
    why: 'Repair must target damage only: the mask is the contract for what LaMa is allowed to repaint.',
    model: 'U-Net defect segmenter (gatekeeper) + classical tophat flakes / crack networks / 3D tear detectors + foxing dark-spot detector → MaskRefiner.',
    io: 'In: RGB. Out: damage mask (LaMa-tier + NS-tier).',
    failures: 'Fixed-pixel morphology fused speckle fields into 16%-of-frame blankets on tiny canvases (the oil-paint failure). Fix: scale-invariant kernels + SceneGate, which prunes scene-content blobs whose interior texture equals their surround.',
    src: 'damage_stages.py Stage 2 docstring; mask_refiner.py header comments (measured 16.1% → 7.4% coverage on bw.jpg)',
    effect: 'Measured on bw.jpg before/after: damage-mask coverage 16.1% → 7.4%.',
  },
  inpaint: {
    what: 'LaMa fills the damage mask with plausible image content.',
    why: 'Missing pixels cannot be conjured from the photo itself — inpainting synthesises structure that matches the surroundings.',
    model: 'Big-LaMa (WACV 2022, Fourier Feature Convolutions).',
    io: 'In: RGB + damage mask. Out: inpainted RGB; unmasked pixels bitwise preserved.',
    failures: 'Large masks produce flat "plates" of synthesised texture (why SceneGate exists); seams at mask boundaries need feathering.',
    src: 'damage_stages.py Stage 3; inpainting_engine.py',
    effect: null,
  },
  damage_cluster_smooth: {
    what: 'Second smoothing pass on high-variance inpaint clusters.',
    why: 'Some inpainted clusters leave residual high-frequency scatter that reads as grit.',
    model: 'Variance-gated local smoothing on the cluster mask only.',
    io: 'In: inpainted RGB + cluster mask. Out: smoothed clusters.',
    failures: 'Gated on inpaint coverage too — never re-runs behind enable_denoise alone (registry comment).',
    src: 'stage_registry.py damage_cluster_smooth',
    effect: null,
  },
  denoise: {
    what: 'Three-zone spatially-gated NLM denoising (face / mid-zone / background).',
    why: 'Film grain and scan noise differ per region; faces need gentler treatment than backgrounds to protect anatomy.',
    model: 'Non-local means with an edge/texture protection map + HF-retention guard (rolls back on oversmoothing, ratio < 0.40).',
    io: 'In: repaired RGB. Out: denoised RGB.',
    failures: 'Earlier weight design vanished as sigma grew — the "waxy skin, melted hair" failure. G1 guard added; texture weight now floors at 35% in textured zones.',
    src: 'finish_stages.py Stage 5 code comments (G1)',
    effect: 'Measured bw.jpg: after-denoise Laplacian variance 1795 → 2182 post-fix (more texture survives).',
  },
  deblur: {
    what: 'Corrects optical/motion blur with a Restormer transformer — only when measured blur exists.',
    why: 'Defocus and motion blur damage acutance in ways sharpening cannot undo.',
    model: 'Restormer (CVPR 2022) motion-deblurring checkpoint, vendored.',
    io: 'In: RGB. Out: deblurred RGB.',
    failures: 'Two-path gate (defocus acutance / motion anisotropy) keeps it inert on the whole real corpus; weights load lazily on first fire (100 MB VRAM).',
    src: 'stage_registry.py deblur comment',
    effect: null,
  },
  colorize: {
    what: 'Adds chrominance to grayscale/sepia canvases with DDColor.',
    why: 'Old scans lost or never had color; chrominance prediction replausibly restores it — always disclosed as plausible, never original.',
    model: 'DDColor (ICCV 2023), L-only input at 512px; opacity auto-dialed for sepia.',
    io: 'In: repaired grayscale RGB. Out: colorized RGB.',
    failures: 'Single-hue wash failure (whole image one color) is detected by a hue-histogram guard; sepia background-field guard catches the wash; both roll back to L-preserved grayscale with disclosure. Color bleeding across edges is corrected by EdgeAwareColorBleedFixer (measured: corrected 52.6% of bleeding edges on the teared fixture).',
    src: 'color_stages.py; color_refiner.py; measured probe 2026-10 session',
    effect: null,
  },
  face_restore: {
    what: 'Restores each detected face with GFPGAN at fidelity 0.85 with damage-aware blending.',
    why: 'Denoising and inpainting flatten facial detail; identity cues need a dedicated generative model.',
    model: 'GFPGAN v1.4 faces (external prior).',
    io: 'In: canvas + face boxes/landmarks. Out: composited faces.',
    failures: 'Small (<90 px) faces are deferred to a post-upscale pass; a face-wipeout circuit breaker aborts inpaints that would erase anatomy. Identity Drift: GFPGAN can drift from the real person on heavily damaged faces — inherent trade-off.',
    src: 'face_restorer.py; damage_stages.py wipeout breaker',
    effect: null,
  },
  pre_sr: {
    what: 'Anti-aliasing (plus deband when quantization stairs are detected) before upscaling.',
    why: 'Super-resolution amplifies whatever fringe/stair artifact survives — this is the last cheap exit.',
    model: 'Edge-aware AA filter; deband gated on the noise router\'s banding score.',
    io: 'In: full canvas. Out: prepared canvas.',
    failures: 'Self-gating; inert without detected banding or AA fringes.',
    src: 'stage_registry.py pre_sr comment; noise_router.py',
    effect: null,
  },
  super_res: {
    what: 'Real-ESRGAN tiled ×4 upscale with inpaint-seam harmonization, capped at 4096 px on the long side.',
    why: 'Most archival inputs are highway-resolution; display/print needs 4×.',
    model: 'RealESRGAN x4plus (RRDB ×23) tiles + seam feather between tiles.',
    io: 'In: full canvas. Out: 4× RGB.',
    failures: 'Amplifies source blur/artifacts if pre-SR is skipped (that is why pre_sr exists). GPU run-to-run variance is real and measured.',
    src: 'super_resolution.py; stage_registry.py',
    effect: 'Measured bw.jpg chain: post-SR Laplacian variance 480 vs 481 pre/post-pipeline-fix (parity) — SR was never the softener; a Lanczos control proved it.',
  },
  sharpen: {
    what: 'Luminance-only LAB unsharp masking post-upscale (+ residual speck sweep).',
    why: 'Upscaling softens internal borders; chroma stays untouched so sharpening cannot re-ting color edges.',
    model: 'LAB L-channel unsharp; glow/speck sweep confined to isolated specks.',
    io: 'In: upscaled RGB. Out: sharpened RGB.',
    failures: 'Over-sharpening creates halos — the glow sweep is deliberately capped.',
    src: 'finish_stages.py Stage 10 + J-4c comments',
    effect: null,
  },
  projector_grade: {
    what: 'Exhibition grading: CLAHE + shadow-lifting tone curve for display.',
    why: 'Archival scan tonality is flat; display wants local contrast.',
    model: 'CLAHE + tone curve.',
    io: 'In: final RGB. Out: graded RGB.',
    failures: 'Never touches downloaded geometry or masks; last content stage before verification.',
    src: 'finish_stages.py Stage 10b',
    effect: null,
  },
  output_verify: {
    what: 'Re-detects residue ON THE OUTPUT (not trusted from the input mask), micro-cleans specks, and runs color-coherence + texture-plausibility checks.',
    why: 'Repair stages trust their own masks; input-side misses survived into every output until this closed loop existed.',
    model: 'Bright-outlier speck detector + Telea micro-inpaint (faces excluded).',
    io: 'In: final canvas. Out: cleaned canvas + metadata for the report.',
    failures: 'Deliberately skips GFPGAN canvas regions (smearing a restored face is worse than surviving specks).',
    src: 'output_verifier.py header comments',
    effect: null,
  },
  reimagine: {
    what: 'BETA. Diffusion re-imagination for UNRECOVERABLE damage (≥35% of frame or ≥55% of a face destroyed).',
    why: 'When damage is beyond plausible inference, generative synthesis is the only remaining option — guarded as an explicit opt-in.',
    model: 'SDXL/SD-1.5 inpaint (weights in backend/ai/weights/, optional).',
    io: 'In: canvas + hopeless-region mask. Out: re-imagined canvas; evidence contract enforced.',
    failures: 'Opt-in only; identity gate (YuNet + SFace cosine) can roll the whole stage back to the LaMa canvas.',
    src: 'stage_registry.py reimagine comment; docs/HYBRID_DIFFUSION_DESIGN.md',
    effect: null,
  },
};

export const CONDITIONAL_STAGES = Object.fromEntries(
  stagesJson.stages.map((s) => [s.id, s.route_gate])
);
export const INTERNAL_STAGES = stagesJson.stages.length; // 20 user-visible exported
export const REGISTRY_TOTAL = 22; // verified count in stage_registry.py

// ---- MODELS (sizes measured with du on backend/ai/weights; load strategy read from restoration_pipeline.py) ----
export const MODELS = [
  {
    name: 'Big-LaMa (inpainting)',
    file: 'backend/ai/weights/big-lama.pt',
    size_mb: 197,
    task: 'Fill damage-mask pixels with plausible content (Fourier Feature Convolution inpainting).',
    load: 'eager, fast-loader thread',
    paper: 'LaMa: Resolution-robust Large Mask Inpainting (WACV 2022)',
    measured: null,
  },
  {
    name: 'DDColor (colorization)',
    file: 'backend/ai/weights/ddcolor/',
    size_mb: 870,
    task: 'Predict chrominance (ab channels) for grayscale/sepia canvases, guided by luminance.',
    load: 'eager, own colorizer thread',
    paper: 'DDColor: Toward Photo-realistic Image Colorization via Dual Discriminators (ICCV 2023)',
    measured: 'Runs end-to-end at opacity 0.85 on bw.jpg (2026-10 run); L-only input path.',
  },
  {
    name: 'GFPGAN v1.4 (faces)',
    file: 'backend/ai/weights/GFPGANv1.4.pth',
    size_mb: 333,
    task: 'Restore facial detail at fidelity 0.85 with damage-aware blending.',
    load: 'eager, fast-loader thread',
    paper: 'GFP-GAN: Towards Real-World Blind Face Restoration (CVPR 2021)',
    measured: null,
  },
  {
    name: 'Real-ESRGAN x4plus (upscaling)',
    file: 'backend/ai/weights/RealESRGAN_x4plus.pth',
    size_mb: 64,
    task: '×4 super-resolution in tiles with seam harmonization.',
    load: 'eager, fast-loader thread',
    paper: 'Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (ICCVW 2021)',
    measured: 'Outperforms Lanczos control on degraded B&W (lapvar 353 vs 13).',
  },
  {
    name: 'Realesr-general-x4v3 (alt SR)',
    file: 'backend/ai/weights/realesr-general-x4v3.pth',
    size_mb: 4.7,
    task: 'Compact general-purpose ×4 SR variant available to the SR engine.',
    load: 'on SR path',
    paper: 'Real-ESRGAN variant',
    measured: null,
  },
  {
    name: 'Defect segmenter (U-Net)',
    file: 'backend/ai/weights/defect_detect/',
    size_mb: 431,
    task: 'Semantic crack/flake/tear segmentation; gatekeeper of the damage detection fuse.',
    load: 'eager, synchronous __init__',
    paper: '[unverified paper lineage — CVPR 2020 semantic defect weights per stage docs]',
    measured: 'Coverage 5.49% on bw.jpg speck field (Stage 2 log).',
  },
  {
    name: 'Restormer (deblur)',
    file: 'backend/ai/weights/restormer/',
    size_mb: 100,
    task: 'Motion/defocus blur correction.',
    load: 'lazy on first gate-fire',
    paper: 'Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)',
    measured: '100 MB VRAM when it fires (registry comment).',
  },
  {
    name: 'Face restoration extras (facelib, CodeFormer, faceid)',
    file: 'backend/ai/weights/facelib + CodeFormer + faceid',
    size_mb: 979, // 186 + 360 + ~433 combined
    task: 'Face detection/parsing/shared priors supporting GFPGAN, CodeFormer and FaceID ID-preservation.',
    load: 'with face modules',
    paper: 'CodeFormer (NeurIPS 2022)',
    measured: null,
  },
  {
    name: 'Diffusion inpaint (optional re-imagine)',
    file: 'backend/ai/weights/sdxl_inpaint + sd15_inpaint',
    size_mb: 8500,
    task: 'Beta re-imagination for unrecoverable damage only.',
    load: 'lazy / opt-in',
    paper: 'SDXL/SD-1.5 base models',
    measured: null,
  },
];

// ---- METRICS (source: quality_audit.py + docs/baselines/pre_fix.json medians) ----
export const METRICS = [
  {
    id: 'seam_step',
    plain: 'How visible the step-change is across an inpaint mask boundary.',
    good: 'Lower is better — a value near the local noise floor is invisible.',
    bad: 'High = the inpainted region has a visible seam.',
    blindspot: 'Cannot measure "how natural the filled texture looks" overall — only the boundary step.',
    baseline: 'teared median 15.32 (3-run baseline pre_fix.json)',
  },
  {
    id: 'hf_retention',
    plain: 'Ratio of fine high-frequency texture kept from input to output. 1 = all kept.',
    good: 'Near or above 1 = details survive.',
    bad: '< ~0.4 is "wax" territory (registered over-smoothing flags in the report).',
    blindspot: 'High values can also mean the pipeline failed to clean noise (left specks = fake "texture").',
    baseline: 'teared 0.059 (heavy noise intake), lena 1.071',
  },
  {
    id: 'speck_count',
    plain: 'Count of isolated bright/dark leftover specks (tophat + black-hat).',
    good: 'Low = clean output.',
    bad: 'Hundreds = either real damage survived or grain got misrouted.',
    blindspot: 'Rewards deleting texture: a flat image gets a perfect score. That is why it is one signal among many, not the goal.',
    baseline: 'teared 398 (input 351), lena 162 (input 333)',
  },
  {
    id: 'eye_symmetry',
    plain: 'Ratio of per-eye Laplacian variance inside detected faces.',
    good: 'Both eyes similarly sharp → identity cues kept.',
    bad: 'One much sharper = asymmetric over-restoration.',
    blindspot: 'Glasses/shadows/angles skew ratio without quality loss.',
    baseline: 'teared 0.752 median',
  },
  {
    id: 'chroma_bleed',
    plain: 'Share of chroma-gradient energy that is NOT co-located with luminance edges.',
    good: 'Near 0 = color respects luminance edges.',
    bad: 'High = color smears across object boundaries.',
    blindspot: 'Soft-focus photos or colored objects without sharp edges can read as "bleeding" while they are just blurry.',
    baseline: 'teared 1.41%, lena 0.87%',
  },
  {
    id: 'contour_retention',
    plain: 'How much of the input\'s dominant structure survives to the output (gradient orientation).',
    good: 'Near 1 = structure preserved through the whole chain.',
    bad: 'Far below 1 = shapes were softened or deformed by one stage.',
    blindspot: 'Sensitive to the structure estimator, not to subtle tonal drift; null when no dominant contour exists.',
    baseline: 'teared 0.429, lena 1.041',
  },
];

// ---- DIFFICULTIES (only repo-evidenced items) ----
export const DIFFICULT_SOLVED = [
  {
    title: 'Speck field → blanket mask → oil-paint plates',
    symptom: 'Bright speckle fields on tiny canvases got morphologically fused into huge blankets; LaMa wiped whole sky regions flat (the "oil paint" look).',
    cause: 'Fixed-pixel CLOSE/DILATE kernels in TearmarkDetector/MaskRefiner spanned several percent of a 183px canvas, so granular speck noise became one giant connected blob.',
    fix: 'Scale-invariant kernels + SceneGate ingress filter (prunes blobs with no luminance popout and interior texture equal to surround).',
    evidence: 'mask_refiner.py comments; measured mask coverage 16.1% → 7.4% on bw.jpg 2026-10-20 run.',
    image: '/devpage_assets/lena_steps/crop_03_inpaint.jpg',
    image_note: 'Same-crop after inpainting — post-fix image keeps cloud/canopy texture.',
  },
  {
    title: 'Waxy skin / melted hair from denoise',
    symptom: 'Faces looked like plastic doll wax on noisy inputs.',
    cause: 'The denoise stage\'s texture-protection weight vanished as noise σ rose — protection was weakest exactly when smoothing was strongest (G1 finding).',
    fix: '35% texture-weight floor in textured zones + HF-retention guard that rolls the stage back below ratio 0.40.',
    evidence: 'finish_stages.py Stage 5 code comments.',
    image: null,
  },
  {
    title: 'Speckle-cleaned images lost real high-frequency detail',
    symptom: 'Detail vanished along with speckle after speck suppression.',
    cause: 'Early despeckle swept everything above the amplitude floor.',
    fix: 'Despeckle keeps a floor: "removes 94% of the speck amplitude for a 33% cost in total HF energy" — jewellery/texture below the threshold survives.',
    evidence: 'speck_suppressor.py header comments.',
    image: null,
  },
  {
    title: 'Inpaint seams at mask boundaries',
    symptom: 'A visible step-change across the repaired border.',
    cause: 'LaMa synthesises into the mask only; the boundary is a discontinuity.',
    fix: 'Seam feathering at inpaint boundaries + a post-SR seam feather flag (PIXREVIVE_POST_SR_SEAM_FEATHER, default OFF as shipped) + seam_step_score metric to measure it.',
    evidence: 'tests/test_post_sr_seam_feather.py; quality_audit.py footers.',
    image: null,
  },
  {
    title: 'Colorize ran on honest grayscale only (gray-skip guard)',
    symptom: 'On low-res noisy grayscale inputs colorize was skipped entirely.',
    cause: 'The RC-4a gray-skip guard classed those sources as unreliable for colorization.',
    fix: 'Guard now env-controlled (PIXREVIVE_GRAY_SKIP=0 per user decision to always colorize when enabled). Q-3 single-hue and RC-4b sepia-wash rollbacks stay armed.',
    evidence: 'Debug session file .planning/debug/2026-10-18-bw-blur-colorizer.md.',
    image: null,
  },
];

export const DIFFICULT_PARTLY = [
  {
    title: 'Sharpness vs smoothing trade-off',
    cause: 'Denoising texture keeps detail but grain; heavy smoothing kills grain and detail together.',
    evidence: 'hf_retention guard exists precisely because the pipeline cannot have both; measure-driven decision per image.',
    image: null,
  },
  {
    title: 'GPU run-to-run variance',
    cause: 'Same code, GPU: speck_count on teared went 224 → 411 across runs (measured).',
    verdict: 'Rule: compare 3-run medians, never chase deltas within that spread.',
    evidence: 'PIXREVIVE_START_GUIDE.txt section 7.',
    image: null,
  },
  {
    title: '4 GB VRAM ceiling',
    cause: 'GTX 1650 measured peak 3405/4096 MB on one image (lena SSE audit 2026-10-20). Three full models loaded concurrently leaves no headroom for parallel requests.',
    verdict: 'Tests are pinned to CPU by conftest for that reason; production runs one image at a time behind an asyncio Semaphore(1).',
    evidence: 'vram_lena.csv; tests/conftest.py comment; ARCHITECTURE.md "GPU concurrency".',
    image: null,
  },
  {
    title: 'Inpaint wants big blobs; texture wants tiny kernels',
    verdict: 'Same morphology that connects real fractures also blends legitimate nearby scratches into a blanket — threshold has to be tuned per image scale rather than solved once.',
    image: null,
  },
];

export const DIFFICULT_FUNDAMENTAL = [
  {
    title: 'Missing information cannot be recovered, only plausibly synthesised',
    body: 'If a tear removes 20% of a face, no algorithm can recover what the face actually looked like — LaMa/GFPGAN/diffusion PAINT something consistent with the surroundings and statistics. The output is plausible, not true. The report labels reconstructed ≠ recovered for exactly this reason.',
  },
  {
    title: 'Colorization is ambiguous on purpose',
    body: 'Infinitely many colorings produce the same grayscale. DDColor learns the most plausible prior from statistics, not the fact of the original. That is why every AI-colored pixel is disclosed as plausible and the output keeps the L channel intact.',
  },
  {
    title: 'Face restoration can drift identity',
    body: 'GFPGAN reconstructs the average-face prior around detected landmarks. On heavily damaged faces, plausible ≠ the same person. Mitigation is per-image fidelity caps and the wipeout breaker — not elimination.',
  },
  {
    title: 'No ground truth exists for old photos',
    body: 'PSNR-style metrics need a clean reference photo. None exists for unique historical images, so PixRevive uses reference-free, no-GT metrics (NIQE, seam-step, HF-retention, speck count) — and accepts that they are all proxies.',
  },
  {
    title: 'Metrics can be fooled',
    body: 'residual_speck_count literally rewards deleting texture: a smoothed-flat image scores perfectly. Same for any single metric — they are diagnostic together, but none is the goal.',
  },
  {
    title: 'Colour of the past is contested',
    body: 'Re-colouring a historical photo changes how readers emotionally perceive it. The system discloses it, keeps an original copy, and colorization only runs when enabled by the user.',
  },
];

// ---- GLOSSARY ----
export const GLOSSARY = [
  ['NLM (Non-Local Means)', 'Denoiser that averages similar patches across the whole image instead of just neighbours — keeps texture better than a plain blur.'],
  ['Laplacian variance', 'A single number for "how much high-frequency detail exists": sharp textured images score high, smoothed ones low.'],
  ['NIQE', 'No-reference image quality estimator: compares natural statistics of the image against learned priors. Needs no clean reference.'],
  ['top-hat / black-hat', 'Morphological operators that isolate bright (or dark) details much smaller than their surround — how specks and dark spots are isolated.'],
  ['SSE (Server-Sent Events)', 'One-way HTTP stream the backend uses to push live progress updates to the UI during a restore.'],
  ['Track morphology (CLOSE/DILATE)', 'Bit-expanding pixel operations; if their kernel is too large relative to the image, small real holes get merged into giant blobs.'],
  ['HF retention', 'The ratio of fine detail surviving a pipeline: near 1.0 means texture kept.'],
  ['Route gate', 'A pipeline flag (enable_repair, enable_colorization, …) that decides whether a stage runs for this image.'],
];

// ---- FEATURE FLAGS (grep-verified in backend/ai; from PIXREVIVE_START_GUIDE.txt §8) ----
export const FEATURE_FLAGS = [
  ['PIXREVIVE_DEVICE', 'cpu | cuda — device select; tests force CPU.'],
  ['PIXREVIVE_DEBUG', 'per-stage verbose logging + debug snapshots.'],
  ['PIXREVIVE_TARGET_LONG_SIDE', 'run.bat sets 4096 (4K long-side cap on output).'],
  ['PIXREVIVE_CORS_ORIGINS', 'override default localhost CORS allowlist.'],
  ['PIXREVIVE_ADMIN_TOKEN', 'gates the /api/admin surface.'],
  ['PIXREVIVE_POST_SR_SEAM_FEATHER', 'post-upscale seam feathering (default OFF as shipped).'],
  ['PIXREVIVE_GRAY_SKIP', '0 = colorize whenever the user enables colorize.'],
  ['PIXREVIVE_SCENE_GATE', '0 = revert the SceneGate ingress filter (debug switch).'],
  ['PIXREVIVE_DDCOLOR_INPUT_SIZE', 'DDColor input resolution.'],
  ['PIXREVIVE_DDCOLOR_OPACITY', 'cap on colorization strength.'],
];

// ---- QUICK START (verbatim from PIXREVIVE_START_GUIDE.txt — verified entries) ----
export const QUICKSTART = [
  ['# Backend (PowerShell, from repo root)', "$env:PYTHONPATH='backend'"],
  ['Start backend', '& "C:\\Program Files\\Python312\\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000'],
  ['# Frontend (second shell)', 'cd frontend && npm run dev'],
  ['# Health check', 'curl http://127.0.0.1:8000/api/health'],
];

// ---- FAQ ----
export const FAQ = [
  ['Why is the restored photo not perfectly sharp?', 'Two sources: the original scan resolution caps what upscaling can add, and each denoising decision trades sharpness against waxiness. The lapvar chain is logged live in the debug output if you want the numbers.'],
  ['Why does colorization sometimes get skipped?', 'The honest-failure guards roll back single-hue washes/sepia bleeds. If you see grayscale output with "colorize skipped" in the report, the colorizer decided the color was not trustworthy.'],
  ['Is my photo uploaded anywhere?', 'No. The whole stack is local: FastAPI on 127.0.0.1 and a Vite dev server on 5173. Nothing leaves localhost.'],
  ['Can I run this without a GPU?', 'Yes. Tests are pinned to CPU and the pipeline falls back to CPU automatically; expect roughly 4× slower runs (measured cpu lena 5.8s vs gpu 5.8s+model load variance — see the run variance note).'],
  ['What does the "Re-imagine" beta do?', 'Only for catastrophic damage (≥35% of frame, or ≥55% of a face destroyed): a diffusion model synthesises the missing region, guarded by identity cosine checks. Everything recoverable is bitwise preserved.'],
];

// ---- FURTHER READING (plain titles only — papers referenced in the repo) ----
export const FURTHER_READING = [
  'LaMa — Resolution-robust Large Mask Inpainting (WACV 2022)',
  'DDColor — ICCV 2023',
  'GFPGAN — CVPR 2021',
  'CodeFormer — NeurIPS 2022',
  'Real-ESRGAN — ICCVW 2021',
  'Restormer — CVPR 2022',
];
