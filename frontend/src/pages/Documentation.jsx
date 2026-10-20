import { ArrowLeft, Cpu, Layers, Workflow, CheckCircle, BookOpen, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import registry from '../generated/stages.json';

// Human-facing copy keyed by stage id. The LIST itself comes from
// frontend/src/generated/stages.json (exported from
// backend/ai/pipeline/stage_registry.py by scripts/export_stage_list.py),
// so this page cannot drift from what the backend actually emits.
const STAGE_EXPLAIN = {
  deskew: 'Hough line-angle voting straightens tilted scans before any detector measures them. Inert on square originals.',
  flat_field: 'Radial/corner-deficit illumination flattening on scans with vignetting; gated on symmetric corner deficit so flat photos are untouched.',
  despeckle: 'ROAD filter + speck-field suppressor remove dust, pinholes and dense low-amplitude specks before any detector can mistake them for emulsion flakes.',
  moire: 'FFT notch filter (HalftoneSuppressor) removes periodic halftone/moiré mesh ahead of damage detection — a periodic mesh is texture, not damage, but every downstream model misread it as content. Self-gating and probe-verified inert on real photos.',
  analyze: 'QualityAnalyzer measures blur, noise σ, damage % and color mode (grayscale / sepia / color), and its routing decision drives which gated stages run.',
  face_detect: 'RetinaFace + Haar cascades find every face with 5-point landmarks once early — for anatomical protection zones and small-face deferral — then again after inpainting to re-detect faces the cracks hid.',
  cast_fix: 'Gray-world/channel-curve rebalance for dye-faded color prints (grayscale/sepia are excluded by type: the sepia tone is historic character).',
  damage_detect: 'U-Net defect segmenter, classical scratch/flake/tear detectors and the foxing dark-spot detector are fused, then refined with scale-invariant kernels and the SceneGate ingress filter (scene-content blobs with interior texture equal to their surround are pruned — they are not damage, and inpainting them is what caused the flat oil-paint failure mode). Facial protection zones are carved out before any pixel is touched.',
  inpaint: 'Big-LaMa fills only the masked pixels; seams are feathered and a face-wipeout circuit breaker aborts inpaints that would erase anatomy. Unmasked pixels are bitwise preserved.',
  damage_cluster_smooth: 'High-variance inpaint clusters get a second targeted smoothing pass (gated on the inpaint coverage that produced them).',
  denoise: 'Three-zone NLM (face / subject / background) scaled to the measured noise floor, with an edge/texture protection map and a high-frequency-retention guard that rolls the stage back when it oversmooths.',
  deblur: 'Restormer motion-deblurring (vendored), strictly gated on measured defocus acutance or motion anisotropy, so inert on the whole real corpus; weights load lazily on the first blur that actually fires the gate.',
  colorize: 'DDColor (ICCV 2023) adds realistic chrominance on grayscale/sepia canvases; ambient color diffuses smoothly across inpaint zones. Hue-histogram and sepia-wash guards roll back to L-preserved gray with disclosure when colorization is not trustworthy.',
  face_restore: 'GFPGAN v1.4 blends every detected face with damage-aware compositing; small (<90 px) faces defer until after upscaling.',
  pre_sr: 'Anti-aliasing (plus deband when quantization stairs are detected) right before super-resolution — the SR step amplifies whatever survives, so this is the last cheap exit.',
  super_res: 'Tiled Real-ESRGAN x4plus upscaling with inpaint-seam harmonization, capped at 4096 px on the long side.',
  sharpen: 'Luminance-only LAB unsharp masking after upscale — chroma stays untouched so sharpening cannot re-ting edges.',
  projector_grade: 'Exhibition CLAHE + shadow-lifting tone curve for projection/display viewing.',
  output_verify: 'Closed-loop verification: residue is RE-detected on the OUTPUT (not trusted from the input-side mask), micro-cleaned, and the color-coherence/texture-plausibility checks feed the honesty report.',
  reimagine: 'Opt-in beta: diffusion re-imagination only for UNRECOVERABLE damage (≥35% of frame or ≥55% of a face destroyed). Evidence contract: pixels recoverable before this stage are bitwise preserved after it; cosine-verified identity guard can roll the whole stage back.',
  refresh_faces: '',
  semantic_validate: '',
};

const GATE_LABEL = {
  enable_repair: 'runs when repair is enabled (default on)',
  enable_denoise: 'runs when denoising is enabled (default on)',
  enable_colorization: 'runs only when colorize is enabled AND the image is grayscale/sepia',
  enable_face_restore: 'runs when face restore is enabled (default on)',
  enable_sr: 'runs when super-resolution is enabled (default on)',
  enable_reimagine: 'beta — opt-in only',
};

const NON_INTERNAL = registry.stages.filter((s) => !['refresh_faces', 'semantic_validate'].includes(s.id));

export default function Documentation() {
  return (
    <div className="w-full max-w-4xl mx-auto px-3.5 sm:px-6 md:px-8 py-6 sm:py-10 md:py-12">
      {/* Back to Home Link */}
      <div className="mb-6 sm:mb-8">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-stone-600 hover:text-stone-900 active:text-black transition-colors text-xs sm:text-sm font-medium min-h-[44px] px-1 py-2"
        >
          <ArrowLeft className="w-4 h-4 shrink-0" />
          <span>Back to Workspace</span>
        </Link>
      </div>

      {/* Page Header */}
      <div className="mb-8 sm:mb-12 border-b border-stone-200 pb-6 sm:pb-8">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-serif text-stone-900 tracking-tight mb-3 sm:mb-4">
          PixRevive Engine Documentation
        </h1>
        <p className="text-sm sm:text-base md:text-lg text-stone-600 font-light max-w-2xl leading-relaxed">
          Understanding the deep learning pipelines, neural architectures, and data flows behind professional archival
          photo restoration.
        </p>
      </div>

      {/* Main Flow Overview Section */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Workflow className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-500 shrink-0" />
          <span>Restoration Pipeline Overview</span>
        </h2>
        <div className="bg-white border border-stone-200 rounded-xl p-4 sm:p-6 mb-6 shadow-xs">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed mb-4">
            PixRevive runs a <strong>22-stage ordered pipeline</strong> (20 user-visible stages + 2 internal orchestration
            stages — post-inpaint face re-detection and semantic plausibility validation — that never emit progress bars,
            and 2 terminal events, loading and complete, that close every stream). The list below is generated from{' '}
            <code className="mx-1 font-mono text-[11px] bg-stone-100 px-1 py-0.5 rounded">
              frontend/src/generated/stages.json
            </code>
            , itself exported from the backend's single source of truth,{' '}
            <code className="mx-1 font-mono text-[11px] bg-stone-100 px-1 py-0.5 rounded">
              backend/ai/pipeline/stage_registry.py
            </code>
            . The ordering is the product: damage is repaired <em>before</em> colorization (DDColor would otherwise
            colorize bright scratch pixels with wrong hues), and impulse noise is suppressed <em>before</em> damage
            detection (specks are indistinguishable from emulsion flakes once they reach the detectors). Several stages
            are conditional: the scan-quality stages (deskew, flat-field, moir&eacute;, cast-fix) self-gate on measured
            content and are inert on clean inputs, while repair, denoise, colorize, faces, and upscale follow the route
            options you pick in the workspace.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 mt-6">
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-4 sm:p-5">
              <span className="text-[10px] bg-indigo-100 border border-indigo-200 text-indigo-700 font-mono px-2 py-0.5 rounded uppercase font-semibold">
                Repair
              </span>
              <h3 className="font-serif text-base sm:text-lg text-stone-800 mt-2 mb-1.5">Damage-first structural repair</h3>
              <p className="text-xs text-stone-600 leading-relaxed">
                Masking &rarr; LaMa inpainting &rarr; denoising, with retinal face detection protecting eyes and mouths
                through every stage. Damaged B&amp;W photos are deliberately routed to repair first, never straight to
                colorize.
              </p>
            </div>
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-4 sm:p-5">
              <span className="text-[10px] bg-amber-100 border border-amber-200 text-amber-700 font-mono px-2 py-0.5 rounded uppercase font-semibold">
                Restore
              </span>
              <h3 className="font-serif text-base sm:text-lg text-stone-800 mt-2 mb-1.5">Colorize, deblur, face-restore, upscale</h3>
              <p className="text-xs text-stone-600 leading-relaxed">
                Only on a clean canvas does DDColor add color, Restormer corrects measured optical blur, GFPGAN rebuilds
                facial detail, and Real-ESRGAN performs the final 4&times; upscale with luminance-only sharpening.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stage table */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-500 shrink-0" />
          <span>Stage-by-Stage Architecture</span>
        </h2>
        <div className="space-y-3">
          {NON_INTERNAL.map((s) => (
            <div
              key={s.id}
              className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 sm:p-4 flex flex-col sm:flex-row gap-2 sm:gap-4 items-start"
            >
              <span className="text-[10px] bg-indigo-100 border border-indigo-200 text-indigo-700 font-mono px-2 py-0.5 rounded uppercase font-semibold shrink-0">
                {s.progress_pct}%
              </span>
              <div className="min-w-0 flex-1">
                <h3 className="font-serif text-sm sm:text-base text-stone-800 mb-1">{s.name}</h3>
                <p className="text-xs text-stone-600 leading-relaxed">{STAGE_EXPLAIN[s.id]}</p>
                {s.route_gate && GATE_LABEL[s.route_gate] && (
                  <p className="text-[11px] text-amber-700 font-mono mt-1">Gate: {GATE_LABEL[s.route_gate]}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Damage masking details */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-500 shrink-0" />
          <span>Multi-Scale Damage Masking</span>
        </h2>
        <div className="space-y-4 sm:space-y-6">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed">
            A learned U-Net segmenter (CVPR 2020 semantic-defect weights) is the damage-detection gatekeeper, fused with
            classical detectors for scratches, white flakes, wide cracks, 3D paper tears, tearmark emulsion flaking and
            foxing/mold dark spots. Before anything is inpainted the fused mask is refined: components are fused with
            scale-invariant morphological kernels (kernel size adapts to image resolution), the SceneGate filter prunes
            blob-like components without luminance popout whose interior texture matches their surround, and ocular/oral
            protection zones are carved out.
          </p>

          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] sm:text-xs p-4 sm:p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner touch-scroll-x">
            <div className="text-indigo-400 mb-2"># Fused damage mask (simplified)</div>
            <div>mask = unet_segmenter(img) | classical_scratches | white_flakes | tearmarks | foxing_spots</div>
            <div className="text-amber-400 mt-2"># pruned: scene-content blobs with no popout and interior texture == surround</div>
            <div>mask = scene_gate(mask, img)</div>
            <div className="text-amber-400 mt-2"># protect anatomy before any pixel is touched</div>
            <div>mask &amp;= ~protect_face_zones(face_landmarks)</div>
            <div className="text-amber-400 mt-2"># Big-LaMa infills only masked pixels; clean pixels are bytewise preserved</div>
            <div>result = lama_inpaint(img, mask)</div>
          </div>
        </div>
      </section>

      {/* Face pipeline details */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Cpu className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-500 shrink-0" />
          <span>Face-Aware Restoration</span>
        </h2>
        <div className="space-y-4 sm:space-y-6">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed">
            RetinaFace + Haar cascades detect <strong>all faces</strong> with 5-point landmarks — once early (to route and
            protect), once after inpainting (to re-cover faces the cracks hid). GFPGAN v1.4 then restores each face with
            damage-aware blending, at fidelity 0.85, and small (&lt;90 px) faces are deferred to a post-upscale pass so
            they are restored at 4&times; resolution instead of being mangled early.
          </p>

          <div className="bg-white border border-stone-200 rounded-xl p-4 sm:p-6 shadow-xs">
            <h4 className="font-serif text-base sm:text-lg text-stone-800 mb-3">Key design guarantees:</h4>
            <ul className="space-y-2 text-xs sm:text-sm text-stone-600">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span>
                  <strong>Clean pixels are never touched:</strong> restoration is composited through the damage mask —
                  original &times; (1&minus;mask) + restored &times; mask.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span>
                  <strong>Identity is gated:</strong> YuNet + SFace cosine verification guards any generative re-imagination,
                  rolling back to the LaMa canvas on failure.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span>
                  <strong>Offline by design:</strong> all weights load from{' '}
                  <code className="bg-stone-100 px-1 py-0.5 rounded text-[11px]">backend/ai/weights/</code> — no hub
                  downloads at runtime.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Colorization details */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 sm:w-6 sm:h-6 text-amber-500 shrink-0" />
          <span>DDColor Colorization</span>
        </h2>
        <div className="space-y-4 sm:space-y-6">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed">
            Colorization uses <strong>DDColor (ICCV 2023)</strong> — when you enable it and the image is grayscale or
            sepia — on the fully repaired canvas. The network predicts chrominance from deep dual-attention features,
            guided by the luminance channel (at 512px input), and ambient color diffuses smoothly across inpainted zones. A
            post-pass reducer fixes banding, rebalances saturation and repairs edge color-bleed. Two honesty guards watch
            the result: a hue-histogram check for the single-hue-wash failure and a sepia background-field check — either
            rolls the image back to L-channel-preserved grayscale and discloses why in the report, because a bad
            colorization is worse than an honest grayscale.
          </p>

          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] sm:text-xs p-4 sm:p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner touch-scroll-x">
            <div className="text-indigo-400 mb-2"># Stage colorize (simplified)</div>
            <div className="text-amber-400 mt-1"># 1. DDColor global transform, L-only input for noise resilience</div>
            <div>colorized = ddcolor_colorizer.colorize(working_image)</div>
            <div className="text-amber-400 mt-2"># 2. ambient chrominance diffusion across inpaint zones</div>
            <div>colorized = diffuse_chroma(colorized, inpaint_mask)</div>
            <div className="text-amber-400 mt-2"># 3. banding / saturation / bleed refinement</div>
            <div>colorized = ColorArtifactReducer().refine(colorized)</div>
            <div className="text-amber-400 mt-2"># 4. guards: single-hue wash and sepia background field roll back to grayscale</div>
          </div>
        </div>
      </section>

      {/* Honesty report */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 sm:w-6 sm:h-6 text-emerald-600 shrink-0" />
          <span>The Restoration Report</span>
        </h2>
        <div className="bg-stone-50 border border-stone-200 rounded-lg p-4 sm:p-6">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed">
            Every run ends with a machine-readable honesty report classifying what happened to your photo:{' '}
            <strong>repaired</strong> (damage removed, original pixels recovered),
            <strong> reconstructed</strong> (generative infill — plausible, not guaranteed accurate), and{' '}
            <strong>flagged</strong> (residual damage worth attention). Colors added by AI are always disclosed as
            plausible, never original.
          </p>
        </div>
      </section>

      {/* Project References & Pipeline Flow */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Workflow className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-500 shrink-0" />
          <span>Pipeline Implementation References</span>
        </h2>
        <div className="bg-white border border-stone-200 rounded-xl p-4 sm:p-6 shadow-xs">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed mb-4">
            The restoration logic is organized across clean, decoupled modules in the backend codebase:
          </p>
          <div className="space-y-3 text-xs font-mono text-stone-600">
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">1. API Entry Point:</strong>
              <code className="break-all">backend/routers/image_router.py</code> &mdash; Receives multipart image uploads,
              exposes the SSE progress stream and result endpoints.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">2. Service Layer Wrapper:</strong>
              <code className="break-all">backend/services/image_service.py</code> &mdash; Singleton pipeline fa&ccedil;ade:
              task store, background execution, status tracking.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">3. Pipeline Orchestrator:</strong>
              <code className="break-all">backend/ai/pipeline/restoration_pipeline.py</code> &mdash; Drives the ordered stage
              registry with monotonic progress reporting.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">4. Stage Contract:</strong>
              <code className="break-all">backend/ai/pipeline/stage_registry.py</code> &mdash; Single source of truth for
              stage ids, names, percentages and route gates.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">5. Colorization Engine:</strong>
              <code className="break-all">backend/ai/colorization/ddcolor_colorizer.py</code> &mdash; DDColor wrapper plus
              chrominance diffusion and artifact reduction.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">6. Restoration Engines:</strong>
              <code className="break-all">backend/ai/restoration/</code> &mdash; LaMa inpainting, GFPGAN faces, Real-ESRGAN
              &times;4.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-800 font-semibold mb-1">7. Honesty Layer:</strong>
              <code className="break-all">backend/ai/report/restoration_report.py</code> &mdash; Classifies recovered /
              reconstructed / flagged evidence.
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Credits */}
      <section className="mb-10 sm:mb-16">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <BookOpen className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-500 shrink-0" />
          <span>Model Credits</span>
        </h2>
        <div className="bg-stone-50 border border-stone-200 rounded-lg p-4 sm:p-6">
          <p className="text-xs sm:text-sm text-stone-700 leading-relaxed mb-4">
            PixRevive builds on published research artifacts:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mt-2">
            <a
              href="https://arxiv.org/abs/2308.11470"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-3 px-3.5 rounded-lg shadow-2xs transition-all flex items-center gap-2 hover:border-stone-300 min-h-[44px]"
            >
              <Layers className="w-4 h-4 text-indigo-500 shrink-0" />
              <span>DDColor &mdash; ICCV 2023</span>
            </a>
            <a
              href="https://arxiv.org/abs/2104.01555"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-3 px-3.5 rounded-lg shadow-2xs transition-all flex items-center gap-2 hover:border-stone-300 min-h-[44px]"
            >
              <Layers className="w-4 h-4 text-indigo-500 shrink-0" />
              <span>Real-ESRGAN &mdash; ICCVW 2021</span>
            </a>
            <a
              href="https://arxiv.org/abs/2101.04027"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-3 px-3.5 rounded-lg shadow-2xs transition-all flex items-center gap-2 hover:border-stone-300 min-h-[44px]"
            >
              <Layers className="w-4 h-4 text-indigo-500 shrink-0" />
              <span>GFPGAN &mdash; CVPR 2021</span>
            </a>
            <a
              href="https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-3 px-3.5 rounded-lg shadow-2xs transition-all flex items-center gap-2 hover:border-stone-300 min-h-[44px]"
            >
              <Layers className="w-4 h-4 text-indigo-500 shrink-0" />
              <span>Restormer &mdash; CVPR 2022</span>
            </a>
            <a
              href="https://arxiv.org/abs/2308.13491"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-3 px-3.5 rounded-lg shadow-2xs transition-all flex items-center gap-2 hover:border-stone-300 min-h-[44px]"
            >
              <Layers className="w-4 h-4 text-indigo-500 shrink-0" />
              <span>LaMa &mdash; WACV 2022</span>
            </a>
          </div>
        </div>
      </section>

      {/* Runtime Device Preference Selection */}
      <section className="mb-10 sm:mb-16 border-t border-stone-200 pt-8 sm:pt-12">
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 mb-4 sm:mb-6 flex items-center gap-2">
          <Cpu className="w-5 h-5 sm:w-6 sm:h-6 text-stone-700 shrink-0" />
          <span>Runtime Device Management</span>
        </h2>
        <p className="text-xs sm:text-sm text-stone-700 leading-relaxed mb-4">
          To ensure deployment versatility, PixRevive supports dynamic device execution routing at the API route level:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 text-sm mt-4">
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 sm:p-4">
            <strong className="block text-stone-800 mb-1 text-xs sm:text-sm">Auto Preference</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Queries CUDA availability on launch. Routes execution to the GPU if active, falling back to CPU if
              unavailable.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 sm:p-4">
            <strong className="block text-stone-800 mb-1 text-xs sm:text-sm">CPU Only</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Forces tensors and model weights to PyTorch CPU device. Best for environments without discrete graphics
              hardware.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 sm:p-4">
            <strong className="block text-stone-800 mb-1 text-xs sm:text-sm">GPU Only</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Forces execution to CUDA. Accelerates inference times by parallelizing matrix convolutions across GPU
              kernels.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
