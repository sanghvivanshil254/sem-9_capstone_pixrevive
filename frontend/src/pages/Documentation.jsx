import { ArrowLeft, Cpu, Layers, Workflow, CheckCircle, BookOpen, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

const STAGES = [
  ['0', 'Impulse & speck suppression', 'ROAD filter + speck-field suppressor remove dust, pinholes and dense low-amplitude specks before any detector can mistake them for emulsion flakes.'],
  ['1', 'Degradation analysis', 'QualityAnalyzer measures blur, noise σ, damage % and color mode (grayscale / sepia / color) to route the image.'],
  ['2', 'Multi-scale fracture masking', 'U-Net defect segmenter, classical scratch/flake/tear detectors and the foxing dark-spot detector are fused and refined into the damage mask.'],
  ['3', 'LaMa deep inpainting', 'Big-LaMa fills the damage mask; seams are feathered and a face-wipeout circuit breaker aborts inpaints that would erase anatomy.'],
  ['5', 'Spatially-gated denoising', 'Three-zone NLM (face / subject / background) scaled to the measured noise floor, preserving facial anatomy.'],
  ['6', 'DDColor colorization', 'DDColor (ICCV 2023) adds realistic chrominance; ambient color diffuses smoothly across inpaint zones.'],
  ['7', 'GFPGAN face restoration', 'GFPGAN v1.4 blends every detected face with damage-aware compositing; small faces defer until after upscaling.'],
  ['9', 'Real-ESRGAN ×4 upscale', 'Tiled Real-ESRGAN x4plus super-resolution with inpaint-seam harmonization, capped at 4096 px.'],
  ['10', 'Sharpen & grade', 'Luminance-only LAB unsharp masking, then exhibition CLAHE + shadow-lifting tone curve.'],
];

export default function Documentation() {
  return (
    <div className="max-w-4xl mx-auto px-4 md:px-8 py-12">
      {/* Back to Home Link */}
      <div className="mb-8">
        <Link to="/" className="inline-flex items-center gap-2 text-stone-500 hover:text-stone-900 transition-colors text-sm font-medium">
          <ArrowLeft className="w-4 h-4" /> Back to Workspace
        </Link>
      </div>

      {/* Page Header */}
      <div className="mb-12 border-b border-stone-200 pb-8">
        <h1 className="text-4xl md:text-5xl font-serif text-stone-900 tracking-tight mb-4">
          PixRevive Engine Documentation
        </h1>
        <p className="text-lg text-stone-600 font-light max-w-2xl leading-relaxed">
          Understanding the deep learning pipelines, neural architectures, and data flows behind professional archival photo restoration.
        </p>
      </div>

      {/* Main Flow Overview Section */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Workflow className="w-6 h-6 text-indigo-500" /> Restoration Pipeline Overview
        </h2>
        <div className="cream-card p-6 mb-6">
          <p className="text-stone-700 leading-relaxed mb-4">
            PixRevive runs a <strong>14-stage ordered pipeline</strong>. The ordering is the product:
            damage is repaired <em>before</em> colorization (DDColor would otherwise colorize bright scratch
            pixels with wrong hues), and impulse noise is suppressed <em>before</em> damage detection
            (specks are indistinguishable from emulsion flakes once they reach the detectors).
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-5">
              <span className="text-xs bg-indigo-100 border border-indigo-200 text-indigo-700 font-mono px-2 py-0.5 rounded uppercase font-semibold">Repair</span>
              <h3 className="font-serif text-lg text-stone-800 mt-2 mb-1.5">Damage-first structural repair</h3>
              <p className="text-xs text-stone-500 leading-relaxed">
                Masking → LaMa inpainting → denoising, with retinal face detection protecting eyes and
                mouths through every stage. Damaged B&amp;W photos are deliberately routed to repair
                first, never straight to colorize.
              </p>
            </div>
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-5">
              <span className="text-xs bg-amber-100 border border-amber-200 text-amber-700 font-mono px-2 py-0.5 rounded uppercase font-semibold">Restore</span>
              <h3 className="font-serif text-lg text-stone-800 mt-2 mb-1.5">Colorize, face-restore, upscale</h3>
              <p className="text-xs text-stone-500 leading-relaxed">
                Only on a clean canvas does DDColor add color, GFPGAN rebuild facial detail, and
                Real-ESRGAN perform the final 4× upscale with luminance-only sharpening.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stage table */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Layers className="w-6 h-6 text-indigo-500" /> Stage-by-Stage
        </h2>
        <div className="space-y-3">
          {STAGES.map(([id, title, body]) => (
            <div key={id} className="bg-stone-50 border border-stone-200 rounded-lg p-4 flex gap-4">
              <span className="text-xs bg-indigo-100 border border-indigo-200 text-indigo-700 font-mono px-2 py-0.5 rounded uppercase font-semibold shrink-0 h-fit">Stage {id}</span>
              <div>
                <h3 className="font-serif text-base text-stone-800 mb-1">{title}</h3>
                <p className="text-xs text-stone-500 leading-relaxed">{body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Damage masking details */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Layers className="w-6 h-6 text-indigo-500" /> Multi-Scale Damage Masking
        </h2>
        <div className="space-y-6">
          <p className="text-stone-700 leading-relaxed">
            A learned U-Net segmenter (CVPR 2020 semantic-defect weights) is the Stage 2 gatekeeper,
            fused with classical detectors for scratches, white flakes, wide cracks, 3D paper tears,
            tearmark emulsion flaking and foxing/mold dark spots. The fused mask is refined
            (component filtering, hole closing, controlled dilation) and ocular/oral protection zones
            carved out before inpainting.
          </p>

          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner">
            <div className="text-indigo-400 mb-2"># Fused damage mask (simplified)</div>
            <div>mask = unet_segmenter(img) | classical_scratches | white_flakes | tearmarks | foxing_spots</div>
            <div className="text-amber-400 mt-2"># protect anatomy before any pixel is touched</div>
            <div>mask &= ~protect_face_zones(face_landmarks)</div>
            <div className="text-amber-400 mt-2"># Big-LaMa infills only masked pixels; clean pixels are bytewise preserved</div>
            <div>result = lama_inpaint(img, mask)</div>
          </div>
        </div>
      </section>

      {/* Face pipeline details */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-indigo-500" /> Face-Aware Restoration
        </h2>
        <div className="space-y-6">
          <p className="text-stone-700 leading-relaxed">
            RetinaFace + Haar cascades detect <strong>all faces</strong> with 5-point landmarks — once
            early (to route and protect), once after inpainting (to re-cover faces the cracks hid).
            GFPGAN v1.4 then restores each face with damage-aware blending, at fidelity 0.85, and
            small (&lt;90 px) faces are deferred to a post-upscale pass so they are restored at 4×
            resolution instead of being mangled early.
          </p>

          <div className="cream-card p-6">
            <h4 className="font-serif text-lg text-stone-800 mb-3">Key design guarantees:</h4>
            <ul className="space-y-2 text-sm text-stone-600">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span><strong>Clean pixels are never touched:</strong> restoration is composited through the damage mask — original × (1−mask) + restored × mask.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span><strong>Identity is gated:</strong> YuNet + SFace cosine verification guards any generative re-imagination, rolling back to the LaMa canvas on failure.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span><strong>Offline by design:</strong> all weights load from <code>backend/ai/weights/</code> — no hub downloads at runtime.</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Colorization details */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Layers className="w-6 h-6 text-amber-500" /> DDColor Colorization
        </h2>
        <div className="space-y-6">
          <p className="text-stone-700 leading-relaxed">
            Colorization uses <strong>DDColor (ICCV 2023)</strong> on the fully repaired canvas. The
            network predicts chrominance from deep dual-attention features, guided by the luminance
            channel, and ambient color diffuses smoothly across inpainted zones. A post-pass reducer
            fixes banding, rebalances saturation and repairs edge color-bleed; a hue-histogram guard
            detects the single-hue wash failure mode and rolls back to L-preserved grayscale with
            disclosure in the report.
          </p>

          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner">
            <div className="text-indigo-400 mb-2"># Stage 6 (simplified)</div>
            <div className="text-amber-400 mt-1"># 1. DDColor global transform at 512px input</div>
            <div>colorized = ddcolor_colorizer.colorize(working_image)</div>
            <div className="text-amber-400 mt-2"># 2. ambient chrominance diffusion across inpaint zones</div>
            <div>colorized = diffuse_chroma(colorized, inpaint_mask)</div>
            <div className="text-amber-400 mt-2"># 3. banding / saturation / bleed refinement</div>
            <div>colorized = ColorArtifactReducer().refine(colorized)</div>
          </div>
        </div>
      </section>

      {/* Honesty report */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <ShieldCheck className="w-6 h-6 text-emerald-600" /> The Restoration Report
        </h2>
        <div className="bg-stone-50 border border-stone-200 rounded-lg p-6">
          <p className="text-stone-700 leading-relaxed text-sm">
            Every run ends with a machine-readable honesty report classifying what happened to your
            photo: <strong>repaired</strong> (damage removed, original pixels recovered),
            <strong> reconstructed</strong> (generative infill — plausible, not guaranteed accurate),
            and <strong>flagged</strong> (residual damage worth attention). Colors added by AI are
            always disclosed as plausible, never original.
          </p>
        </div>
      </section>

      {/* Project References & Pipeline Flow */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Workflow className="w-6 h-6 text-indigo-500" /> Pipeline Implementation References
        </h2>
        <div className="cream-card p-6">
          <p className="text-stone-700 leading-relaxed mb-4 text-sm">
            The restoration logic is organized across clean, decoupled modules in the backend codebase:
          </p>
          <div className="space-y-4 text-xs font-mono text-stone-600">
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">1. API Entry Point:</strong>
              <code>backend/routers/image_router.py</code> - Receives multipart image uploads, exposes the SSE progress stream and result endpoints.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">2. Service Layer Wrapper:</strong>
              <code>backend/services/image_service.py</code> - Singleton pipeline façade: task store, background execution, status tracking.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">3. Pipeline Orchestrator:</strong>
              <code>backend/ai/pipeline/restoration_pipeline.py</code> - Drives the ordered stage registry (analysis → masking → inpainting → denoise → colorize → faces → upscale) with monotonic progress reporting.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">4. Colorization Engine:</strong>
              <code>backend/ai/colorization/ddcolor_colorizer.py</code> - DDColor wrapper (512px input) plus chrominance diffusion and artifact reduction.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">5. Restoration Engines:</strong>
              <code>backend/ai/restoration/</code> - LaMa inpainting (inpainting_engine.py), GFPGAN faces (face_restorer.py), Real-ESRGAN ×4 (super_resolution.py).
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">6. Honesty Layer:</strong>
              <code>backend/ai/report/restoration_report.py</code> - Classifies recovered / reconstructed / flagged evidence and emits the report JSON.
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Credits */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-indigo-500" /> Model Credits
        </h2>
        <div className="bg-stone-50 border border-stone-200 rounded-lg p-6">
          <p className="text-stone-700 leading-relaxed mb-4 text-sm">
            PixRevive builds on the following published research artifacts:
          </p>
          <div className="flex flex-wrap items-center justify-between gap-4 mt-2">
            <a
              href="https://arxiv.org/abs/2308.11470"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-2.5 px-4 rounded-lg shadow-sm transition-all flex items-center gap-2 hover:border-stone-300"
            >
              <Layers className="w-4 h-4 text-indigo-500" /> DDColor — ICCV 2023
            </a>
            <a
              href="https://arxiv.org/abs/2104.01555"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-2.5 px-4 rounded-lg shadow-sm transition-all flex items-center gap-2 hover:border-stone-300"
            >
              <Layers className="w-4 h-4 text-indigo-500" /> Real-ESRGAN — ICCVW 2021
            </a>
            <a
              href="https://arxiv.org/abs/2101.04027"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-2.5 px-4 rounded-lg shadow-sm transition-all flex items-center gap-2 hover:border-stone-300"
            >
              <Layers className="w-4 h-4 text-indigo-500" /> GFPGAN — CVPR 2021
            </a>
          </div>
        </div>
      </section>

      {/* Runtime Device Preference Selection */}
      <section className="mb-16 border-t border-stone-200 pt-12">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-stone-700" /> Runtime Device Management
        </h2>
        <p className="text-stone-700 leading-relaxed mb-4">
          To ensure deployment versatility, PixRevive supports dynamic device execution routing at the API route level:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mt-4">
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-4">
            <strong className="block text-stone-800 mb-1">Auto Preference</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Queries CUDA availability on launch. Routes execution to the GPU if active, falling back to CPU if unavailable.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-4">
            <strong className="block text-stone-800 mb-1">CPU Only</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Forces tensors and model weights to PyTorch CPU device. Best for environments without discrete graphics hardware.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-4">
            <strong className="block text-stone-800 mb-1">GPU Only</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Forces execution to CUDA. Accelerates inference times by parallelizing matrix convolutions across GPU kernels.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
