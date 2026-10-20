import React, { useState } from 'react';
import { ArrowLeft, Workflow, Cpu, Layers, Gauge, BookOpen, CircleHelp, TerminalSquare } from 'lucide-react';
import { Link } from 'react-router-dom';
import registry from '../generated/stages.json';
import WeightsAdmin from './WeightsAdmin';
import { Section, Card, BeforeAfter, DepthToggle, DepthNote, HealthBadge, TocRail } from '../devpage/DevWeightsSections';
import PipelineExplorer from '../devpage/DevWeightsPipeline';
import {
  REGISTRY_TOTAL,
  MODELS,
  METRICS,
  DIFFICULT_SOLVED,
  DIFFICULT_PARTLY,
  DIFFICULT_FUNDAMENTAL,
  GLOSSARY,
  FEATURE_FLAGS,
  QUICKSTART,
  FAQ,
  FURTHER_READING,
} from '../devpage/content';

const TOC = [
  { id: 'what', num: '§1', label: 'What PixRevive does' },
  { id: 'how', num: '§2', label: 'How it works' },
  { id: 'models', num: '§3', label: 'Models & weights' },
  { id: 'stack', num: '§4', label: 'Architecture & stack' },
  { id: 'hard', num: '§5', label: 'Difficulties & lessons' },
  { id: 'metrics', num: '§6', label: 'How quality is measured' },
  { id: 'edu', num: '§7', label: 'Learn the concepts' },
  { id: 'extr', num: '§8', label: 'Run it yourself' },
  { id: 'admin', num: '§9', label: 'Admin tools' },
];

function ArchitectureSvg() {
  return (
    <div className="w-full bg-white border border-stone-200 rounded-xl p-3 sm:p-5 shadow-xs overflow-hidden">
      <div className="w-full overflow-x-auto touch-scroll-x overscroll-x-contain pb-2">
        <div className="min-w-[580px] sm:min-w-[680px] lg:min-w-0">
          <svg viewBox="0 0 760 300" className="w-full h-auto" role="img" aria-label="Architecture diagram: browser to FastAPI to pipeline to models">
            <defs>
              <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#78716c"/></marker>
            </defs>
            <g fontFamily="monospace" fontSize="11" fill="#44403c">
              <rect x="12" y="40" width="150" height="66" rx="8" fill="#faf9f6" stroke="#a8a29e"/>
              <text x="26" y="62">React 19 (Vite)</text>
              <text x="26" y="78" fontSize="9" fill="#78716c">upload card · progress ·</text>
              <text x="26" y="92" fontSize="9" fill="#78716c">report · :5173</text>

              <rect x="240" y="40" width="210" height="66" rx="8" fill="#faf9f6" stroke="#a8a29e"/>
              <text x="254" y="62">FastAPI :8000</text>
              <text x="254" y="78" fontSize="9" fill="#78716c">image_router · admin_router</text>
              <text x="254" y="92" fontSize="9" fill="#78716c">ImageService (task store, Semaphore(1))</text>

              <rect x="528" y="24" width="210" height="50" rx="8" fill="#eef2ff" stroke="#818cf8"/>
              <text x="542" y="46">RestorationPipeline</text>
              <text x="542" y="62" fontSize="9" fill="#6366f1">stage_registry.py · 22 stages</text>

              <rect x="528" y="96" width="210" height="50" rx="8" fill="#fef3c7" stroke="#f59e0b"/>
              <text x="542" y="118">Models (ai/weights/)</text>
              <text x="542" y="134" fontSize="9" fill="#b45309">LaMa · DDColor · GFPGAN · ESRGAN…</text>

              <rect x="240" y="170" width="498" height="56" rx="8" fill="#faf9f6" stroke="#a8a29e"/>
              <text x="254" y="192">SSE stream /api/process-stream  →  progress events (stage, percent, message)</text>
              <text x="254" y="209" fontSize="9" fill="#78716c">POST multipart image → task id → live stage events → GET /api/process/{'{id}'}/result (PNG)</text>

              <rect x="12" y="170" width="150" height="56" rx="8" fill="#faf9f6" stroke="#a8a29e"/>
              <text x="26" y="192">User's browser</text>
              <text x="26" y="209" fontSize="9" fill="#78716c">progress bar drains</text>
              <text x="26" y="222" fontSize="9" fill="#78716c">towards 100%</text>

              <line x1="162" y1="73" x2="238" y2="73" stroke="#78716c" markerEnd="url(#arr)"/>
              <text x="172" y="66" fontSize="9" fill="#78716c">POST image</text>
              <line x1="450" y1="73" x2="526" y2="49" stroke="#78716c" markerEnd="url(#arr)"/>
              <line x1="526" y1="98" x2="450" y2="90" stroke="#78716c" markerEnd="url(#arr)"/>
              <text x="462" y="72" fontSize="9" fill="#78716c">restore()</text>
              <line x1="633" y1="74" x2="633" y2="94" stroke="#a8a29e" markerEnd="url(#arr)"/>
              <line x1="489" y1="106" x2="489" y2="168" stroke="#78716c" markerEnd="url(#arr)"/>
              <line x1="238" y1="198" x2="164" y2="198" stroke="#78716c" markerEnd="url(#arr)"/>
              <text x="170" y="192" fontSize="9" fill="#78716c">SSE events</text>
            </g>
          </svg>
        </div>
      </div>
      <p className="text-[11px] text-stone-400 font-mono mt-1.5 text-center sm:hidden">
        Scroll horizontally &harr; to explore architecture map
      </p>
    </div>
  );
}

export default function DevWeights() {
  const [depth, setDepth] = useState('beginner');
  const tech = depth === 'technical';

  return (
    <div className="w-full pt-2 sm:pt-4">
      <div className="w-full max-w-7xl mx-auto px-3.5 sm:px-6 md:px-10 lg:px-12 py-4 sm:py-8">
        <Link
          to="/"
          className="flex items-center gap-2 text-stone-600 hover:text-stone-900 active:text-black transition-colors mb-4 sm:mb-6 font-medium text-xs sm:text-sm w-fit min-h-[44px] px-1"
        >
          <ArrowLeft className="w-4 h-4 shrink-0" /> Back to Application
        </Link>

        {/* HERO */}
        <div className="mb-10 sm:mb-16 grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-10 items-start">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full dev-badge text-[11px] sm:text-xs font-mono mb-3 sm:mb-4">
              <TerminalSquare className="w-3.5 h-3.5 shrink-0" /> DEVELOPER GUIDE
            </div>
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-serif text-stone-900 mb-4 tracking-tight leading-tight">
              How PixRevive restores photographs
            </h1>
            <p className="text-stone-600 leading-relaxed text-sm sm:text-base md:text-lg mb-4 sm:mb-5">
              PixRevive is a local-only photo-restoration engine: a 22-stage pipeline that straightens,
              cleans, repairs, colours, de-blurs and upscales damaged scans &mdash; then tells you honestly
              what it did and where it is guessing.
            </p>
            <div className="mb-4">
              <HealthBadge />
            </div>
            <div className="mt-4 sm:mt-6">
              <DepthToggle depth={depth} setDepth={setDepth} />
            </div>
            <DepthNote show={tech}>
              Every fact on this page is sourced from the repo: stage_registry.py, restoration_pipeline.py,
              quality_audit.py, docs/baselines/pre_fix.json and _qa/ measurement artifacts. No marketing claims.
            </DepthNote>
          </div>
          <div>
            <BeforeAfter
              before="/devpage_assets/lena_input.jpg"
              after="/devpage_assets/lena_final.jpg"
              beforeLabel="degraded input"
              afterLabel="restored"
              alt="Lena speckle test fixture: degraded input versus restored output"
              caption="Lena speckle test figure: input (left) → after the full pipeline (right). Drag the handle. Asset from _qa/lena_input_small.png / _qa/lena_final_small.jpg."
            />
            <DepthNote show={tech}>
              Restored laplacian variance 1008 (final) vs 6431 (raw speckled input) — sharpness loss is mostly
              source resolution + denoising, SR itself measured at parity (480 vs 481) against its input. See §5/§6.
            </DepthNote>
          </div>
        </div>

        {/* Content Layout with Responsive TocRail */}
        <div className="flex flex-col xl:flex-row gap-8 xl:gap-12 items-start min-w-0 w-full">
          <TocRail sections={TOC} />
          
          <main className="flex-1 min-w-0 w-full max-w-4xl mx-auto xl:mx-0 overflow-x-hidden">

            <Section id="what" num="§1" title="What PixRevive does" kicker="In plain language — no ML background needed.">
              <Card>
                <p className="text-xs sm:text-sm text-stone-700 leading-relaxed mb-3">
                  Old photographs are damaged in specific ways: they are skewed, they carry dust specks,
                  they have cracks and torn areas, they are gray or sepia, faces are soft, and the whole
                  scan is low-resolution. PixRevive attacks those problems <strong>in a fixed order</strong> &mdash;
                  the order is the point. Specks must be removed before the damage detector runs, damage
                  must be repaired before colorization, and faces must be protected before anything
                  generative touches them.
                </p>
                <p className="text-xs sm:text-sm text-stone-600 leading-relaxed">
                  It runs entirely on your machine: React front-end on :5173, FastAPI on :8000, all model
                  weights in <code className="text-xs bg-stone-100 px-1 py-0.5 rounded">backend/ai/weights/</code> (~14 GB on disk in this checkout).
                  Nothing is uploaded anywhere. The pipeline streams live progress over SSE, and finishes
                  with a machine-readable "<em>Restoration Report</em>" that classifies what happened to
                  your photo: <strong>repaired</strong>, <strong>reconstructed</strong> (plausible, not guaranteed) and <strong>flagged</strong>.
                </p>
              </Card>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                <BeforeAfter
                  before="/devpage_assets/teared_off.jpg"
                  after="/devpage_assets/teared_on.jpg"
                  beforeLabel="teared input"
                  afterLabel="restored"
                  alt="Torn sepia portrait before and after restoration"
                  caption="teared.jpg fixture. Asset from _qa/p2_teared_off_small.jpg / _qa/p2_teared_on_small.jpg."
                />
                <BeforeAfter
                  before="/devpage_assets/case4_input.jpg"
                  after="/devpage_assets/case4_final.jpg"
                  beforeLabel="torn, pre-refine"
                  afterLabel="repaired"
                  alt="Torn damage case before and after inpainting"
                  caption="case4: torn damage before/after repair — honest failure modes exist too (see §5). Assets from _qa/case4_*."
                />
              </div>
            </Section>

            <Section id="how" num="§2" title="How it works — the pipeline" kicker="One card per real stage, in execution order. Expand any card.">
              <PipelineExplorer stages={registry.stages} totalRegistered={REGISTRY_TOTAL} />
            </Section>

            <Section id="models" num="§3" title="Models & weights" kicker="What each network is for, what it weighs, and when it loads. Live backend status in the badge above.">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {MODELS.map((m) => (
                  <Card key={m.name}>
                    <h3 className="font-serif text-sm sm:text-base text-stone-800 mb-2 font-semibold">{m.name}</h3>
                    <dl className="text-xs space-y-1.5">
                      <div className="flex justify-between border-b border-stone-100 pb-1">
                        <dt className="text-stone-500">on disk</dt>
                        <dd className="font-mono text-stone-800">{m.size_mb < 1000 ? `${m.size_mb} MB` : `${(m.size_mb / 1024).toFixed(1)} GB`}</dd>
                      </div>
                      <div className="flex justify-between border-b border-stone-100 pb-1">
                        <dt className="text-stone-500">load strategy</dt>
                        <dd className="text-stone-800">{m.load}</dd>
                      </div>
                      <div className="pt-1 text-stone-600 leading-relaxed flex">
                        <dt className="shrink-0 text-stone-500 mr-1">task</dt>
                        <dd>{m.task}</dd>
                      </div>
                      <div className="flex">
                        <dt className="shrink-0 text-stone-500 mr-1">paper</dt>
                        <dd className="text-stone-500 italic">{m.paper}</dd>
                      </div>
                      {m.measured && (
                        <div className="flex">
                          <dt className="shrink-0 text-stone-500 mr-1">measured</dt>
                          <dd className="text-emerald-700 font-medium">{m.measured}</dd>
                        </div>
                      )}
                      <div className="text-[10px] text-stone-400 font-mono pt-1 break-all">{m.file}</div>
                    </dl>
                  </Card>
                ))}
              </div>
              <DepthNote show={tech}>
                Weights dir total: 14 GB (du, this checkout). Largest optional slices: sdxl_inpaint 6.5 GB,
                sd15_inpaint 2.0 GB. GPU budget measured: GTX 1650 4 GB peaked at 3405/4096 MB during a full
                lena restore (vram_lena.csv, 1 Hz poll), GPU util 99%. Model loading: defect_segmenter and
                face_detector build synchronously; fast-loader thread loads lama + gfpgan + realesrgan;
                ddcolor loads in its own thread; Restormer deblur loads lazily on the first blur-gate fire.
              </DepthNote>
            </Section>

            <Section id="stack" num="§4" title="Architecture & stack" kicker="How the pieces talk — from upload pixel to finished PNG.">
              <ArchitectureSvg />
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-6">
                <Card>
                  <h3 className="font-serif text-sm sm:text-base text-stone-800 mb-2 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-500 shrink-0" /> Frontend
                  </h3>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    React 19 + Vite 8, Tailwind v4, react-router 7, lucide-react icons, axios. Dev server on
                    :5173; production build via <code>npm run build</code>. Talks to the backend at
                    <code> VITE_API_BASE</code>.
                  </p>
                </Card>
                <Card>
                  <h3 className="font-serif text-sm sm:text-base text-stone-800 mb-2 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-indigo-500 shrink-0" /> Backend
                  </h3>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    FastAPI + uvicorn on :8000 (Python 3.12, torch 2.13/cu126). One REST + SSE surface:
                    POST /api/process-stream &rarr; SSE progress &rarr; result endpoint returns PNG.
                  </p>
                </Card>
                <Card className="sm:col-span-2 md:col-span-1">
                  <h3 className="font-serif text-sm sm:text-base text-stone-800 mb-2 flex items-center gap-2">
                    <Workflow className="w-4 h-4 text-indigo-500 shrink-0" /> Runtime &amp; GPU
                  </h3>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Device auto-selects CUDA &rarr; CPU. <strong>4 GB VRAM is a real constraint</strong>: tests are
                    pinned to CPU, inference is serialized by a per-process Semaphore(1).
                  </p>
                </Card>
              </div>
              <DepthNote show={tech}>
                Streaming detail: SSE events carry {`{stage, percent, message}`} as they fire from stage_registry.
                Two background loader threads load models asynchronously in initialise_models() so the first
                upload does not wait for the ~2.5 GB of weights. Event order is guaranteed monotonic in progress_pct.
              </DepthNote>
            </Section>

            <Section id="hard" num="§5" title="Difficulties we hit — and what they teach" kicker="Honest, source-cited. Green = solved/solvable. Amber = trade-off. Red = fundamentally limited.">
              <div className="mb-6 sm:mb-8 border-l-4 border-l-emerald-500 pl-3 sm:pl-4">
                <h3 className="font-serif text-base sm:text-lg text-emerald-800 mb-3 font-semibold">SOLVED / SOLVABLE</h3>
                <div className="space-y-3">
                  {DIFFICULT_SOLVED.map((d) => (
                    <Card key={d.title}>
                      <h4 className="font-serif text-xs sm:text-sm text-stone-800 mb-1.5 font-semibold">{d.title}</h4>
                      <p className="text-xs text-stone-600 leading-relaxed mb-1"><strong>Symptom:</strong> {d.symptom || d.cause}</p>
                      {d.cause && d.symptom && <p className="text-xs text-stone-600 leading-relaxed mb-1"><strong>Root cause:</strong> {d.cause}</p>}
                      {d.fix && <p className="text-xs text-emerald-700 leading-relaxed mb-1 font-medium"><strong>Fix:</strong> {d.fix}</p>}
                      <p className="text-[10px] text-stone-400 font-mono break-all">{d.evidence}</p>
                    </Card>
                  ))}
                </div>
              </div>

              <div className="mb-6 sm:mb-8 border-l-4 border-l-amber-500 pl-3 sm:pl-4">
                <h3 className="font-serif text-base sm:text-lg text-amber-700 mb-3 font-semibold">PARTLY SOLVABLE — trade-offs</h3>
                <div className="space-y-3">
                  {DIFFICULT_PARTLY.map((d) => (
                    <Card key={d.title}>
                      <h4 className="font-serif text-xs sm:text-sm text-stone-800 mb-1.5 font-semibold">{d.title}</h4>
                      {d.cause && <p className="text-xs text-stone-600 leading-relaxed mb-1">{d.cause}</p>}
                      {d.verdict && <p className="text-xs text-amber-700 leading-relaxed mb-1 font-medium"><strong>Standing answer:</strong> {d.verdict}</p>}
                      <p className="text-[10px] text-stone-400 font-mono break-all">{d.evidence}</p>
                    </Card>
                  ))}
                </div>
              </div>

              <div className="mb-6 sm:mb-8 border-l-4 border-l-red-500 pl-3 sm:pl-4">
                <h3 className="font-serif text-base sm:text-lg text-red-800 mb-3 font-semibold">FUNDAMENTALLY LIMITED — and why</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {DIFFICULT_FUNDAMENTAL.map((d) => (
                    <Card key={d.title} className="bg-red-50/30">
                      <h4 className="font-serif text-xs sm:text-sm text-stone-800 mb-1.5 font-semibold">{d.title}</h4>
                      <p className="text-xs text-stone-600 leading-relaxed">{d.body}</p>
                    </Card>
                  ))}
                </div>
              </div>
            </Section>

            <Section id="metrics" num="§6" title="How quality is measured" kicker="Reference-free metrics — because no 'ground truth' photo exists for an old damaged scan.">
              <div className="space-y-3">
                {METRICS.map((m) => (
                  <Card key={m.id}>
                    <h4 className="font-mono text-xs text-indigo-600 font-semibold mb-1">{m.id}</h4>
                    <p className="text-xs text-stone-700 leading-relaxed mb-2">{m.plain}</p>
                    <div className="text-xs grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <p className="text-emerald-700 font-medium">&check; good: {m.good}</p>
                      <p className="text-stone-600">&cross; bad: {m.bad}</p>
                      <p className="text-amber-700">&Delta; can't see: {m.blindspot}</p>
                    </div>
                    <p className="text-[10px] text-stone-400 font-mono mt-1.5 break-all">
                      baseline (3-run median): {m.baseline}
                    </p>
                  </Card>
                ))}
              </div>
              <DepthNote show={tech}>
                Why 3-run medians: GPU run-to-run variance is real and measured — speck_count on teared went
                224 &rarr; 411 minute-to-minute on identical code. Comparing single runs chases noise; medians
                stabilise.
              </DepthNote>
            </Section>

            <Section id="edu" num="§7" title="Learn the concepts" kicker="Short, honest explainers. Switch Beginner/Technical at the top for depth.">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Restoration vs generation</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Restoring = inferring what a damaged photo probably looked like, staying as close to surviving pixels as possible. Generating = inventing new content freely. PixRevive is mostly the first.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">How inpainting works (LaMa)</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    LaMa receives the image and a mask of "missing" pixels, then synthesises content into just those holes by learning surrounding structure. Everything outside mask is bitwise untouched.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Diffusion vs GAN vs transformer</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Three families of generative models appear here: GANs (GFPGAN), transformers (Restormer), and diffusion (beta re-imagine). DDColor is a CNN-based dual-attention colorization head.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Super-resolution &amp; upscaling</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Real-ESRGAN enlarges photos 4&times; by hallucinating plausible high-frequency detail. It also amplifies any surviving artifacts, which is why pre_sr anti-aliasing runs right before it.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Colorization (DDColor)</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Colorization predicts CIELAB a/b (chroma) channels given L (lightness). PixRevive keeps the L channel untouched so colorization can never flatten photographic detail.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Denoising: non-local means</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Instead of blurring a pixel with neighbors, NLM averages pixels whose surrounding patches look similar anywhere in the image, preserving repeated textures like fabric.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Face restoration (GFPGAN)</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    A generative prior trained on many faces re-synthesises eyes and mouths around detected landmarks. PixRevive runs it at fidelity 0.85 and keeps small faces for a post-upscale pass.
                  </p>
                </Card>
                <Card>
                  <h4 className="font-serif text-xs sm:text-sm mb-1.5 text-stone-800 font-semibold">Ethics of restoring the past</h4>
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Color and identity are contested. PixRevive's answer is transparency: AI color is always disclosed as plausible, generative stages are opt-in, and every run produces a machine-readable report.
                  </p>
                </Card>
              </div>

              <Card className="mt-6">
                <h3 className="font-serif text-base sm:text-lg mb-3 text-stone-800 font-semibold">Glossary</h3>
                <dl className="space-y-2.5">
                  {GLOSSARY.map(([term, def]) => (
                    <div key={term} className="flex flex-col sm:flex-row sm:gap-3 text-xs">
                      <dt className="sm:shrink-0 sm:w-52 font-semibold text-stone-800 mb-0.5 sm:mb-0">{term}</dt>
                      <dd className="text-stone-600 leading-relaxed">{def}</dd>
                    </div>
                  ))}
                </dl>
              </Card>
            </Section>

            <Section id="extr" num="§8" title="Run it yourself" kicker="Verified commands from PIXREVIVE_START_GUIDE.txt. Feature flags and FAQ from the code.">
              <Card>
                <h3 className="font-serif text-sm sm:text-base mb-3 text-stone-800 flex items-center gap-2 font-semibold">
                  <TerminalSquare className="w-4 h-4 shrink-0" /> Quick start
                </h3>
                <div className="bg-stone-900 text-stone-300 font-mono text-[11px] sm:text-xs p-4 rounded-lg overflow-x-auto leading-relaxed whitespace-pre-wrap break-all touch-scroll-x">
                  {QUICKSTART.map(([c1, c2]) => (
                    <div key={c1} className="break-all">
                      <span className={c1.startsWith('#') ? 'text-stone-500' : 'text-stone-300'}>
                        {c1.startsWith('#') ? c1 : c2}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-stone-500 mt-2 leading-relaxed">
                  Windows paths from the guide; the backend needs Python 3.12 exactly (never a venv) and start_frontend.bat / start_backend.bat wrap both halves.
                </p>
              </Card>

              <Card className="mt-4">
                <h3 className="font-serif text-sm sm:text-base mb-3 text-stone-800 flex items-center gap-2 font-semibold">
                  <Gauge className="w-4 h-4 text-indigo-500 shrink-0" /> Feature flags (grep-verified in backend/ai)
                </h3>
                <dl className="space-y-2 text-xs">
                  {FEATURE_FLAGS.map(([f, d]) => (
                    <div key={f} className="flex flex-col sm:flex-row sm:gap-3 pb-1 border-b border-stone-100 last:border-0">
                      <dt className="sm:shrink-0 sm:w-64 font-mono text-[11px] text-indigo-600 font-semibold break-all">{f}</dt>
                      <dd className="text-stone-600 leading-relaxed">{d}</dd>
                    </div>
                  ))}
                </dl>
              </Card>

              <Card className="mt-4">
                <h3 className="font-serif text-sm sm:text-base mb-3 text-stone-800 flex items-center gap-2 font-semibold">
                  <CircleHelp className="w-4 h-4 text-amber-500 shrink-0" /> FAQ
                </h3>
                <div className="space-y-3">
                  {FAQ.map(([q, a]) => (
                    <details key={q} className="group border border-stone-200 rounded-lg">
                      <summary className="cursor-pointer px-3.5 sm:px-4 py-2.5 sm:py-3 text-xs sm:text-sm font-semibold text-stone-800 hover:bg-stone-50 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 select-none min-h-[44px] flex items-center">
                        {q}
                      </summary>
                      <p className="px-3.5 sm:px-4 pb-3 pt-1 text-xs text-stone-600 leading-relaxed">{a}</p>
                    </details>
                  ))}
                </div>
              </Card>

              <Card className="mt-4">
                <h3 className="font-serif text-sm sm:text-base mb-2 text-stone-800 flex items-center gap-2 font-semibold">
                  <BookOpen className="w-4 h-4 text-indigo-500 shrink-0" /> Further reading
                </h3>
                <ul className="text-xs text-stone-600 space-y-1.5 list-disc pl-5">
                  {FURTHER_READING.map((t) => <li key={t} className="leading-snug">{t}</li>)}
                </ul>
              </Card>
            </Section>

            {/* §9 Admin Tools: WeightsAdmin is now natively responsive down to 320px with NO artificial min-width */}
            <Section
              id="admin"
              num="§9"
              title="Admin tools — weights & dataset management"
              kicker="Hot-swap RealESRGAN weights, manage training pairs, and run the benchmark pipeline directly from this responsive workspace."
            >
              <div className="w-full">
                <WeightsAdmin />
              </div>
            </Section>

          </main>
        </div>
      </div>
    </div>
  );
}
