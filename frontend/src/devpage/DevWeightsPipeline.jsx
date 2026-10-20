import React, { useState } from 'react';
import { ChevronDown, Zap, Eye } from 'lucide-react';
import { STAGE_DETAIL } from './content';
import { Card } from './DevWeightsSections';

// Uses stages.json (frontend/src/generated/stages.json) — the backend-exported
// ordered stage list — as the canonical list, then enriches each with educational copy.

const GATE_LABEL = {
  enable_repair: 'conditional — repair level (default on)',
  enable_denoise: 'conditional — denoise level (default on)',
  enable_colorization: 'conditional — colorize on + grayscale/sepia source only',
  enable_face_restore: 'conditional — face restore level (default on)',
  enable_sr: 'conditional — super-resolution level (default on)',
  enable_reimagine: 'BETA — opt-in only',
};

const MODEL_BY_STAGE = {
  loading: '—all models',
  deskew: 'Classical Hough transform',
  flat_field: 'Background envelope',
  despeckle: 'ROAD filter + tophat',
  moire: 'HalftoneSuppressor (FFT)',
  analyze: 'QualityAnalyzer',
  face_detect: 'RetinaFace + YuNet + Haar',
  cast_fix: 'Channel stats',
  damage_detect: 'U-Net defect segmenter + classical detectors + MaskRefiner',
  inpaint: 'Big-LaMa (WACV 2022)',
  damage_cluster_smooth: 'Variance-gated smoothing',
  denoise: 'Non-Local Means (3-zone)',
  deblur: 'Restormer (CVPR 2022) — lazy',
  colorize: 'DDColor (ICCV 2023)',
  face_restore: 'GFPGAN v1.4',
  pre_sr: 'Anti-alias filter + deband',
  super_res: 'Real-ESRGAN x4plus (tiles)',
  sharpen: 'LAB unsharp + speck sweep',
  projector_grade: 'CLAHE + tone curve',
  output_verify: 'Bright-outlier detector + Telea',
  reimagine: 'SDXL/SD-1.5 diffusion (beta)',
};

function StageCard({ id, name, pct, gate }) {
  const [open, setOpen] = useState(false);
  const d = STAGE_DETAIL[id];
  const measured = d?.effect;

  return (
    <div className="cream-card p-0 overflow-hidden w-full transition-all">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full text-left p-3.5 sm:p-4 flex items-center gap-2.5 sm:gap-3 hover:bg-stone-50/80 active:bg-stone-100/60 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 min-h-[52px] cursor-pointer"
      >
        <span className="text-[10px] font-mono bg-stone-900 text-stone-100 px-2 py-1 rounded shrink-0 font-semibold">
          {pct}%
        </span>
        <span className="flex-1 min-w-0">
          <span className="block font-serif text-xs sm:text-sm text-stone-800 truncate font-semibold">{name}</span>
          <span className="block text-[10px] text-stone-500 font-mono truncate">{MODEL_BY_STAGE[id] || ''}</span>
        </span>
        {gate && (
          <span className="hidden sm:inline text-[9px] font-mono uppercase bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 rounded shrink-0">
            conditional
          </span>
        )}
        <ChevronDown
          className={`w-4 h-4 text-stone-400 transition-transform shrink-0 ${open ? 'rotate-180 text-stone-700' : ''}`}
        />
      </button>

      {open && d && (
        <div className="px-3.5 sm:px-4 pb-4 space-y-2.5 border-t border-stone-100 pt-3 text-xs leading-relaxed animate-in fade-in duration-150">
          {gate && (
            <p className="text-[10px] font-mono text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1 leading-snug">
              {GATE_LABEL[gate]}
            </p>
          )}
          <p>
            <strong className="text-stone-800 font-semibold">What:</strong> <span className="text-stone-600">{d.what}</span>
          </p>
          <p>
            <strong className="text-stone-800 font-semibold">Why it exists:</strong>{' '}
            <span className="text-stone-600">{d.why}</span>
          </p>
          <p>
            <strong className="text-stone-800 font-semibold">Model / algorithm:</strong>{' '}
            <span className="text-stone-600">{d.model}</span>
          </p>
          <p>
            <strong className="text-stone-800 font-semibold">In &rarr; out:</strong>{' '}
            <span className="text-stone-600">{d.io}</span>
          </p>
          <p>
            <strong className="text-stone-800 font-semibold">Typical failures + what was learned:</strong>{' '}
            <span className="text-stone-600">{d.failures}</span>
          </p>
          {measured && (
            <p className="text-emerald-800 bg-emerald-50 border border-emerald-200 rounded p-2 flex items-start gap-1.5 leading-snug font-medium">
              <Zap className="w-3.5 h-3.5 mt-0.5 shrink-0 text-emerald-600" />
              <span>Measured: {measured}</span>
            </p>
          )}
          <p className="text-stone-400 text-[10px] font-mono break-all pt-1">source: {d.src}</p>
        </div>
      )}

      {open && !d && (
        <div className="px-3.5 sm:px-4 pb-4 pt-3 text-xs text-stone-400 border-t border-stone-100">
          Detail copy not yet written for this stage &mdash; see <code>backend/ai/pipeline/stage_registry.py</code>.
        </div>
      )}
    </div>
  );
}

const CROP_STEPS = [
  ['01_analyze', 'after analysis'],
  ['03_inpaint', 'after inpaint (LaMa)'],
  ['05_denoise', 'after denoise'],
  ['07_face', 'after face restore'],
  ['09_sr', 'after 4× upscale'],
  ['10_sharpen', 'after sharpen'],
  ['10b_grade', 'after grade (final)'],
];

function SameCropViewer() {
  const [idx, setIdx] = useState(0);
  const [id, label] = CROP_STEPS[idx];

  return (
    <Card className="mt-6 w-full">
      <h3 className="font-serif text-base sm:text-lg text-stone-800 mb-1 flex items-center gap-2">
        <Eye className="w-4 h-4 text-indigo-500 shrink-0" />
        <span>Same crop across stages</span>
      </h3>
      <p className="text-xs text-stone-500 mb-4 leading-relaxed">
        One region of the Lena speckle test fixture, captured after each stage (PIXREVIVE_DEBUG snapshots, _qa/lena_steps/).
        Use it to see exactly where the pipeline removes speckle, fills tears, restores texture, and grades.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
        <div className="w-full rounded-lg border border-stone-200 bg-stone-50 overflow-hidden">
          <img
            src={`/devpage_assets/lena_steps/crop_${id}.jpg`}
            alt={`Pipeline output crop at step: ${label}`}
            loading="lazy"
            className="w-full h-auto object-contain"
          />
        </div>

        <div className="text-xs text-stone-600 space-y-3">
          <div className="flex items-center justify-between border-b border-stone-100 pb-2">
            <span className="font-mono text-xs uppercase tracking-wide text-indigo-700 font-bold">
              {label}
            </span>
            <span className="text-[11px] font-mono text-stone-400">
              {idx + 1} of {CROP_STEPS.length}
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5 sm:gap-2">
            {CROP_STEPS.map((c, i) => (
              <button
                key={c[0]}
                type="button"
                onClick={() => setIdx(i)}
                aria-pressed={i === idx}
                className={`text-xs font-mono px-3 py-2 rounded-lg border transition-all cursor-pointer min-h-[38px] flex items-center justify-center ${
                  i === idx
                    ? 'bg-stone-900 text-stone-100 border-stone-900 shadow-xs'
                    : 'bg-white text-stone-600 border-stone-200 hover:bg-stone-50 active:bg-stone-100'
                }`}
              >
                {c[0].replace('_', ' ')}
              </button>
            ))}
          </div>

          <p className="text-[11px] text-stone-500 leading-relaxed pt-1">
            Reading tip: compare inpaint &rarr; denoise to watch grain flatten without losing the canopy texture, then SR
            &rarr; sharpen to watch internal borders come back.
          </p>
        </div>
      </div>
    </Card>
  );
}

export default function PipelineExplorer({ stages, totalRegistered }) {
  return (
    <div className="w-full">
      <p className="text-xs text-stone-500 mb-4 font-mono leading-relaxed">
        {totalRegistered} stages registered in stage_registry.py &bull; {stages.length} user-visible in this list &bull;
        2 internal (refresh_faces, semantic_validate) &bull; conditional stages marked amber.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
        {stages.map((s) => (
          <StageCard key={s.id} id={s.id} name={s.name} pct={s.progress_pct} gate={s.route_gate} />
        ))}
      </div>
      <SameCropViewer />
    </div>
  );
}
