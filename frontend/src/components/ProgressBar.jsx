import { CheckCircle2, Loader2 } from 'lucide-react';

// Ordered to match the emission order of `_emit(...)` in
// backend/ai/pipeline/restoration_pipeline.py, with each stage's nominal
// percentage. `at` is what lets the dots stay correct when a stage is skipped:
// damage repair, colour refinement and the identity check are all conditional,
// so their events never arrive for most images.
const STAGES = [
  { id: 'despeckle',             at: 3,  label: 'Suppressing speckle' },
  { id: 'analyze',               at: 5,  label: 'Analyzing image' },
  { id: 'face_detect',           at: 12, label: 'Detecting faces' },
  { id: 'damage_detect',         at: 20, label: 'Detecting damage' },
  { id: 'refine_mask',           at: 28, label: 'Refining damage mask' },
  { id: 'inpaint',               at: 35, label: 'Inpainting damage' },
  { id: 'damage_cluster_smooth', at: 38, label: 'Smoothing damage clusters' },
  { id: 'denoise',               at: 50, label: 'Spatially-gated denoising' },
  { id: 'smart_denoise',         at: 50, label: 'Denoising' },
  { id: 'colorize',              at: 65, label: 'Colorizing (DDColor)' },
  { id: 'color_refine',          at: 72, label: 'Reducing color artifacts' },
  { id: 'face_restore',          at: 78, label: 'Restoring faces (GFPGAN)' },
  { id: 'pre_sr',                at: 85, label: 'Pre-SR conditioning' },
  { id: 'super_res',             at: 90, label: 'Upscaling (Real-ESRGAN)' },
  { id: 'upscale',               at: 90, label: 'Upscaling' },
  { id: 'harmonize',             at: 92, label: 'Harmonizing colors' },
  { id: 'sharpen',               at: 95, label: 'Crispness calibration' },
  { id: 'enhance',               at: 95, label: 'Enhancing image' },
  { id: 'projector_grade',       at: 98, label: 'Projector display grading' },
  { id: 'final_refine',          at: 98, label: 'Final color refinement' },
  { id: 'identity_check',        at: 98, label: 'Identity check' },
  { id: 'validate',              at: 99, label: 'Validating quality' },
  { id: 'encode',                at: 99, label: 'Encoding output' },
];

// Stages that exist but get no dot: `loading` happens before the pipeline
// proper, `done`/`complete` is the terminal state (the dot row is hidden at 100%).
const EDGE_LABELS = {
  loading: 'Loading AI models',
  complete: 'Complete',
  done: 'Complete',
};

const labelFor = (stage) =>
  EDGE_LABELS[stage] || STAGES.find((s) => s.id === stage)?.label || 'Processing...';

export default function ProgressBar({ progress }) {
  if (!progress) return null;

  const { stage, message } = progress;
  // A missing or non-numeric percent used to render `NaN%` and a `width: NaN%`
  // bar, which collapses to zero width.
  const pct = Math.min(100, Math.max(0, Number(progress.percent) || 0));

  // Resolve the active stage by name, and fall back to the highest stage whose
  // nominal percentage has been passed. The fallback also marks skipped stages
  // as done — with name matching alone an unrecognised stage yielded -1 and
  // every dot stayed grey.
  const byName = STAGES.findIndex((s) => s.id === stage);
  const byPercent = STAGES.reduce((best, s, i) => (pct >= s.at ? i : best), -1);
  const currentIdx = Math.max(byName, byPercent);

  return (
    <div className="w-full max-w-md mx-auto space-y-3">
      {/* Progress bar */}
      <div
        className="relative w-full h-3 bg-stone-200 rounded-full overflow-hidden"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={message || labelFor(stage)}
      >
        <div
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        >
          {/* Shimmer confined to the filled region. The old version pulsed a
              fixed w-1/3 strip clipped by inset(0 100-percent% 0 0), which
              tracked nothing. */}
          {pct < 100 && <div className="absolute inset-0 bg-white/25 rounded-full animate-pulse" />}
        </div>
      </div>

      {/* Percentage + status text */}
      <div className="flex items-center justify-between text-xs text-stone-500">
        <div className="flex items-center gap-2 min-w-0">
          {pct >= 100 ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          ) : (
            <Loader2 className="w-4 h-4 text-indigo-500 animate-spin shrink-0" />
          )}
          <span className="truncate font-medium text-stone-700">{message || labelFor(stage)}</span>
        </div>
        <span className="font-semibold tabular-nums text-stone-600 ml-2">{pct}%</span>
      </div>

      {/* Stage indicators */}
      {pct < 100 && (
        <div className="flex flex-wrap gap-1.5 mt-1">
          {STAGES.map((s, i) => (
            <div
              key={s.id}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                i < currentIdx
                  ? 'bg-emerald-400'
                  : i === currentIdx
                  ? 'bg-indigo-500 animate-pulse'
                  : 'bg-stone-200'
              }`}
              title={s.label}
            />
          ))}
        </div>
      )}
    </div>
  );
}
