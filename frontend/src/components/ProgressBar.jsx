import { CheckCircle2, Loader2 } from 'lucide-react';
import registry from '../generated/stages.json';

// Single source of truth: backend/ai/pipeline/stage_registry.py, exported via
// scripts/export_stage_list.py. `at` is each stage's nominal percentage — it
// keeps the dots correct when conditional stages (repair, denoise,
// colorization...) are skipped and their events never arrive.
const STAGES = registry.stages.map((s) => ({
  id: s.id,
  at: s.progress_pct,
  label: s.name,
}));

// Edge events exist but get no dot: `loading` happens before the pipeline
// proper, `done`/`complete` are terminal (the dot row is hidden at 100%).
const EDGE_LABELS = Object.fromEntries(
  registry.edge_events.map((e) => [e.id, e.name]),
);

const labelFor = (stage) =>
  EDGE_LABELS[stage] || STAGES.find((s) => s.id === stage)?.label || 'Processing...';

export default function ProgressBar({ progress }) {
  if (!progress) return null;

  const { stage, message } = progress;
  const pct = Math.min(100, Math.max(0, Number(progress.percent) || 0));

  const byName = STAGES.findIndex((s) => s.id === stage);
  const byPercent = STAGES.reduce((best, s, i) => (pct >= s.at ? i : best), -1);
  const currentIdx = Math.max(byName, byPercent);

  return (
    <div className="w-full max-w-md mx-auto space-y-3 px-1">
      {/* Progress bar line */}
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
          {pct < 100 && <div className="absolute inset-0 bg-white/25 rounded-full animate-pulse" />}
        </div>
      </div>

      {/* Percentage + status text row */}
      <div className="flex items-center justify-between text-xs text-stone-500 gap-2 min-w-0">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {pct >= 100 ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          ) : (
            <Loader2 className="w-4 h-4 text-indigo-500 animate-spin shrink-0" />
          )}
          <span className="truncate font-medium text-stone-700 text-xs sm:text-sm">
            {message || labelFor(stage)}
          </span>
        </div>
        <span className="font-semibold tabular-nums text-stone-700 shrink-0 text-xs sm:text-sm ml-1">
          {pct}%
        </span>
      </div>

      {/* Responsive stage indicators */}
      {pct < 100 && (
        <div className="pt-1">
          <div className="flex items-center justify-between gap-1 w-full">
            {STAGES.map((s, i) => (
              <div
                key={s.id}
                className={`flex-1 h-1.5 rounded-full transition-all duration-300 ${
                  i < currentIdx
                    ? 'bg-emerald-500'
                    : i === currentIdx
                    ? 'bg-indigo-600 animate-pulse'
                    : 'bg-stone-200'
                }`}
                title={`${s.label} (${s.at}%)`}
              />
            ))}
          </div>
          <p className="text-[10px] font-mono text-stone-400 mt-1 text-right">
            Stage {Math.max(1, currentIdx + 1)} of {STAGES.length}
          </p>
        </div>
      )}
    </div>
  );
}
