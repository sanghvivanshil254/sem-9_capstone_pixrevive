import { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, CheckCircle2, Sparkles, AlertTriangle } from 'lucide-react';

/**
 * RestorationReportCard — the honesty layer UI (T-2, plan 5.1).
 *
 * Renders the machine-readable report the backend attaches to the
 * process-stream `complete` event (and /api/process/{id}/status):
 *   recovered     — damage removed using surviving pixels
 *   reconstructed — content invented plausibly by generative models
 *   flagged       — conditions a reviewer should know about
 *
 * Colorized results always carry the "plausible colors, not original
 * colors" disclaimer, per the plan's honesty requirement.
 */
export default function RestorationReportCard({ report }) {
  const [open, setOpen] = useState(false);

  if (!report || typeof report !== 'object') return null;

  const recovered = report.recovered || [];
  const reconstructed = report.reconstructed || [];
  const flagged = report.flagged || [];
  const hasContent = recovered.length + reconstructed.length + flagged.length > 0;
  if (!hasContent) return null;

  return (
    <div className="mt-4 sm:mt-6 border border-stone-200 rounded-xl bg-stone-50/70 overflow-hidden w-full transition-all">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3.5 sm:px-4 py-3 text-left hover:bg-stone-100/70 active:bg-stone-100 transition-colors cursor-pointer min-h-[44px]"
        aria-expanded={open}
        aria-label="Toggle Restoration Honesty Report"
      >
        <span className="flex items-center gap-2 text-xs sm:text-sm font-semibold text-stone-800 flex-wrap">
          <FileText className="w-4 h-4 text-stone-500 shrink-0" />
          <span>Restoration Report</span>
          {flagged.length > 0 && (
            <span className="inline-flex items-center rounded-full bg-amber-100 text-amber-800 text-[11px] font-medium px-2 py-0.5 border border-amber-200/50">
              {flagged.length} note{flagged.length > 1 ? 's' : ''}
            </span>
          )}
        </span>
        {open ? (
          <ChevronUp className="w-4 h-4 text-stone-500 shrink-0 ml-2" />
        ) : (
          <ChevronDown className="w-4 h-4 text-stone-500 shrink-0 ml-2" />
        )}
      </button>

      {open && (
        <div className="px-3.5 sm:px-4 pb-4 pt-1 space-y-4 text-xs sm:text-sm animate-in fade-in duration-200 border-t border-stone-200/50">
          {report.colorization_disclaimer && (
            <p className="text-xs sm:text-[13px] text-stone-600 italic border-l-2 border-indigo-400 pl-3 leading-relaxed">
              This photo was colorized by AI. Colors are plausible estimates, not the original colors.
            </p>
          )}

          {/* Phase H1/H2 — re-imagine evidence disclosure */}
          {report.reimagine?.mode === 'hybrid-restoration' && (
            <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-3 text-xs sm:text-[13px]">
              <p className="font-semibold text-amber-800 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 shrink-0 text-amber-600" />
                Hybrid restoration
              </p>
              <p className="text-stone-700 mt-1.5 leading-relaxed">
                <strong>{Number(report.reimagine.real_pixels_pct ?? 0).toFixed(0)}% original evidence</strong>{' '}
                preserved pixel-for-pixel &bull;{' '}
                <strong>{Number(report.reimagine.generated_pixels_pct ?? 0).toFixed(0)}% re-imagined by AI</strong>
                {report.reimagine.identity?.similarity != null &&
                  ` (identity similarity ${Number(report.reimagine.identity.similarity).toFixed(2)})`}
                . The re-imagined parts are plausible, not original.
              </p>
              {report.reimagine.resource?.tier && (
                <p className="text-[11px] text-stone-500 mt-1 font-mono break-all">
                  generator: {report.reimagine.generator || '—'} &bull; tier:{' '}
                  {report.reimagine.resource.tier} &bull; seed: {report.reimagine.seed ?? '—'}
                </p>
              )}
              {report.reimagine.identity_conditioning &&
                report.reimagine.identity_conditioning !== 'none' && (
                  <p className="text-[11px] text-stone-500 mt-0.5">
                    identity signal:{' '}
                    {report.reimagine.identity_conditioning === 'faceid'
                      ? 'FaceID (ArcFace embedding of the surviving face)'
                      : 'CLIP image adapter (damaged-crop reference)'}
                  </p>
                )}
            </div>
          )}

          {report.reimagine?.mode === 'not_applied' &&
            report.reimagine.reason &&
            report.reimagine.status === 'skipped_degraded' && (
              <p className="text-xs sm:text-[13px] text-stone-600 italic border-l-2 border-rose-300 pl-3 leading-relaxed">
                Re-imagine was attempted but rolled back (the AI-generated face did not match the original closely
                enough) — this output is the standard restoration.
              </p>
            )}

          {report.reimagine?.mode === 'not_applied' &&
            report.reimagine.status === 'skipped' &&
            report.reimagine.reason && (
              <p className="text-xs text-stone-500 italic border-l-2 border-stone-300 pl-3 leading-relaxed">
                Re-imagine (beta) did not run: {report.reimagine.reason}
              </p>
            )}

          {recovered.length > 0 && (
            <div>
              <p className="flex items-center gap-1.5 font-semibold text-emerald-700 mb-1.5 text-xs sm:text-sm">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                Repaired
              </p>
              <ul className="space-y-1 text-stone-600 pl-1">
                {recovered.map((item, i) => (
                  <li key={i} className="leading-snug text-xs sm:text-sm flex items-start gap-1.5">
                    <span className="text-stone-400 select-none">&bull;</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {reconstructed.length > 0 && (
            <div>
              <p className="flex items-center gap-1.5 font-semibold text-indigo-700 mb-1.5 text-xs sm:text-sm">
                <Sparkles className="w-4 h-4 shrink-0 text-indigo-500" />
                Reconstructed by AI (plausible, not original)
              </p>
              <ul className="space-y-1 text-stone-600 pl-1">
                {reconstructed.map((item, i) => (
                  <li key={i} className="leading-snug text-xs sm:text-sm flex items-start gap-1.5">
                    <span className="text-stone-400 select-none">&bull;</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {flagged.length > 0 && (
            <div>
              <p className="flex items-center gap-1.5 font-semibold text-amber-700 mb-1.5 text-xs sm:text-sm">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                Notes
              </p>
              <ul className="space-y-1 text-stone-600 pl-1">
                {flagged.map((item, i) => (
                  <li key={i} className="leading-snug text-xs sm:text-sm flex items-start gap-1.5">
                    <span className="text-stone-400 select-none">&bull;</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
