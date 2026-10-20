// ResultCard — the before/after comparison panes (plan 5.6 / A-7).
// Renders the original pane plus the restored-output pane in its three
// states: result (with the Restoration Report), processing, awaiting.
// Phase H2 "Show what's real": hybrid runs carry an evidence-overlay URL
// in the report; a toggle stacks it over the restored output (identical
// canvas, pixel-aligned) to show exactly which zones are original.
import { useState } from 'react';
import { Image as ImageIcon, Eye, EyeOff } from 'lucide-react';
import ProgressBar from './ProgressBar';
import RestorationReportCard from './RestorationReportCard';
import { backendUrl } from '../services/api';

function PaneHeader({ title, size }) {
  return (
    <div className="flex items-center justify-between mb-2.5 sm:mb-3 border-b border-stone-100 pb-2">
      <h3 className="text-stone-800 font-semibold text-xs sm:text-sm uppercase tracking-wider">
        {title}
      </h3>
      {size?.width && size?.height ? (
        <span className="text-[11px] font-mono text-stone-500 bg-stone-100 px-2 py-0.5 rounded border border-stone-200/60">
          {size.width} &times; {size.height}px
        </span>
      ) : null}
    </div>
  );
}

export default function ResultCard({
  preview,
  resultImage,
  status,
  progress,
  restorationReport,
  originalSize,
  resultSize,
  setOriginalSize,
  setResultSize,
}) {
  const [showEvidence, setShowEvidence] = useState(false);

  // Only hybrid runs publish an evidence overlay (design doc §7).
  const evidenceOverlay =
    restorationReport?.reimagine?.mode === 'hybrid-restoration'
      ? restorationReport.reimagine.evidence_overlay
      : null;

  return (
    <div className="flex flex-col lg:flex-row gap-6 lg:gap-8 items-start justify-center w-full">
      {/* Original Image Pane */}
      <div className="flex-1 flex flex-col w-full min-w-0">
        <PaneHeader title="Original Source" size={originalSize} />
        <div className="relative w-full aspect-[4/3] sm:min-h-[280px] max-h-[60vh] bg-stone-50 rounded-lg border border-stone-200 flex items-center justify-center p-2 overflow-hidden shadow-xs">
          <img
            src={preview}
            alt="Original source photograph"
            title={originalSize ? `${originalSize.width} × ${originalSize.height}px` : 'Original image'}
            onLoad={(e) => {
              const img = e.currentTarget;
              setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight });
            }}
            className="max-w-full max-h-full object-contain rounded-xs select-none"
          />
        </div>
      </div>

      {/* Restored Image Pane */}
      {resultImage ? (
        <div className="flex-1 flex flex-col w-full min-w-0">
          <PaneHeader title="Restored Output" size={resultSize} />
          <div className="relative w-full aspect-[4/3] sm:min-h-[280px] max-h-[60vh] bg-stone-50 rounded-lg border border-stone-200 flex items-center justify-center p-2 shadow-xs overflow-hidden">
            <img
              src={resultImage}
              alt="AI Restored output"
              title={resultSize ? `${resultSize.width} × ${resultSize.height}px` : 'Restored image'}
              onLoad={(e) => {
                const img = e.currentTarget;
                setResultSize({ width: img.naturalWidth, height: img.naturalHeight });
              }}
              className="max-w-full max-h-full object-contain rounded-xs select-none"
            />

            {/* Evidence overlay for hybrid runs */}
            {showEvidence && evidenceOverlay && (
              <img
                src={backendUrl(evidenceOverlay)}
                alt="Evidence zones: green = original pixels, magenta = AI-re-imagined"
                className="absolute inset-2 w-[calc(100%-1rem)] h-[calc(100%-1rem)] object-contain rounded-xs pointer-events-none"
              />
            )}

            {/* Show what's real toggle */}
            {evidenceOverlay && (
              <button
                type="button"
                onClick={() => setShowEvidence((v) => !v)}
                className="absolute top-2.5 right-2.5 z-10 flex items-center gap-1.5 rounded-lg bg-stone-900/85 backdrop-blur-xs px-3 py-2 text-xs font-semibold text-stone-50 hover:bg-stone-900 active:bg-black transition-all shadow-md min-h-[44px] min-w-[44px] cursor-pointer"
                title="Green = surviving original pixels · Magenta = AI-re-imagined zones"
                aria-label={showEvidence ? 'Hide evidence zones' : "Show what's real evidence zones"}
              >
                {showEvidence ? <EyeOff className="w-4 h-4 text-amber-300" /> : <Eye className="w-4 h-4 text-emerald-300" />}
                <span>{showEvidence ? 'Hide zones' : "Show what's real"}</span>
              </button>
            )}
          </div>

          {/* Restoration Report */}
          <RestorationReportCard report={restorationReport} />
        </div>
      ) : status === 'processing' ? (
        <div className="flex-1 flex flex-col w-full min-w-0">
          <PaneHeader title="Restoring..." />
          <div className="relative w-full aspect-[4/3] sm:min-h-[280px] max-h-[60vh] bg-stone-50 rounded-lg border border-stone-200 flex flex-col items-center justify-center p-4 sm:p-6 shadow-xs">
            <ProgressBar progress={progress} />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col w-full min-w-0 opacity-40">
          <PaneHeader title="Restored Output" />
          <div className="relative w-full aspect-[4/3] sm:min-h-[280px] max-h-[60vh] bg-stone-50 rounded-lg border border-stone-200 flex flex-col items-center justify-center p-4 text-center">
            <ImageIcon className="w-10 h-10 sm:w-12 sm:h-12 text-stone-300 mb-2 stroke-[1.25]" />
            <span className="text-stone-500 text-xs sm:text-sm font-medium">Awaiting processing</span>
          </div>
        </div>
      )}
    </div>
  );
}
