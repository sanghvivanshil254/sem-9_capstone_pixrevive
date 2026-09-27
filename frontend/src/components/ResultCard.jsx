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

function PaneHeader({ children }) {
  return (
    <div className="flex items-center justify-between mb-3 border-b border-stone-100 pb-2">
      <h3 className="text-stone-800 font-semibold text-sm uppercase tracking-wider">{children}</h3>
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
    <div className="flex flex-col md:flex-row gap-8 items-start justify-center">
      {/* Original Image */}
      <div className="flex-1 flex flex-col w-full">
        <PaneHeader>Original Source</PaneHeader>
        <div className="relative w-full aspect-[4/3] bg-stone-50 rounded border border-stone-200 flex items-center justify-center p-2">
          <img
            src={preview}
            alt="Original"
            title={originalSize ? `${originalSize.width} × ${originalSize.height}px` : 'Original image size'}
            onLoad={(e) => {
              const img = e.currentTarget;
              setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight });
            }}
            className="max-w-full max-h-full object-contain rounded-sm"
          />
        </div>
      </div>

      {/* Restored Image */}
      {resultImage ? (
        <div className="flex-1 flex flex-col w-full">
          <PaneHeader>Restored Output</PaneHeader>
          <div className="relative w-full aspect-[4/3] bg-stone-50 rounded border border-stone-200 flex items-center justify-center p-2 shadow-sm">
            <img
              src={resultImage}
              alt="Restored"
              title={resultSize ? `${resultSize.width} × ${resultSize.height}px` : 'Restored image size'}
              onLoad={(e) => {
                const img = e.currentTarget;
                setResultSize({ width: img.naturalWidth, height: img.naturalHeight });
              }}
              className="max-w-full max-h-full object-contain rounded-sm"
            />
            {showEvidence && evidenceOverlay && (
              <img
                src={backendUrl(evidenceOverlay)}
                alt="Evidence zones: green = original pixels, magenta = AI-re-imagined"
                className="absolute inset-2 w-[calc(100%-1rem)] h-[calc(100%-1rem)] object-contain rounded-sm pointer-events-none"
              />
            )}
            {evidenceOverlay && (
              <button
                type="button"
                onClick={() => setShowEvidence((v) => !v)}
                className="absolute top-2 right-2 z-10 flex items-center gap-1.5 rounded-md bg-stone-900/80 px-2.5 py-1.5 text-[11px] font-medium text-stone-50 hover:bg-stone-900 transition-colors"
                title="Green = surviving original pixels · Magenta = AI-re-imagined zones"
              >
                {showEvidence ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                {showEvidence ? 'Hide zones' : "Show what's real"}
              </button>
            )}
          </div>
          <RestorationReportCard report={restorationReport} />
        </div>
      ) : status === 'processing' ? (
        <div className="flex-1 flex flex-col w-full">
          <PaneHeader>Restoring...</PaneHeader>
          <div className="relative w-full aspect-[4/3] bg-stone-50 rounded border border-stone-200 flex flex-col items-center justify-center p-2">
            <ProgressBar progress={progress} />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col w-full hidden md:flex opacity-30">
          <PaneHeader>Restored Output</PaneHeader>
          <div className="relative w-full aspect-[4/3] bg-stone-50 rounded border border-stone-200 flex flex-col items-center justify-center p-2">
            <ImageIcon className="w-12 h-12 text-stone-300 mb-2 stroke-1" />
            <span className="text-stone-400 text-sm">Awaiting processing</span>
          </div>
        </div>
      )}
    </div>
  );
}
