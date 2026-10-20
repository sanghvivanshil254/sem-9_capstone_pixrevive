// ImageUploader — composition root (plan 5.6 / A-7).
// Job state lives in useRestorationJob; rendering is split into ResultCard
// (before/after + Restoration Report) and OptionsPanel (presets + toggles).
import { useState, useCallback } from 'react';
import { Upload, Loader2, Download, RotateCcw, Sparkles } from 'lucide-react';
import { useRestorationJob, MAX_FILE_SIZE_MB } from '../hooks/useRestorationJob';
import ResultCard from './ResultCard';
import OptionsPanel from './OptionsPanel';

export default function ImageUploader() {
  const job = useRestorationJob();
  const [showMoreOptions, setShowMoreOptions] = useState(false);

  const {
    file,
    preview,
    status,
    errorMsg,
    resultImage,
    progress,
    originalSize,
    resultSize,
    restorationReport,
    enableRestore,
    enableRepair,
    enableDenoise,
    enableColorize,
    enableReimagine,
    setEnableRestore,
    setEnableRepair,
    setEnableDenoise,
    setEnableColorize,
    setEnableReimagine,
    selectedDevice,
    setSelectedDevice,
    presets,
    activePreset,
    applyPreset,
    handleFileSelection,
    processImage,
    setOriginalSize,
    setResultSize,
  } = job;

  const onDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile && droppedFile.type.startsWith('image/')) {
        handleFileSelection(droppedFile);
      }
    },
    [handleFileSelection]
  );

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleDropzoneKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      document.getElementById('fileUpload')?.click();
    }
  };

  const activeOptionsCount = [enableRestore, enableRepair, enableDenoise, enableColorize].filter(Boolean).length;

  return (
    <div className="bg-white border border-stone-200 rounded-xl p-3.5 sm:p-6 md:p-8 shadow-sm transition-all w-full">
      {!preview ? (
        <div className="w-full">
          <div
            className="border-2 border-dashed border-stone-300 bg-stone-50 hover:bg-stone-100 hover:border-stone-400 focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-indigo-300 rounded-xl p-6 sm:p-10 md:p-14 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[220px] sm:min-h-[280px] md:min-h-[320px] outline-none"
            onDragOver={onDragOver}
            onDrop={onDrop}
            onClick={() => document.getElementById('fileUpload')?.click()}
            onKeyDown={handleDropzoneKeyDown}
            tabIndex={0}
            role="button"
            aria-label="Upload an image to restore. Click or drop file here."
          >
            <input
              type="file"
              id="fileUpload"
              className="hidden"
              accept="image/jpeg, image/png, image/webp"
              onChange={handleFileChange}
            />
            <div className="w-12 h-12 sm:w-14 sm:h-14 rounded-full bg-stone-100 flex items-center justify-center mb-3 sm:mb-4 text-stone-500">
              <Upload className="w-6 h-6 sm:w-7 sm:h-7 stroke-[1.5]" />
            </div>
            <p className="text-base sm:text-lg text-stone-800 font-medium">Select or drag an image</p>
            <p className="text-xs sm:text-sm text-stone-500 mt-1.5 font-light">
              JPG, PNG, WEBP up to {MAX_FILE_SIZE_MB}MB
            </p>
            <span className="mt-4 px-3.5 py-1.5 rounded-full bg-white border border-stone-200 text-xs font-semibold text-stone-600 shadow-xs sm:hidden">
              Tap to Browse
            </span>
          </div>

          {status === 'error' && (
            <div className="text-red-700 bg-red-50 p-3.5 sm:p-4 mt-4 sm:mt-6 rounded-lg border border-red-200 max-w-lg mx-auto text-xs sm:text-sm text-center leading-relaxed">
              {errorMsg}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6 sm:space-y-8 w-full">
          <ResultCard
            preview={preview}
            resultImage={resultImage}
            status={status}
            progress={progress}
            restorationReport={restorationReport}
            originalSize={originalSize}
            resultSize={resultSize}
            setOriginalSize={setOriginalSize}
            setResultSize={setResultSize}
          />

          {status === 'idle' && (
            <OptionsPanel
              presets={presets}
              activePreset={activePreset}
              applyPreset={applyPreset}
              activeOptionsCount={activeOptionsCount}
              showMoreOptions={showMoreOptions}
              setShowMoreOptions={setShowMoreOptions}
              enableRestore={enableRestore}
              enableRepair={enableRepair}
              enableDenoise={enableDenoise}
              enableColorize={enableColorize}
              enableReimagine={enableReimagine}
              setEnableRestore={setEnableRestore}
              setEnableRepair={setEnableRepair}
              setEnableDenoise={setEnableDenoise}
              setEnableColorize={setEnableColorize}
              setEnableReimagine={setEnableReimagine}
              selectedDevice={selectedDevice}
              setSelectedDevice={setSelectedDevice}
            />
          )}

          {/* Action Buttons Row */}
          <div className="flex flex-col sm:flex-row justify-center items-stretch sm:items-center gap-3 sm:gap-4 pt-4 sm:pt-6 border-t border-stone-100 w-full">
            <button
              type="button"
              onClick={() => handleFileSelection(null)}
              className="px-5 py-2.5 rounded-lg border border-stone-300 bg-white hover:bg-stone-50 active:bg-stone-100 text-stone-700 text-sm font-medium transition-colors min-h-[44px] flex items-center justify-center gap-2 cursor-pointer"
            >
              <RotateCcw className="w-4 h-4 text-stone-500 shrink-0" />
              <span>Start Over</span>
            </button>

            {(status === 'idle' || status === 'error') && (
              <button
                type="button"
                onClick={processImage}
                className="px-6 py-2.5 rounded-lg bg-stone-900 hover:bg-stone-800 active:bg-stone-950 text-white text-sm font-semibold transition-colors shadow-sm min-h-[44px] flex items-center justify-center gap-2 cursor-pointer"
              >
                <Sparkles className="w-4 h-4 text-amber-400 shrink-0" />
                <span>{status === 'error' ? 'Try Again' : 'Process Image'}</span>
              </button>
            )}

            {status === 'processing' && (
              <button
                disabled
                className="px-5 py-2.5 rounded-lg border border-stone-200 bg-stone-50 text-stone-600 text-xs sm:text-sm font-medium flex items-center justify-center gap-2 cursor-not-allowed min-h-[44px] max-w-full"
              >
                <Loader2 className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
                <span className="truncate">
                  {progress?.stage === 'done'
                    ? 'Loading result...'
                    : `${progress?.percent || 0}% — ${progress?.message || 'Processing...'}`}
                </span>
              </button>
            )}

            {status === 'done' && resultImage && (
              <a
                href={resultImage}
                download={`restored_${file?.name?.replace(/\.[^.]+$/, '') || 'image'}.png`}
                className="px-6 py-2.5 rounded-lg bg-stone-900 hover:bg-stone-800 active:bg-stone-950 text-white text-sm font-semibold transition-colors shadow-sm flex items-center justify-center gap-2 min-h-[44px] cursor-pointer"
              >
                <Download className="w-4 h-4 shrink-0" />
                <span>Download Result</span>
              </a>
            )}
          </div>

          {status === 'error' && (
            <div className="text-red-700 bg-red-50 p-3.5 sm:p-4 rounded-lg border border-red-200 max-w-lg mx-auto text-xs sm:text-sm text-center leading-relaxed">
              {errorMsg}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
