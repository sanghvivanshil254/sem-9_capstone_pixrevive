// ImageUploader — composition root (plan 5.6 / A-7).
// Job state lives in useRestorationJob; rendering is split into ResultCard
// (before/after + Restoration Report) and OptionsPanel (presets + toggles).
import { useState, useCallback } from 'react';
import { Upload, Loader2, Download } from 'lucide-react';
import { useRestorationJob, MAX_FILE_SIZE_MB } from '../hooks/useRestorationJob';
import ResultCard from './ResultCard';
import OptionsPanel from './OptionsPanel';

export default function ImageUploader() {
  const job = useRestorationJob();
  const [showMoreOptions, setShowMoreOptions] = useState(false);

  const {
    file, preview, status, errorMsg, resultImage, progress,
    originalSize, resultSize, restorationReport,
    enableRestore, enableRepair, enableDenoise, enableColorize, enableReimagine,
    setEnableRestore, setEnableRepair, setEnableDenoise, setEnableColorize, setEnableReimagine,
    selectedDevice, setSelectedDevice,
    presets, activePreset, applyPreset,
    handleFileSelection, processImage,
    setOriginalSize, setResultSize,
  } = job;

  const onDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      handleFileSelection(droppedFile);
    }
  }, [handleFileSelection]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const activeOptionsCount = [enableRestore, enableRepair, enableDenoise, enableColorize].filter(Boolean).length;

  return (
    <div className="bg-white border border-stone-200 rounded-lg p-8 shadow-sm">
      {!preview ? (
        <div>
          <div
            className="border border-dashed border-stone-300 bg-stone-50 hover:bg-stone-100 hover:border-stone-400 rounded-lg p-16 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[300px]"
            onDragOver={onDragOver}
            onDrop={onDrop}
            onClick={() => document.getElementById('fileUpload').click()}
          >
            <input
              type="file"
              id="fileUpload"
              className="hidden"
              accept="image/jpeg, image/png, image/webp"
              onChange={handleFileChange}
            />
            <Upload className="w-10 h-10 text-stone-400 mb-4 stroke-1" />
            <p className="text-lg text-stone-700 font-medium">Select or drag an image</p>
            <p className="text-sm text-stone-500 mt-2 font-light">
              JPG, PNG, WEBP up to {MAX_FILE_SIZE_MB}MB
            </p>
          </div>
          {status === 'error' && (
            <div className="text-red-700 bg-red-50 p-4 mt-6 rounded border border-red-200 max-w-lg mx-auto text-sm text-center">
              {errorMsg}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-8">
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

          <div className="flex flex-wrap justify-center items-center gap-4 pt-6 border-t border-stone-100">
            <button
              onClick={() => handleFileSelection(null)}
              className="px-6 py-2.5 rounded border border-stone-300 bg-white hover:bg-stone-50 text-stone-700 text-sm font-medium transition-colors"
            >
              Start Over
            </button>

            {(status === 'idle' || status === 'error') && (
              <button
                onClick={processImage}
                className="px-6 py-2.5 rounded bg-stone-900 hover:bg-stone-800 text-white text-sm font-medium transition-colors shadow-sm"
              >
                {status === 'error' ? 'Try Again' : 'Process Image'}
              </button>
            )}

            {status === 'processing' && (
              <button disabled className="px-6 py-2.5 rounded border border-stone-200 bg-stone-50 text-stone-500 text-sm font-medium flex items-center gap-2 cursor-not-allowed">
                <Loader2 className="w-4 h-4 animate-spin" />
                {progress?.stage === 'done' ? 'Loading result...' : `${progress?.percent || 0}% - ${progress?.message || 'Processing...'}`}
              </button>
            )}

            {status === 'done' && resultImage && (
              <a
                href={resultImage}
                download={`restored_${file?.name?.replace(/\.[^.]+$/, '') || 'image'}.png`}
                className="px-6 py-2.5 rounded bg-stone-900 hover:bg-stone-800 text-white text-sm font-medium transition-colors shadow-sm flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download Result
              </a>
            )}
          </div>

          {status === 'error' && (
            <div className="text-red-700 bg-red-50 p-4 rounded border border-red-200 max-w-lg mx-auto text-sm text-center">
              {errorMsg}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
