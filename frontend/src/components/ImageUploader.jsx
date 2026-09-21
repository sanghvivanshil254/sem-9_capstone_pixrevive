import { useState, useCallback, useEffect, useRef } from 'react';
import { Upload, Loader2, Download, Image as ImageIcon, ShieldCheck, ChevronDown, ChevronUp, Settings2 } from 'lucide-react';
import ProgressBar from './ProgressBar';

const API_BASE = 'http://localhost:8000/api';
// Keep in sync with MAX_FILE_SIZE in backend/routers/image_router.py.
const MAX_FILE_SIZE_MB = 25;

export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState('idle'); 
  const [errorMsg, setErrorMsg] = useState('');
  const [resultImage, setResultImage] = useState(null);
  const [progress, setProgress] = useState(null);
  const [enableRestore, setEnableRestore] = useState(true);
  const [enableRepair, setEnableRepair] = useState(true);
  const [enableDenoise, setEnableDenoise] = useState(true);
  const [enableColorize, setEnableColorize] = useState(true);
  const [selectedDevice, setSelectedDevice] = useState('auto');
  const [showMoreOptions, setShowMoreOptions] = useState(false);
  const [originalSize, setOriginalSize] = useState(null);
  const [resultSize, setResultSize] = useState(null);
  const abortRef = useRef(null);
  // Object URLs currently handed to <img>/<a>. Tracked in refs so they can be
  // revoked when replaced and on unmount — otherwise every selected file and
  // every result blob leaks for the lifetime of the page.
  const previewUrlRef = useRef(null);
  const resultUrlRef = useRef(null);

  const setPreviewUrl = useCallback((url) => {
    if (previewUrlRef.current && previewUrlRef.current !== url) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    previewUrlRef.current = url;
    setPreview(url);
  }, []);

  const setResultUrl = useCallback((url) => {
    if (resultUrlRef.current && resultUrlRef.current !== url) {
      URL.revokeObjectURL(resultUrlRef.current);
    }
    resultUrlRef.current = url;
    setResultImage(url);
  }, []);

  // Unmount-only cleanup. Safe under StrictMode's double-mount: both refs are
  // still null on the first mount, so the simulated unmount revokes nothing.
  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current);
    previewUrlRef.current = null;
    resultUrlRef.current = null;
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const handleFileSelection = useCallback((selectedFile) => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (!selectedFile) {
      setFile(null);
      setPreviewUrl(null);
      setResultUrl(null);
      setOriginalSize(null);
      setResultSize(null);
      setStatus('idle');
      setErrorMsg('');
      setProgress(null);
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      // Reject locally rather than uploading megabytes only to get a 400 back.
      setFile(null);
      setPreviewUrl(null);
      setResultUrl(null);
      setOriginalSize(null);
      setResultSize(null);
      setProgress(null);
      setStatus('error');
      setErrorMsg(
        `That file is ${(selectedFile.size / 1024 / 1024).toFixed(1)} MB. ` +
        `Maximum size is ${MAX_FILE_SIZE_MB} MB.`
      );
      return;
    }
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setOriginalSize(null);
    setResultUrl(null);
    setResultSize(null);
    setStatus('idle');
    setErrorMsg('');
    setProgress(null);
  }, [setPreviewUrl, setResultUrl]);

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

  const processImage = async () => {
    if (!file) return;
    setStatus('processing');
    setProgress({ stage: 'analyze', percent: 0, message: 'Starting...' });
    setErrorMsg('');
    setResultUrl(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('restore', enableRestore);
    formData.append('repair', enableRepair);
    formData.append('colorize', enableColorize);
    formData.append('denoise', enableDenoise);
    formData.append('device', selectedDevice);

    try {
      // Use EventSource-compatible streaming via fetch
      const controller = new AbortController();
      abortRef.current = controller;

      const response = await fetch(`${API_BASE}/process-stream`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `Server error ${response.status}`);
      }
      if (!response.body) {
        throw new Error('Streaming is not supported by this browser.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      // Declared OUTSIDE the read loop: an `event:` line and its `data:` line
      // can land in different chunks, and resetting per chunk would drop the
      // event type — silently discarding completion and error events.
      let eventType = '';
      let terminated = false;

      while (!terminated) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
            continue;
          }
          if (!line.startsWith('data: ')) continue;

          const dataStr = line.slice(6).trim();
          let data;
          try {
            // Only the parse is guarded. Wrapping the handlers too would
            // swallow the errors they throw and leave the UI spinning
            // in 'processing' forever.
            data = JSON.parse(dataStr);
          } catch (parseErr) {
            console.warn('SSE parse warning:', parseErr, dataStr);
            continue;
          }

          if (eventType === 'progress') {
            setProgress({ stage: data.stage, percent: data.percent, message: data.message });
          } else if (eventType === 'complete') {
            setProgress({ stage: 'done', percent: 100, message: data.message });

            // Fetch the result using the task_id
            const imgResponse = await fetch(`${API_BASE}/process/${data.task_id}/result`);
            if (!imgResponse.ok) {
              let detail = `Failed to fetch result (${imgResponse.status})`;
              try {
                const body = await imgResponse.json();
                if (body?.detail) detail = body.detail;
              } catch {
                // Non-JSON error body — keep the status-code message.
              }
              throw new Error(detail);
            }
            const blob = await imgResponse.blob();
            setResultUrl(URL.createObjectURL(blob));
            setResultSize(null);
            setStatus('done');
            terminated = true;
            break;
          } else if (eventType === 'error') {
            throw new Error(data.message || 'Processing failed');
          }
        }
      }

      if (!terminated) {
        // The stream closed without a `complete` or `error` event — e.g. the
        // server process died mid-run. Without this the button spins forever.
        throw new Error('Connection closed before restoration finished.');
      }
    } catch (error) {
      if (error.name === 'AbortError') return;
      console.error(error);
      setStatus('error');
      setProgress(null);
      setErrorMsg(error.message || 'Network Error or Backend is not running.');
    } finally {
      abortRef.current = null;
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
          <div className="flex flex-col md:flex-row gap-8 items-start justify-center">
            {/* Original Image */}
            <div className="flex-1 flex flex-col w-full">
              <div className="flex items-center justify-between mb-3 border-b border-stone-100 pb-2">
                 <h3 className="text-stone-800 font-semibold text-sm uppercase tracking-wider">Original Source</h3>
              </div>
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
                <div className="flex items-center justify-between mb-3 border-b border-stone-100 pb-2">
                  <h3 className="text-stone-800 font-semibold text-sm uppercase tracking-wider">Restored Output</h3>
                </div>
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
                </div>
              </div>
            ) : status === 'processing' ? (
              <div className="flex-1 flex flex-col w-full">
                <div className="flex items-center justify-between mb-3 border-b border-stone-100 pb-2">
                  <h3 className="text-stone-800 font-semibold text-sm uppercase tracking-wider">Restoring...</h3>
                </div>
                <div className="relative w-full aspect-[4/3] bg-stone-50 rounded border border-stone-200 flex flex-col items-center justify-center p-2">
                  <ProgressBar progress={progress} />
                </div>
              </div>
            ) : (
               <div className="flex-1 flex flex-col w-full hidden md:flex opacity-30">
                 <div className="flex items-center justify-between mb-3 border-b border-stone-100 pb-2">
                    <h3 className="text-stone-800 font-semibold text-sm uppercase tracking-wider">Restored Output</h3>
                 </div>
                 <div className="relative w-full aspect-[4/3] bg-stone-50 rounded border border-stone-200 flex flex-col items-center justify-center p-2">
                    <ImageIcon className="w-12 h-12 text-stone-300 mb-2 stroke-1" />
                    <span className="text-stone-400 text-sm">Awaiting processing</span>
                 </div>
               </div>
            )}
          </div>
          

          
          {/* Auto-applied info + More Options toggle */}
          {status === 'idle' && (
            <div className="flex flex-col items-center gap-3 mt-4">
              
              {/* Auto-applied badge */}
              <div className="flex items-center gap-2 text-sm text-stone-500">
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
                <span>
                  <strong className="text-stone-700">{activeOptionsCount} enhancements</strong> will be applied automatically
                </span>
              </div>

              {/* More Options collapsible */}
              <button
                type="button"
                onClick={() => setShowMoreOptions(!showMoreOptions)}
                className="flex items-center gap-1.5 text-sm text-stone-500 hover:text-stone-700 transition-colors cursor-pointer font-medium"
              >
                <Settings2 className="w-3.5 h-3.5" />
                More Options
                {showMoreOptions ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showMoreOptions && (
                <div className="flex flex-col items-center gap-3 text-sm text-stone-600 bg-stone-50 py-4 px-6 rounded border border-stone-200 w-fit mx-auto animate-in fade-in duration-200">
                  
                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-indigo-600" />
                    <input 
                      type="checkbox" 
                      id="enableRestore" 
                      checked={enableRestore}
                      onChange={(e) => setEnableRestore(e.target.checked)}
                      className="w-4 h-4 text-indigo-600 rounded border-stone-300 focus:ring-indigo-500 cursor-pointer"
                    />
                    <label htmlFor="enableRestore" className="cursor-pointer font-medium select-none">
                      AI Restoration &amp; Upscaling (4× RealESRGAN)
                    </label>
                  </div>

                  <div className="w-full h-px bg-stone-200"></div>

                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-violet-500" />
                    <input 
                      type="checkbox" 
                      id="enableRepair" 
                      checked={enableRepair}
                      onChange={(e) => setEnableRepair(e.target.checked)}
                      className="w-4 h-4 text-violet-500 rounded border-stone-300 focus:ring-violet-500 cursor-pointer"
                    />
                    <label htmlFor="enableRepair" className="cursor-pointer font-medium select-none">
                      Repair Damage (Scratches, White Spots, Cracks)
                    </label>
                  </div>

                  <div className="w-full h-px bg-stone-200"></div>

                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-rose-500" />
                    <input 
                      type="checkbox" 
                      id="enableDenoise" 
                      checked={enableDenoise}
                      onChange={(e) => setEnableDenoise(e.target.checked)}
                      className="w-4 h-4 text-rose-500 rounded border-stone-300 focus:ring-rose-500 cursor-pointer"
                    />
                    <label htmlFor="enableDenoise" className="cursor-pointer font-medium select-none">
                      Remove Dust &amp; Scratches (Denoise)
                    </label>
                  </div>

                  <div className="w-full h-px bg-stone-200"></div>

                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-indigo-500" />
                    <input 
                      type="checkbox" 
                      id="enableColorize" 
                      checked={enableColorize}
                      onChange={(e) => setEnableColorize(e.target.checked)}
                      className="w-4 h-4 text-indigo-500 rounded border-stone-300 focus:ring-indigo-500 cursor-pointer"
                    />
                    <label htmlFor="enableColorize" className="cursor-pointer font-medium select-none">
                      AI Colorization (For B&amp;W photos)
                    </label>
                  </div>

                  {/* Device Preference */}
                  <div className="flex flex-col sm:flex-row items-center gap-3 w-full justify-between pt-1">
                    <span className="font-semibold text-xs text-stone-500 uppercase tracking-wider">Device:</span>
                    <div className="inline-flex rounded-lg border border-stone-200 bg-stone-100 p-0.5 shadow-sm text-xs font-semibold">
                      {[
                        { id: 'auto', label: 'Auto' },
                        { id: 'cpu', label: 'CPU' },
                        { id: 'gpu', label: 'GPU' }
                      ].map((dev) => (
                        <button
                          key={dev.id}
                          type="button"
                          onClick={() => setSelectedDevice(dev.id)}
                          className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                            selectedDevice === dev.id
                              ? 'bg-white text-stone-900 shadow-sm border border-stone-200/50'
                              : 'text-stone-500 hover:text-stone-800'
                          }`}
                        >
                          {dev.label}
                        </button>
                      ))}
                    </div>
                  </div>

                </div>
              )}

            </div>
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
