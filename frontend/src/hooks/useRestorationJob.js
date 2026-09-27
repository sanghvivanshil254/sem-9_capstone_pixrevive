// useRestorationJob — job state + SSE streaming (plan 5.6 / A-7).
//
// Extracted from ImageUploader.jsx so the component only renders.
// Also hosts the Q-8 preset selector state: presets come from /api/presets
// and applying one sets the pipeline toggles.
import { useState, useCallback, useEffect, useRef } from 'react';
import { apiUrl } from '../services/api';

// Keep in sync with MAX_FILE_SIZE in backend/routers/image_router.py.
export const MAX_FILE_SIZE_MB = 25;

export function useRestorationJob() {
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
  // Phase H1/H2 — Re-imagine (beta): opt-in evidence-preserving diffusion for
  // unrecoverable damage. Off by default; the backend still gates on hopeless
  // coverage and available resources, and the report discloses the outcome.
  const [enableReimagine, setEnableReimagine] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState('auto');
  const [originalSize, setOriginalSize] = useState(null);
  const [resultSize, setResultSize] = useState(null);
  const [restorationReport, setRestorationReport] = useState(null);
  const abortRef = useRef(null);

  // ── Q-8 presets ──────────────────────────────────────────────────────
  // { id → {label, description, toggles:{enable_*}} } fetched from the
  // backend (single source of truth). `null` until loaded.
  const [presets, setPresets] = useState(null);
  const [activePreset, setActivePreset] = useState('default');
  // Apply a preset from the latest presets without making applyPreset a
  // dependency of the fetch effect (a presets→applyPreset→effect→setPresets
  // cycle here caused an infinite /api/presets refetch loop that made every
  // option feel un-clickable — the state churn reset it under the cursor).
  const presetsRef = useRef(null);
  presetsRef.current = presets;
  const applyPreset = useCallback((id) => {
    const preset = presetsRef.current?.[id];
    if (!preset) return;
    const t = preset.toggles || {};
    // The UI's "AI Restoration & Upscaling" checkbox collapses the backend's
    // enable_face_restore + enable_sr toggles (restore=True ⇒ both on).
    setEnableRestore(t.enable_face_restore !== false);
    setEnableRepair(t.enable_repair !== false);
    setEnableDenoise(t.enable_denoise === true);
    setEnableColorize(t.enable_colorization !== false);
    setEnableReimagine(t.enable_reimagine === true);
    setActivePreset(id);
  }, []);
  useEffect(() => {
    // Fetch presets ONCE per mount. setPresets stores a STABLE object (the
    // parsed response); the ref above keeps applyPreset dependency-free.
    let cancelled = false;
    fetch(apiUrl('/presets'))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data) => {
        if (cancelled || !data?.presets) return;
        setPresets(data.presets);
        // Seed toggles from the default preset without depending on it.
        const t = data.presets.default?.toggles || {};
        setEnableRestore(t.enable_face_restore !== false);
        setEnableRepair(t.enable_repair !== false);
        setEnableDenoise(t.enable_denoise === true);
        setEnableColorize(t.enable_colorization !== false);
        setEnableReimagine(t.enable_reimagine === true);
      })
      .catch((err) => console.warn('Presets unavailable, using defaults:', err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Object-URL lifecycle ─────────────────────────────────────────────
  // URLs currently handed to <img>/<a>. Tracked in refs so they can be
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

  // ── Job control ──────────────────────────────────────────────────────
  const resetJob = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setFile(null);
    setPreviewUrl(null);
    setResultUrl(null);
    setOriginalSize(null);
    setResultSize(null);
    setRestorationReport(null);
    setStatus('idle');
    setErrorMsg('');
    setProgress(null);
  }, [setPreviewUrl, setResultUrl]);

  const handleFileSelection = useCallback((selectedFile) => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (!selectedFile) {
      resetJob();
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      // Reject locally rather than uploading megabytes only to get a 400 back.
      setFile(null);
      setPreviewUrl(null);
      setResultUrl(null);
      setOriginalSize(null);
      setResultSize(null);
      setRestorationReport(null);
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
    setRestorationReport(null);
    setStatus('idle');
    setErrorMsg('');
    setProgress(null);
  }, [resetJob, setPreviewUrl, setResultUrl]);

  const processImage = useCallback(async () => {
    if (!file) return;
    setStatus('processing');
    setProgress({ stage: 'analyze', percent: 0, message: 'Starting...' });
    setErrorMsg('');
    setResultUrl(null);
    setRestorationReport(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('restore', enableRestore);
    formData.append('repair', enableRepair);
    formData.append('colorize', enableColorize);
    formData.append('denoise', enableDenoise);
    formData.append('reimagine', enableReimagine);
    formData.append('device', selectedDevice);

    try {
      // Use EventSource-compatible streaming via fetch
      const controller = new AbortController();
      abortRef.current = controller;

      const response = await fetch(apiUrl('/process-stream'), {
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
            setRestorationReport(data.restoration_report || null);

            // Fetch the result using the task_id
            const imgResponse = await fetch(apiUrl(`/process/${data.task_id}/result`));
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
  }, [file, enableRestore, enableRepair, enableColorize, enableDenoise, enableReimagine, selectedDevice, setResultUrl]);

  return {
    // media + job state
    file, preview, status, errorMsg, resultImage, progress,
    originalSize, resultSize, restorationReport,
    // toggles
    enableRestore, enableRepair, enableDenoise, enableColorize, enableReimagine,
    setEnableRestore, setEnableRepair, setEnableDenoise, setEnableColorize, setEnableReimagine,
    selectedDevice, setSelectedDevice,
    // presets
    presets, activePreset, applyPreset, setActivePreset,
    // actions
    handleFileSelection, processImage,
    setOriginalSize, setResultSize,
  };
}
