import { Camera, CheckCircle2, Loader2, AlertCircle, AlertTriangle, ExternalLink, RefreshCw } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { BACKEND_ORIGIN as BACKEND_URL } from '../services/api';

export default function Navbar() {
  // Only the setter is used — the count is read via the updater form in
  // handleCameraClick, never rendered.
  const [, setClicks] = useState(0);
  const [backendState, setBackendState] = useState({
    status: 'loading', // 'loading' | 'online' | 'offline'
    url: BACKEND_URL,
    colorizerReady: false,
    failedModels: [], // model names whose loader raised — degraded mode
    message: 'Connecting to backend...',
  });
  const timerRef = useRef(null);
  const navigate = useNavigate();

  const checkBackendHealth = async () => {
    try {
      setBackendState(prev => ({ ...prev, status: prev.status === 'online' ? 'online' : 'loading', message: 'Checking backend status...' }));
      const response = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(4000) });
      if (response.ok) {
        const data = await response.json();
        const colorizerReady = data.models?.colorizer ?? data.models?.ddcolor ?? false;
        // Degraded mode: backend up, but specific model loaders failed.
        // load_errors is a { modelName: error } map from pipeline.health_status().
        const failedModels = Object.keys(data.load_errors ?? {});
        setBackendState({
          status: 'online',
          url: data.url || BACKEND_URL,
          colorizerReady,
          failedModels,
          message: failedModels.length > 0
            ? `Degraded — ${failedModels.join(', ')} failed to load`
            : colorizerReady
            ? 'Backend & All AI Models Ready'
            : 'Backend Ready (DDColor loading in background)',
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch {
      setBackendState({
        status: 'offline',
        url: BACKEND_URL,
        colorizerReady: false,
        failedModels: [],
        message: 'Backend unreachable. Retrying...',
      });
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 5000);
    return () => {
      clearInterval(interval);
      // The easter-egg tap timer also has to go, or it fires after unmount.
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleCameraClick = () => {
    setClicks((prev) => {
      const newCount = prev + 1;
      if (newCount >= 3) {
        navigate('/developer/weights');
        return 0;
      }
      return newCount;
    });

    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => {
      setClicks(0);
    }, 1500);
  };

  return (
    <nav className="bg-white border-b border-stone-200 text-stone-800 shadow-sm w-full relative">
      {/* Top Animated Loading Line while backend is connecting/loading */}
      {backendState.status === 'loading' && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-amber-100 overflow-hidden z-50">
          <div className="h-full bg-amber-500 animate-pulse w-full origin-left transition-all duration-500" />
        </div>
      )}

      <div className="w-full px-4 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-3">
        {/* Logo & Title */}
        <div className="flex items-center space-x-3">
          <div onClick={handleCameraClick} className="cursor-pointer select-none pointer-events-auto hover:opacity-80 transition-opacity">
            <Camera className="w-7 h-7 text-stone-700 pointer-events-none" />
          </div>
          <Link to="/" className="text-xl font-semibold tracking-wide text-stone-900 hover:text-stone-700 transition-colors">
            PixRevive
          </Link>
        </div>

        {/* Backend Loading & URL Link Indicator */}
        <div className="flex items-center space-x-2 text-xs md:text-sm">
          <div
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border transition-all ${
              backendState.status === 'loading'
                ? 'bg-amber-50 border-amber-300 text-amber-900 shadow-sm animate-pulse'
                : backendState.status === 'online'
                ? (backendState.failedModels?.length > 0
                  ? 'bg-amber-50 border-amber-300 text-amber-900'
                  : 'bg-emerald-50 border-emerald-300 text-emerald-900')
                : 'bg-rose-50 border-rose-300 text-rose-900'
            }`}
          >
            {backendState.status === 'loading' && (
              <Loader2 className="w-4 h-4 text-amber-600 animate-spin flex-shrink-0" />
            )}
            {backendState.status === 'online' && (backendState.failedModels?.length > 0 ? (
              <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
            ) : (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
            ))}
            {backendState.status === 'offline' && (
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            )}

            <div className="flex items-center space-x-1.5">
              <span className="font-medium">
                {backendState.status === 'loading' && 'Backend Loading:'}
                {backendState.status === 'online' && (backendState.failedModels?.length > 0 ? 'Backend Degraded:' : 'Backend Online:')}
                {backendState.status === 'offline' && 'Backend Disconnected:'}
              </span>

              <a
                href={`${backendState.url}/health`}
                target="_blank"
                rel="noreferrer"
                className="underline font-mono font-medium hover:text-stone-900 inline-flex items-center space-x-0.5"
                title="Click to view backend status page"
              >
                <span>{backendState.url}</span>
                <ExternalLink className="w-3 h-3 opacity-60 ml-0.5" />
              </a>

              {backendState.status === 'online' && (
                <span className="text-[11px] opacity-75 font-normal ml-1 hidden sm:inline">
                  ({backendState.colorizerReady ? 'AI Fully Loaded' : 'DDColor loading...'})
                </span>
              )}
            </div>

            {backendState.status === 'offline' && (
              <button
                onClick={checkBackendHealth}
                className="ml-1 p-1 hover:bg-rose-100 rounded-full transition-colors"
                title="Retry connecting to backend"
              >
                <RefreshCw className="w-3.5 h-3.5 text-rose-700" />
              </button>
            )}
          </div>
        </div>

        {/* Navigation Links */}
        <div className="flex items-center space-x-4">
          <Link to="/documentation" className="text-sm font-medium text-stone-500 hover:text-stone-900 transition-colors">
            Documentation
          </Link>
        </div>
      </div>
    </nav>
  );
}
