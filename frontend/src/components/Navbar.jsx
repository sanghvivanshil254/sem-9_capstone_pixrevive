import {
  Camera,
  CheckCircle2,
  Loader2,
  AlertCircle,
  AlertTriangle,
  ExternalLink,
  RefreshCw,
  Menu,
  X,
  BookOpen,
  Cpu
} from 'lucide-react';
import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { BACKEND_ORIGIN as BACKEND_URL } from '../services/api';

export default function Navbar() {
  const [, setClicks] = useState(0);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [backendState, setBackendState] = useState({
    status: 'loading', // 'loading' | 'online' | 'offline'
    url: BACKEND_URL,
    colorizerReady: false,
    failedModels: [],
    message: 'Connecting to backend...',
  });
  const timerRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  // Close mobile menu on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileMenuOpen]);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [mobileMenuOpen]);

  const checkBackendHealth = useCallback(async () => {
    try {
      setBackendState((prev) => ({
        ...prev,
        status: prev.status === 'online' ? 'online' : 'loading',
        message: 'Checking backend status...',
      }));
      const response = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(4000) });
      if (response.ok) {
        const data = await response.json();
        const colorizerReady = data.models?.colorizer ?? data.models?.ddcolor ?? false;
        const failedModels = Object.keys(data.load_errors ?? {});
        setBackendState({
          status: 'online',
          url: data.url || BACKEND_URL,
          colorizerReady,
          failedModels,
          message:
            failedModels.length > 0
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
  }, []);

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 5000);
    return () => {
      clearInterval(interval);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [checkBackendHealth]);

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

  const getStatusBadgeClasses = () => {
    if (backendState.status === 'loading') {
      return 'bg-amber-50 border-amber-300 text-amber-900 shadow-sm animate-pulse';
    }
    if (backendState.status === 'online') {
      return backendState.failedModels?.length > 0
        ? 'bg-amber-50 border-amber-300 text-amber-900'
        : 'bg-emerald-50 border-emerald-300 text-emerald-900';
    }
    return 'bg-rose-50 border-rose-300 text-rose-900';
  };

  return (
    <nav className="bg-white border-b border-stone-200 text-stone-800 shadow-sm w-full relative z-40 pt-[env(safe-area-inset-top,0px)]">
      {/* Top Animated Loading Line while backend is connecting */}
      {backendState.status === 'loading' && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-amber-100 overflow-hidden z-50">
          <div className="h-full bg-amber-500 animate-pulse w-full origin-left transition-all duration-500" />
        </div>
      )}

      <div className="w-full max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 py-2.5 sm:py-3 flex items-center justify-between gap-2 sm:gap-4">
        {/* Logo & Triple-tap easter egg */}
        <div className="flex items-center space-x-2 sm:space-x-3 shrink-0">
          <button
            type="button"
            onClick={handleCameraClick}
            aria-label="PixRevive camera trigger (tap 3 times for Developer Weights)"
            className="p-2 -m-1.5 rounded-lg hover:bg-stone-100 active:bg-stone-200 transition-colors cursor-pointer flex items-center justify-center min-w-[44px] min-h-[44px]"
          >
            <Camera className="w-6 h-6 sm:w-7 sm:h-7 text-stone-700 pointer-events-none" />
          </button>
          <Link
            to="/"
            className="text-lg sm:text-xl font-semibold tracking-wide text-stone-900 hover:text-stone-700 transition-colors py-1"
          >
            PixRevive
          </Link>
        </div>

        {/* Backend Status Pill (Desktop & Tablet) */}
        <div className="hidden sm:flex items-center space-x-2 text-xs md:text-sm min-w-0">
          <div className={`flex items-center space-x-2 px-2.5 sm:px-3 py-1.5 rounded-full border transition-all ${getStatusBadgeClasses()}`}>
            {backendState.status === 'loading' && (
              <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-600 animate-spin shrink-0" />
            )}
            {backendState.status === 'online' &&
              (backendState.failedModels?.length > 0 ? (
                <AlertTriangle className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-600 shrink-0" />
              ) : (
                <CheckCircle2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-600 shrink-0" />
              ))}
            {backendState.status === 'offline' && (
              <AlertCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-rose-600 shrink-0" />
            )}

            <div className="flex items-center space-x-1.5 min-w-0">
              <span className="font-medium shrink-0 text-xs sm:text-sm">
                {backendState.status === 'loading' && 'Backend Loading:'}
                {backendState.status === 'online' &&
                  (backendState.failedModels?.length > 0 ? 'Backend Degraded:' : 'Backend Online:')}
                {backendState.status === 'offline' && 'Backend Disconnected:'}
              </span>

              <a
                href={`${backendState.url}/health`}
                target="_blank"
                rel="noreferrer"
                className="underline font-mono font-medium hover:text-stone-900 inline-flex items-center space-x-0.5 truncate max-w-[130px] md:max-w-[200px] text-xs sm:text-sm"
                title={`Click to view backend status page: ${backendState.url}/health`}
              >
                <span className="truncate">{backendState.url}</span>
                <ExternalLink className="w-3 h-3 opacity-60 ml-0.5 shrink-0" />
              </a>

              {backendState.status === 'online' && (
                <span className="text-[11px] opacity-75 font-normal ml-1 hidden lg:inline shrink-0">
                  ({backendState.colorizerReady ? 'AI Fully Loaded' : 'DDColor loading...'})
                </span>
              )}
            </div>

            {backendState.status === 'offline' && (
              <button
                type="button"
                onClick={checkBackendHealth}
                className="ml-1 p-1 hover:bg-rose-100 rounded-full transition-colors min-w-[28px] min-h-[28px] flex items-center justify-center shrink-0 cursor-pointer"
                title="Retry connecting to backend"
                aria-label="Retry connecting to backend"
              >
                <RefreshCw className="w-3.5 h-3.5 text-rose-700" />
              </button>
            )}
          </div>
        </div>

        {/* Compact Status Indicator for Small Mobile Screens */}
        <div className="flex sm:hidden items-center">
          <a
            href={`${backendState.url}/health`}
            target="_blank"
            rel="noreferrer"
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-mono font-medium ${getStatusBadgeClasses()}`}
            title={`Backend ${backendState.status}: ${backendState.url}`}
          >
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                backendState.status === 'online'
                  ? 'bg-emerald-500 animate-pulse'
                  : backendState.status === 'loading'
                  ? 'bg-amber-500 animate-pulse'
                  : 'bg-rose-500'
              }`}
            />
            <span className="capitalize">{backendState.status}</span>
          </a>
        </div>

        {/* Desktop Navigation Links */}
        <div className="hidden md:flex items-center space-x-5 shrink-0">
          <Link
            to="/documentation"
            className="text-sm font-medium text-stone-600 hover:text-stone-900 transition-colors py-2 px-1 min-h-[44px] flex items-center"
          >
            Documentation
          </Link>
          <Link
            to="/developer/weights"
            className="text-sm font-medium text-stone-500 hover:text-stone-800 transition-colors py-2 px-1 min-h-[44px] flex items-center gap-1.5"
            title="Developer Architecture & Weights Admin"
          >
            <Cpu className="w-3.5 h-3.5 text-indigo-500" />
            <span>Architecture</span>
          </Link>
        </div>

        {/* Mobile Hamburger Button */}
        <div className="flex md:hidden items-center">
          <button
            type="button"
            onClick={() => setMobileMenuOpen((open) => !open)}
            aria-expanded={mobileMenuOpen}
            aria-controls="mobile-navigation-drawer"
            aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            className="p-2 -mr-1.5 text-stone-600 hover:text-stone-900 hover:bg-stone-100 rounded-lg transition-colors flex items-center justify-center min-w-[44px] min-h-[44px] cursor-pointer"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer / Dropdown Menu */}
      {mobileMenuOpen && (
        <div
          id="mobile-navigation-drawer"
          className="md:hidden fixed inset-x-0 top-[calc(56px+env(safe-area-inset-top,0px))] bottom-0 bg-stone-900/40 backdrop-blur-xs z-50 flex flex-col justify-start"
          onClick={(e) => {
            if (e.target === e.currentTarget) setMobileMenuOpen(false);
          }}
        >
          <div className="bg-white border-b border-stone-200 px-4 py-5 shadow-xl space-y-4 animate-in slide-in-from-top-2 duration-200 safe-px">
            {/* Mobile Detailed Status Card */}
            <div className={`p-3 rounded-lg border text-xs ${getStatusBadgeClasses()}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-semibold flex items-center gap-1.5">
                  {backendState.status === 'loading' && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  {backendState.status === 'online' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />}
                  {backendState.status === 'offline' && <AlertCircle className="w-3.5 h-3.5 text-rose-600" />}
                  Status: {backendState.status.toUpperCase()}
                </span>
                {backendState.status === 'offline' && (
                  <button
                    type="button"
                    onClick={checkBackendHealth}
                    className="p-1 hover:bg-rose-100 rounded text-rose-800 font-medium flex items-center gap-1 text-[11px]"
                  >
                    <RefreshCw className="w-3 h-3" /> Retry
                  </button>
                )}
              </div>
              <p className="text-stone-600 text-[11px] mb-2">{backendState.message}</p>
              <a
                href={`${backendState.url}/health`}
                target="_blank"
                rel="noreferrer"
                className="underline font-mono text-[11px] break-all flex items-center gap-1 text-stone-700"
              >
                <span>{backendState.url}/health</span>
                <ExternalLink className="w-3 h-3 shrink-0" />
              </a>
            </div>

            {/* Mobile Nav Links */}
            <div className="flex flex-col space-y-1 pt-1">
              <Link
                to="/"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium text-stone-800 hover:bg-stone-100 active:bg-stone-200 min-h-[44px] transition-colors"
              >
                <Camera className="w-5 h-5 text-stone-500" />
                <span>Restoration Workspace</span>
              </Link>

              <Link
                to="/documentation"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium text-stone-800 hover:bg-stone-100 active:bg-stone-200 min-h-[44px] transition-colors"
              >
                <BookOpen className="w-5 h-5 text-indigo-500" />
                <span>Engine Documentation</span>
              </Link>

              <Link
                to="/developer/weights"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium text-stone-800 hover:bg-stone-100 active:bg-stone-200 min-h-[44px] transition-colors"
              >
                <Cpu className="w-5 h-5 text-amber-500" />
                <span>Architecture & Weights</span>
              </Link>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
