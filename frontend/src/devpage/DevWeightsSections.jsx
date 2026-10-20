import React, { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';
import { CheckCircle, XCircle } from 'lucide-react';
import { API_BASE as API_URL } from '../services/api';

// Reused design language from the app (cream-card / mono badges / stone palette).

export function Section({ id, num, title, kicker, children }) {
  return (
    <section id={id} className="scroll-mt-24 mb-10 sm:mb-14 md:mb-20 w-full">
      <div className="mb-4 sm:mb-6 flex items-baseline gap-2.5 sm:gap-3 flex-wrap">
        <span className="text-xs font-mono text-indigo-600 font-bold">{num}</span>
        <h2 className="text-xl sm:text-2xl font-serif text-stone-800 tracking-tight">{title}</h2>
      </div>
      {kicker && <p className="text-xs sm:text-sm text-stone-600 leading-relaxed mb-4 sm:mb-6 -mt-2 sm:-mt-3">{kicker}</p>}
      {children}
    </section>
  );
}

export function Card({ children, className = '' }) {
  return <div className={`cream-card p-4 sm:p-6 ${className}`}>{children}</div>;
}

// Draggable before/after comparison slider:
// Uses CSS clipPath for 100% robust responsiveness across rotations & resizes,
// touch-action: none to allow smooth dragging on phones, pointer capture for reliable tracking,
// and 44x44px minimum touch targets.
export function BeforeAfter({ before, after, beforeLabel, afterLabel, caption, alt }) {
  const [pos, setPos] = useState(50);
  const containerRef = useRef(null);
  const isDragging = useRef(false);

  const updatePosition = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const percentage = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(100, Math.max(0, percentage)));
  }, []);

  const handlePointerDown = (e) => {
    isDragging.current = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // Fallback if pointer capture is not supported
    }
    updatePosition(e.clientX);
  };

  const handlePointerMove = (e) => {
    if (isDragging.current) {
      updatePosition(e.clientX);
    }
  };

  const handlePointerUp = (e) => {
    isDragging.current = false;
    try {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    } catch {
      // Ignore
    }
  };

  return (
    <figure className="mb-4 sm:mb-6 w-full">
      <div
        ref={containerRef}
        className="relative w-full overflow-hidden rounded-xl border border-stone-200 select-none cursor-ew-resize bg-stone-100 touch-none shadow-xs"
        style={{ aspectRatio: '4 / 3' }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 5));
          if (e.key === 'ArrowRight') setPos((p) => Math.min(100, p + 5));
        }}
        tabIndex={0}
        role="slider"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos)}
        aria-label="Before and after photo restoration comparison slider"
      >
        {/* After (Restored) Image (Bottom Layer) */}
        <img
          src={after}
          alt={alt}
          loading="lazy"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          draggable="false"
        />

        {/* Before (Degraded) Image (Top Layer clipped by pos) */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        >
          <img
            src={before}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            draggable="false"
          />
        </div>

        {/* Divider Line & Grab Handle */}
        <div
          className="absolute top-0 bottom-0 bg-white/90 w-0.5 pointer-events-none transition-none shadow-sm"
          style={{ left: `${pos}%` }}
        >
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-11 h-11 rounded-full bg-white/95 border border-stone-300 shadow-md flex items-center justify-center pointer-events-none backdrop-blur-xs">
            <span className="text-xs font-mono text-stone-700 select-none">&harr;</span>
          </div>
        </div>

        {/* Before / After Labels */}
        <span className="absolute top-2.5 left-2.5 text-[10px] font-mono uppercase bg-stone-900/75 backdrop-blur-xs text-stone-100 px-2 py-0.5 rounded shadow-xs pointer-events-none select-none">
          {beforeLabel}
        </span>
        <span className="absolute top-2.5 right-2.5 text-[10px] font-mono uppercase bg-indigo-600/85 backdrop-blur-xs text-white px-2 py-0.5 rounded shadow-xs pointer-events-none select-none">
          {afterLabel}
        </span>
      </div>
      {caption && <figcaption className="text-xs text-stone-500 mt-2 leading-relaxed">{caption}</figcaption>}
    </figure>
  );
}

// Depth toggle shared through context-free props drilling with accessible 44px touch targets.
export function DepthToggle({ depth, setDepth }) {
  return (
    <div className="inline-flex rounded-lg border border-stone-200 overflow-hidden text-xs font-semibold shadow-2xs">
      <button
        type="button"
        onClick={() => setDepth('beginner')}
        aria-pressed={depth === 'beginner'}
        className={`px-4 py-2.5 min-h-[44px] transition-colors cursor-pointer flex items-center justify-center ${
          depth === 'beginner'
            ? 'bg-stone-900 text-stone-100'
            : 'bg-white text-stone-600 hover:bg-stone-50 active:bg-stone-100'
        }`}
      >
        Beginner
      </button>
      <button
        type="button"
        onClick={() => setDepth('technical')}
        aria-pressed={depth === 'technical'}
        className={`px-4 py-2.5 min-h-[44px] transition-colors cursor-pointer flex items-center justify-center ${
          depth === 'technical'
            ? 'bg-stone-900 text-stone-100'
            : 'bg-white text-stone-600 hover:bg-stone-50 active:bg-stone-100'
        }`}
      >
        Technical
      </button>
    </div>
  );
}

export function DepthNote({ show, children }) {
  if (!show) return null;
  return (
    <div className="mt-3 text-xs text-stone-600 leading-relaxed bg-stone-50 border border-stone-200 rounded-lg p-3 sm:p-4">
      {children}
    </div>
  );
}

// Live backend health badge with graceful offline fallback.
export function HealthBadge() {
  const [health, setHealth] = useState(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const res = await axios.get(`${API_URL}/health`, { timeout: 4000 });
        if (alive) {
          setHealth(res.data);
          setDown(false);
        }
      } catch {
        if (alive) setDown(true);
      }
    };
    poll();
    const t = setInterval(poll, 15000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (down) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-mono text-stone-500 border border-stone-200 bg-stone-50 px-3 py-1.5 rounded-full flex-wrap">
        <XCircle className="w-3.5 h-3.5 text-stone-400 shrink-0" />
        <span>backend offline — showing on-disk facts only</span>
      </span>
    );
  }
  if (!health) return null;
  const ready = health.ready;
  const device = health.device;
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs font-mono px-3 py-1.5 rounded-full border flex-wrap ${
        ready
          ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
          : 'text-amber-700 bg-amber-50 border-amber-200'
      }`}
    >
      <CheckCircle className="w-3.5 h-3.5 shrink-0" />
      <span>backend {String(device)} &bull; {ready ? 'ready' : 'loading models'}</span>
    </span>
  );
}

// Scroll-spy table of contents.
export function TocRail({ sections }) {
  const [active, setActive] = useState(sections[0]?.id);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id);
        });
      },
      { rootMargin: '-15% 0px -70% 0px' }
    );
    sections.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [sections]);

  return (
    <>
      {/* Mobile / Tablet Collapsible Menu */}
      <div className="xl:hidden mb-6 w-full">
        <details className="cream-card p-3 sm:p-4 text-sm w-full group">
          <summary className="cursor-pointer font-semibold text-stone-800 text-xs sm:text-sm uppercase tracking-wider flex items-center justify-between min-h-[44px] px-1 select-none">
            <span>Table of Contents ({sections.length} sections)</span>
            <span className="text-stone-400 font-normal text-xs group-open:rotate-180 transition-transform">&darr;</span>
          </summary>
          <nav className="mt-3 flex flex-col gap-1 border-t border-stone-100 pt-3">
            {sections.map((s) => (
              <a
                key={s.id}
                href={`#${s.id}`}
                className={`px-3 py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-colors flex items-center gap-2 min-h-[40px] ${
                  active === s.id
                    ? 'bg-stone-900 text-white font-semibold'
                    : 'text-stone-600 hover:bg-stone-100 hover:text-stone-900'
                }`}
              >
                <span className="font-mono text-[10px] opacity-75 shrink-0">{s.num}</span>
                <span>{s.label}</span>
              </a>
            ))}
          </nav>
        </details>
      </div>

      {/* Desktop Sticky Rail (>= 1280px) */}
      <aside className="hidden xl:block w-64 shrink-0 sticky top-6 self-start max-h-[calc(100vh-4rem)] overflow-y-auto touch-scroll-y pr-2">
        <div className="cream-card p-4">
          <p className="text-[11px] font-mono uppercase tracking-wider text-stone-400 font-semibold mb-3 px-2">
            Contents
          </p>
          <nav className="flex flex-col gap-1 text-xs">
            {sections.map((s) => (
              <a
                key={s.id}
                href={`#${s.id}`}
                className={`px-2.5 py-2 rounded-md transition-colors flex items-center gap-2 ${
                  active === s.id
                    ? 'bg-stone-900 text-stone-100 font-semibold shadow-xs'
                    : 'text-stone-600 hover:bg-stone-100 hover:text-stone-900'
                }`}
              >
                <span className="font-mono text-[10px] opacity-60 shrink-0">{s.num}</span>
                <span className="truncate">{s.label}</span>
              </a>
            ))}
          </nav>
        </div>
      </aside>
    </>
  );
}
