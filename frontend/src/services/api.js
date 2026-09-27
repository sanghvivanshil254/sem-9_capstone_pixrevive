// Central API client config (plan 5.5 / A-6).
//
// Every backend URL in the app must derive from these two constants. The
// origin is overridable per environment via VITE_API_BASE (see .env.example):
//   VITE_API_BASE=http://localhost:8000        (default)
//   VITE_API_BASE=https://pixrevive.example.com
// The trailing /api is part of the backend route namespace, not the origin.

const RAW_ORIGIN = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const BACKEND_ORIGIN = RAW_ORIGIN.replace(/\/+$/, '');
export const API_BASE = `${BACKEND_ORIGIN}/api`;

/** Full URL for an API route, e.g. apiUrl('/presets') → {origin}/api/presets */
export function apiUrl(path) {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

/** Full URL for a backend origin path, e.g. backendUrl(pair.bw_url) */
export function backendUrl(path) {
  return `${BACKEND_ORIGIN}${path?.startsWith('/') ? path : `/${path}`}`;
}
