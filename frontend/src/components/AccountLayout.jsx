import { Link, NavLink, useNavigate } from 'react-router-dom';
import { Aperture, ArrowRight, CircleUserRound, LayoutDashboard, LogOut, Menu, Sparkles, X } from 'lucide-react';
import { useState } from 'react';

export function Brand({ light = false }) {
  return <Link to="/" className={`pr-brand ${light ? 'pr-brand-light' : ''}`} aria-label="PixRevive home"><span className="pr-brand-mark"><Aperture size={21} strokeWidth={1.8} /></span><span>pixrevive<span className="pr-brand-dot">.</span></span></Link>;
}

export function PublicNav() {
  return <header className="pr-nav"><div className="pr-nav-inner"><Brand /><nav className="pr-nav-links"><a href="/#how-it-works">How it works</a><a href="/#features">Features</a><a href="/#stories">Our gallery</a></nav><div className="pr-nav-actions"><Link className="pr-nav-login" to="/login">Log in</Link><Link className="pr-button pr-button-dark pr-button-small" to="/register">Restore a photo <ArrowRight size={15}/></Link></div></div></header>;
}

export function AppNav({ active = 'home' }) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  function logout() { localStorage.removeItem('pixrevive-user'); navigate('/'); }
  return <header className="pr-nav pr-nav-app"><div className="pr-nav-inner"><Brand /><nav className={`pr-nav-links ${open ? 'pr-nav-open' : ''}`}><NavLink end to="/app" onClick={() => setOpen(false)}>Home</NavLink><NavLink to="/dashboard" onClick={() => setOpen(false)}>My restorations</NavLink><NavLink to="/profile" onClick={() => setOpen(false)}>Profile</NavLink></nav><div className="pr-nav-actions"><Link className="pr-avatar-mini" to="/profile"><CircleUserRound size={19}/></Link><button className="pr-icon-button pr-mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle menu">{open ? <X/> : <Menu/>}</button><button className="pr-nav-login pr-logout" onClick={logout}><LogOut size={15}/> Sign out</button></div></div></header>;
}

export function PageFooter() { return <footer className="pr-footer"><Brand/><span>Preserve the past. Bring memories back to life.</span><span>© {new Date().getFullYear()} PixRevive</span></footer>; }

export function AppShell({ children, active }) { return <div className="pr-app"><AppNav active={active}/>{children}<PageFooter/></div>; }

export function Eyebrow({ children }) { return <div className="pr-eyebrow"><span/> {children}</div>; }
