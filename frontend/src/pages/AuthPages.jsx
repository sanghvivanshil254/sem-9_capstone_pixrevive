import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Aperture,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  UserRound,
} from "lucide-react";
import { Brand } from "../components/AccountLayout";

function AuthFrame({ children, title, subtitle, mode }) {
  return (
    <main className="pr-auth-page">
      <div className="pr-auth-aside">
        <Brand />
        <div className="pr-auth-quote">
          <span className="pr-auth-quote-mark">“</span>
          <h2>Some things are too precious to fade away.</h2>
          <p>Keep the moments that made you, close to you.</p>
          <div className="pr-auth-aside-foot">
            <span>PIXREVIVE · MEMORY ARCHIVE</span>
            <span>01 / 03</span>
          </div>
        </div>
        <img
          className="pr-auth-bg"
          src="https://images.unsplash.com/photo-1511895426328-dc8714191300?auto=format&fit=crop&w=1200&q=85"
          alt="Family memories"
        />
      </div>
      <section className="pr-auth-main">
        <Link className="pr-back-link" to="/">
          <ArrowLeft size={16} /> Back to home
        </Link>
        <div className="pr-auth-form-wrap">
          <div className="pr-auth-mobile-brand">
            <Brand />
          </div>
          <div className="pr-auth-kicker">
            {mode === "login" ? "WELCOME BACK" : "START YOUR ARCHIVE"}
          </div>
          <h1>{title}</h1>
          <p className="pr-auth-subtitle">{subtitle}</p>
          {children}
          <p className="pr-auth-switch">
            {mode === "login"
              ? "New to PixRevive?"
              : "Already have an account?"}{" "}
            <Link to={mode === "login" ? "/register" : "/login"}>
              {mode === "login" ? "Create an account" : "Log in"}
            </Link>
          </p>
          <div className="pr-auth-privacy">
            <LockKeyhole size={14} /> Your photos are personal. Your privacy
            matters.
          </div>
        </div>
        <footer className="pr-auth-footer">
          © {new Date().getFullYear()} PixRevive{" "}
          <span>Made for memories that matter.</span>
        </footer>
      </section>
    </main>
  );
}

function Field({
  label,
  type = "text",
  value,
  onChange,
  placeholder,
  icon: Icon,
  autoComplete,
  required = true,
}) {
  const [show, setShow] = useState(false);
  const actualType = type === "password" && show ? "text" : type;
  return (
    <label className="pr-field">
      <span>{label}</span>
      <div className="pr-input-wrap">
        {Icon && <Icon size={17} />}
        <input
          required={required}
          type={actualType}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          autoComplete={autoComplete}
        />
        {type === "password" && (
          <button
            type="button"
            className="pr-password-toggle"
            onClick={() => setShow(!show)}
            aria-label={show ? "Hide password" : "Show password"}
          >
            {show ? <EyeOff size={17} /> : <Eye size={17} />}
          </button>
        )}
      </div>
    </label>
  );
}

export function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  function submit(e) {
    e.preventDefault();
    const stored = JSON.parse(localStorage.getItem("pixrevive-user") || "null");
    if (stored && stored.email.toLowerCase() !== email.toLowerCase()) {
      setError("No demo account matches this email. Try registering first.");
      return;
    }
    const user = stored || {
      name: email.split("@")[0] || "Memory Keeper",
      email,
    };
    localStorage.setItem("pixrevive-user", JSON.stringify(user));
    navigate("/app");
  }
  return (
    <AuthFrame
      mode="login"
      title="Welcome home."
      subtitle="Sign in to return to the memories you’re restoring."
    >
      <form className="pr-auth-form" onSubmit={submit}>
        <Field
          label="Email address"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          icon={Mail}
          autoComplete="email"
        />
        <Field
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter your password"
          icon={LockKeyhole}
          autoComplete="current-password"
        />
        <div className="pr-form-meta">
          <label className="pr-check-label">
            <input type="checkbox" /> Keep me signed in
          </label>
          <span className="pr-demo-note">Demo sign-in</span>
        </div>
        {error && (
          <p className="pr-form-error" role="alert">
            {error}
          </p>
        )}
        <button
          className="pr-button pr-button-dark pr-auth-submit"
          type="submit"
        >
          Log in to PixRevive <ArrowRight size={17} />
        </button>
        <p className="pr-demo-help">
          This is a front-end demo. Authentication is stored locally in this
          browser.
        </p>
      </form>
    </AuthFrame>
  );
}

export function Register() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  function submit(e) {
    e.preventDefault();
    if (password.length < 6) {
      setError("Use at least 6 characters for your password.");
      return;
    }
    if (password !== confirm) {
      setError("Your passwords do not match.");
      return;
    }
    const user = {
      name: name.trim(),
      email: email.trim(),
      joinedAt: new Date().toISOString(),
    };
    localStorage.setItem("pixrevive-user", JSON.stringify(user));
    navigate("/app");
  }
  return (
    <AuthFrame
      mode="register"
      title="Make room for memories."
      subtitle="Create your free archive and give your old photographs a new chapter."
    >
      <form className="pr-auth-form" onSubmit={submit}>
        <Field
          label="Your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="How should we call you?"
          icon={UserRound}
          autoComplete="name"
        />
        <Field
          label="Email address"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          icon={Mail}
          autoComplete="email"
        />
        <Field
          label="Create password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="At least 6 characters"
          icon={LockKeyhole}
          autoComplete="new-password"
        />
        <Field
          label="Confirm password"
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          placeholder="Enter password again"
          icon={LockKeyhole}
          autoComplete="new-password"
        />
        {error && (
          <p className="pr-form-error" role="alert">
            {error}
          </p>
        )}
        <button
          className="pr-button pr-button-dark pr-auth-submit"
          type="submit"
        >
          Create my archive <ArrowRight size={17} />
        </button>
        <p className="pr-demo-help">
          Demo only: account details stay in local browser storage. No backend
          account is created.
        </p>
      </form>
    </AuthFrame>
  );
}
