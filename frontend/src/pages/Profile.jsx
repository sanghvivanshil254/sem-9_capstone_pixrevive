import { useEffect, useState } from "react";
import {
  Check,
  CircleUserRound,
  LockKeyhole,
  Mail,
  Save,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import { AppShell, Eyebrow } from "../components/AccountLayout";

export default function Profile() {
  const [form, setForm] = useState({ name: "", email: "" });
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("pixrevive-user") || "{}");
    setForm({ name: user.name || "", email: user.email || "" });
  }, []);
  function update(key, value) {
    setForm((old) => ({ ...old, [key]: value }));
    setSaved(false);
  }
  function submit(e) {
    e.preventDefault();
    const user = JSON.parse(localStorage.getItem("pixrevive-user") || "{}");
    localStorage.setItem(
      "pixrevive-user",
      JSON.stringify({ ...user, ...form }),
    );
    setSaved(true);
  }
  return (
    <AppShell active="profile">
      <main className="pr-app-content">
        <div className="pr-page-heading">
          <div>
            <Eyebrow>YOUR CORNER OF THE ARCHIVE</Eyebrow>
            <h1>
              Personal <em>details.</em>
            </h1>
            <p>Keep your account details up to date.</p>
          </div>
        </div>
        <div className="pr-profile-layout">
          <aside className="pr-profile-summary">
            <div className="pr-profile-avatar">
              <CircleUserRound size={43} />
            </div>
            <span className="pr-card-label">MEMORY KEEPER</span>
            <h2>{form.name || "Your name"}</h2>
            <p>{form.email || "Add your email address"}</p>
            <div className="pr-profile-rule" />
            <div className="pr-profile-security">
              <ShieldCheck size={17} />
              <div>
                <strong>Private by design</strong>
                <span>Your profile is stored in this browser demo.</span>
              </div>
            </div>
          </aside>
          <section className="pr-profile-form-card">
            <div className="pr-profile-card-heading">
              <div>
                <Eyebrow>ACCOUNT INFORMATION</Eyebrow>
                <h2>About you</h2>
              </div>
              <span className="pr-profile-heading-icon">
                <UserRound size={19} />
              </span>
            </div>
            <form className="pr-profile-form" onSubmit={submit}>
              <label className="pr-field">
                <span>Full name</span>
                <div className="pr-input-wrap">
                  <UserRound size={17} />
                  <input
                    required
                    value={form.name}
                    onChange={(e) => update("name", e.target.value)}
                    placeholder="Your name"
                    autoComplete="name"
                  />
                </div>
              </label>
              <label className="pr-field">
                <span>Email address</span>
                <div className="pr-input-wrap">
                  <Mail size={17} />
                  <input
                    required
                    type="email"
                    value={form.email}
                    onChange={(e) => update("email", e.target.value)}
                    placeholder="you@example.com"
                    autoComplete="email"
                  />
                </div>
              </label>
              <div className="pr-profile-info">
                <LockKeyhole size={16} />
                <p>
                  Password and account security are demo placeholders. Connect
                  an authentication backend before using real accounts.
                </p>
              </div>
              <div className="pr-profile-form-bottom">
                {saved && (
                  <span className="pr-saved-message">
                    <Check size={15} /> Changes saved
                  </span>
                )}
                <button className="pr-button pr-button-dark" type="submit">
                  <Save size={16} /> Save changes
                </button>
              </div>
            </form>
          </section>
        </div>
      </main>
    </AppShell>
  );
}
