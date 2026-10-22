import { Link } from "react-router-dom";
import {
  ArrowRight,
  Clock3,
  ImagePlus,
  Sparkles,
  WandSparkles,
} from "lucide-react";
import { AppShell, Eyebrow } from "../components/AccountLayout";

export default function AppHome() {
  const user = JSON.parse(localStorage.getItem("pixrevive-user") || "{}");
  return (
    <AppShell active="home">
      <main className="pr-app-content">
        <section className="pr-welcome-banner">
          <div>
            <Eyebrow>YOUR MEMORY ATELIER</Eyebrow>
            <h1>
              Welcome back,{" "}
              <em>{user.name?.split(" ")[0] || "Memory Keeper"}.</em>
            </h1>
            <p>There are more memories waiting to be brought back to life.</p>
            <div className="pr-hero-actions">
              <Link className="pr-button pr-button-dark" to="/dashboard">
                <ImagePlus size={17} /> Open my archive <ArrowRight size={16} />
              </Link>
              <span className="pr-soft-note">
                <Sparkles size={15} /> One photo can bring it all back.
              </span>
            </div>
          </div>
          <div className="pr-welcome-art">
            <img
              src="https://images.unsplash.com/photo-1511895426328-dc8714191300?auto=format&fit=crop&w=900&q=85"
              alt="A family memory"
            />
            <span>MEMORY NO. 001</span>
          </div>
        </section>
        <section className="pr-app-section">
          <div className="pr-inline-heading">
            <div>
              <Eyebrow>A PLACE FOR YOUR STORIES</Eyebrow>
              <h2>What would you like to do?</h2>
            </div>
          </div>
          <div className="pr-action-grid">
            <Link to="/dashboard" className="pr-action-card">
              <span className="pr-action-icon">
                <ImagePlus />
              </span>
              <span className="pr-card-label">YOUR ARCHIVE</span>
              <h3>My restorations</h3>
              <p>
                Return to your saved work and keep your restored photos
                together.
              </p>
              <span className="pr-card-link">
                View archive <ArrowRight size={16} />
              </span>
            </Link>
            <Link to="/dashboard#new-restoration" className="pr-action-card">
              <span className="pr-action-icon pr-action-icon-warm">
                <WandSparkles />
              </span>
              <span className="pr-card-label">A NEW CHAPTER</span>
              <h3>Restore a photograph</h3>
              <p>
                Start with a faded portrait, a scratched print or a memory in
                black and white.
              </p>
              <span className="pr-card-link">
                Start restoring <ArrowRight size={16} />
              </span>
            </Link>
            <Link to="/profile" className="pr-action-card">
              <span className="pr-action-icon pr-action-icon-blue">
                <Clock3 />
              </span>
              <span className="pr-card-label">YOUR DETAILS</span>
              <h3>Personal archive</h3>
              <p>
                Manage your profile and the details connected to your PixRevive
                account.
              </p>
              <span className="pr-card-link">
                View profile <ArrowRight size={16} />
              </span>
            </Link>
          </div>
        </section>
        <section className="pr-quote-strip">
          <span className="pr-quote-glyph">“</span>
          <p>We do not remember days, we remember moments.</p>
          <span>— CESARE PAVESE</span>
        </section>
      </main>
    </AppShell>
  );
}
