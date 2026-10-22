import { Link } from "react-router-dom";
import {
  ArrowDown,
  ArrowRight,
  Check,
  CirclePlay,
  ScanFace,
  ShieldCheck,
  Sparkles,
  WandSparkles,
} from "lucide-react";
import {
  Brand,
  Eyebrow,
  PageFooter,
  PublicNav,
} from "../components/AccountLayout";

const examples = [
  {
    title: "A day by the seaside",
    meta: "Family archive · 1964",
    image:
      "https://images.unsplash.com/photo-1511895426328-dc8714191300?auto=format&fit=crop&w=900&q=85",
  },
  {
    title: "Sunday in the garden",
    meta: "Personal collection · 1978",
    image:
      "https://images.unsplash.com/photo-1511632765486-a01980e01a18?auto=format&fit=crop&w=900&q=85",
  },
  {
    title: "A portrait to remember",
    meta: "Found photograph · 1949",
    image:
      "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?auto=format&fit=crop&w=900&q=85",
  },
];
export default function PublicHome() {
  return (
    <div className="pr-site">
      <PublicNav />
      <main>
        <section className="pr-hero">
          <div className="pr-hero-copy">
            <Eyebrow>THE ART OF REMEMBERING</Eyebrow>
            <h1>
              Some memories
              <br />
              deserve <em>another life.</em>
            </h1>
            <p className="pr-hero-lede">
              Bring your most treasured photographs back into focus. Thoughtful
              AI restoration, made for the moments that matter most.
            </p>
            <div className="pr-hero-actions">
              <Link to="/register" className="pr-button pr-button-dark">
                Restore your first photo <ArrowRight size={17} />
              </Link>
              <a className="pr-text-link" href="#how-it-works">
                <CirclePlay size={18} /> See how it works
              </a>
            </div>
            <div className="pr-proof">
              <div className="pr-proof-avatars">
                <span>J</span>
                <span>M</span>
                <span>A</span>
              </div>
              <div>
                <strong>Made for your memories</strong>
                <small>Private by design · You stay in control</small>
              </div>
            </div>
          </div>
          <div className="pr-hero-art">
            <div className="pr-photo-back">
              <img
                src="https://images.unsplash.com/photo-1511895426328-dc8714191300?auto=format&fit=crop&w=1000&q=90"
                alt="Family spending time together"
              />
            </div>
            <div className="pr-photo-front">
              <img
                src="https://images.unsplash.com/photo-1511632765486-a01980e01a18?auto=format&fit=crop&w=700&q=90"
                alt="A cherished family memory"
              />
              <span className="pr-photo-caption">
                FIG. 01 — A MEMORY, RESTORED
              </span>
            </div>
            <div className="pr-stamp">
              <Sparkles size={18} />
              <span>
                Made with
                <br />
                <b>care & AI</b>
              </span>
            </div>
            <div className="pr-art-note">
              Every crease tells a story.
              <br />
              <em>Let it keep telling yours.</em>
            </div>
          </div>
        </section>
        <div className="pr-trust-strip">
          <span>FOR EVERY KIND OF MEMORY</span>
          <i /> <span>Faded prints</span>
          <i /> <span>Old family portraits</span>
          <i /> <span>Damaged photographs</span>
          <i /> <span>Black & white moments</span>
        </div>
        <section className="pr-section pr-process" id="how-it-works">
          <div className="pr-section-heading">
            <Eyebrow>A GENTLE PROCESS</Eyebrow>
            <h2>
              From fragile to <em>forever.</em>
            </h2>
            <p>
              Three simple steps between you and a memory that feels close
              again.
            </p>
          </div>
          <div className="pr-steps">
            <article>
              <span className="pr-step-number">01</span>
              <div className="pr-step-icon">
                <ScanFace />
              </div>
              <h3>Bring your photo</h3>
              <p>
                Upload a scan or a picture of your old photograph. Every era is
                welcome.
              </p>
            </article>
            <article>
              <span className="pr-step-number">02</span>
              <div className="pr-step-icon">
                <WandSparkles />
              </div>
              <h3>Let care do its work</h3>
              <p>
                Our restoration tools reduce damage, recover detail and bring
                color back thoughtfully.
              </p>
            </article>
            <article>
              <span className="pr-step-number">03</span>
              <div className="pr-step-icon">
                <ShieldCheck />
              </div>
              <h3>Keep it close</h3>
              <p>
                Preview your restored image, save your result and revisit it
                whenever you like.
              </p>
            </article>
          </div>
        </section>
        <section className="pr-gallery-section" id="stories">
          <div className="pr-section-heading pr-gallery-heading">
            <Eyebrow>THE MEMORY ARCHIVE</Eyebrow>
            <h2>
              Every photograph holds
              <br />
              <em>a little piece of us.</em>
            </h2>
            <p>A glimpse of the moments we help bring back into view.</p>
          </div>
          <div className="pr-gallery-grid">
            {examples.map((item, i) => (
              <article
                className={`pr-gallery-card pr-gallery-card-${i + 1}`}
                key={item.title}
              >
                <img src={item.image} alt={item.title} />
                <div className="pr-gallery-overlay">
                  <span>{item.meta}</span>
                  <h3>{item.title}</h3>
                </div>
              </article>
            ))}
          </div>
        </section>
        <section className="pr-feature-band" id="features">
          <div className="pr-feature-copy">
            <Eyebrow>TECHNOLOGY, WITH TENDERNESS</Eyebrow>
            <h2>
              Not just an image.
              <br />
              <em>A piece of your story.</em>
            </h2>
            <p>
              PixRevive combines modern restoration technology with a simple,
              considered experience. Your memories deserve more than a filter.
            </p>
            <ul>
              <li>
                <Check size={16} /> Repair scratches and visual damage
              </li>
              <li>
                <Check size={16} /> Improve clarity and faded details
              </li>
              <li>
                <Check size={16} /> Colorize selected black-and-white photos
              </li>
              <li>
                <Check size={16} /> Keep your personal archive in one place
              </li>
            </ul>
            <Link className="pr-text-link" to="/register">
              Begin your archive <ArrowRight size={16} />
            </Link>
          </div>
          <div className="pr-feature-visual">
            <div className="pr-feature-image">
              <img
                src="https://images.unsplash.com/photo-1511895426328-dc8714191300?auto=format&fit=crop&w=1000&q=90"
                alt="A timeless family photograph"
              />
            </div>
            <div className="pr-feature-label">
              <span>THE PIXREVIVE PROMISE</span>
              <strong>Every memory matters.</strong>
              <ArrowDown size={18} />
            </div>
          </div>
        </section>
        <section className="pr-cta">
          <div>
            <Eyebrow>YOUR STORY IS WAITING</Eyebrow>
            <h2>
              Let's bring a little
              <br />
              <em>yesterday back.</em>
            </h2>
            <p>Start with one photograph. See where the memories take you.</p>
            <Link className="pr-button pr-button-light" to="/register">
              Start restoring <ArrowRight size={17} />
            </Link>
          </div>
          <div className="pr-cta-flower">✳</div>
        </section>
      </main>
      <PageFooter />
    </div>
  );
}
