import { useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  Check,
  Clock3,
  FileImage,
  ImagePlus,
  Search,
  Sparkles,
  Trash2,
  UploadCloud,
  WandSparkles,
} from "lucide-react";
import { AppShell, Eyebrow } from "../components/AccountLayout";

export default function Dashboard() {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState("");
  function addFiles(list) {
    const next = Array.from(list || [])
      .filter((f) => f.type.startsWith("image/"))
      .map((file) => ({
        id: `${file.name}-${file.lastModified}`,
        name: file.name,
        size: (file.size / 1024 / 1024).toFixed(2),
        url: URL.createObjectURL(file),
        date: new Date().toLocaleDateString(),
      }));
    setFiles((old) => [...next, ...old]);
    setNotice(
      next.length
        ? `${next.length} photo${next.length > 1 ? "s" : ""} added to this session.`
        : "Please choose an image file.",
    );
  }
  function remove(id) {
    setFiles((old) => old.filter((f) => f.id !== id));
  }
  const visible = files.filter((f) =>
    f.name.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <AppShell active="dashboard">
      <main className="pr-app-content">
        <div className="pr-page-heading">
          <div>
            <Eyebrow>YOUR PERSONAL COLLECTION</Eyebrow>
            <h1>
              My <em>restorations.</em>
            </h1>
            <p>
              A quiet place for the photographs you’re bringing back to life.
            </p>
          </div>
          <span className="pr-archive-count">
            <strong>{files.length.toString().padStart(2, "0")}</strong>
            <small>THIS SESSION</small>
          </span>
        </div>
        <section className="pr-upload-panel" id="new-restoration">
          <div className="pr-upload-copy">
            <span className="pr-upload-icon">
              <UploadCloud size={23} />
            </span>
            <Eyebrow>A NEW MEMORY</Eyebrow>
            <h2>
              Give an old photo
              <br />
              <em>a new beginning.</em>
            </h2>
            <p>
              Choose a photograph from your device. JPG, PNG and WEBP images
              work best.
            </p>
            <button
              className="pr-button pr-button-dark"
              onClick={() => inputRef.current?.click()}
            >
              <ImagePlus size={17} /> Choose a photograph{" "}
              <ArrowRight size={16} />
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/*"
              multiple
              hidden
              onChange={(e) => addFiles(e.target.files)}
            />
            <span className="pr-upload-limit">
              Images are previewed locally in this demo.
            </span>
          </div>
          <button
            className="pr-dropzone"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              addFiles(e.dataTransfer.files);
            }}
          >
            <span className="pr-drop-icon">
              <FileImage size={28} />
            </span>
            <strong>Drop your photos here</strong>
            <span>or browse your computer</span>
            <span className="pr-drop-types">PNG · JPG · WEBP</span>
          </button>
        </section>
        {notice && (
          <p className="pr-notice" role="status">
            <Check size={15} />
            {notice}
          </p>
        )}
        <section className="pr-archive-section">
          <div className="pr-archive-header">
            <div>
              <Eyebrow>THE MEMORY SHELF</Eyebrow>
              <h2>Your photo shelf</h2>
            </div>
            <label className="pr-search">
              <Search size={16} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Find a photograph"
              />
            </label>
          </div>
          {visible.length === 0 ? (
            <div className="pr-empty-state">
              <div className="pr-empty-illustration">
                <ImagePlus size={29} />
                <Sparkles size={16} />
              </div>
              <h3>
                {files.length
                  ? "No matching photographs"
                  : "Your next memory starts here."}
              </h3>
              <p>
                {files.length
                  ? "Try a different name or clear your search."
                  : "Upload a photo above to create a local preview in your collection."}
              </p>
              <button
                className="pr-text-link"
                onClick={() => inputRef.current?.click()}
              >
                Choose your first photo <ArrowRight size={16} />
              </button>
            </div>
          ) : (
            <div className="pr-photo-grid">
              {visible.map((file) => (
                <article className="pr-photo-card" key={file.id}>
                  <div className="pr-photo-thumb">
                    <img src={file.url} alt={file.name} />
                    <span className="pr-local-badge">
                      <Clock3 size={12} /> Local preview
                    </span>
                  </div>
                  <div className="pr-photo-info">
                    <h3 title={file.name}>{file.name}</h3>
                    <p>
                      {file.size} MB · Added {file.date}
                    </p>
                    <div className="pr-photo-actions">
                      <a href={file.url} download={file.name}>
                        <ArrowDownToLine size={14} /> Save original
                      </a>
                      <button
                        onClick={() => remove(file.id)}
                        aria-label={`Remove ${file.name}`}
                      >
                        <Trash2 size={14} /> Remove
                      </button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
        <div className="pr-demo-banner">
          <WandSparkles size={18} />
          <p>
            <strong>Restoration engine connection</strong>
            <br />
            These pages are a new UI layer. Actual AI restoration and
            cloud-saved history are not connected here yet.
          </p>
        </div>
      </main>
    </AppShell>
  );
}
