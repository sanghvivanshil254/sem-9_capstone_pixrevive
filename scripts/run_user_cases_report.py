"""run_user_cases_report.py — run user case images through the full pipeline.

Runs the adopted user benchmark cases (case8-case12) through
RestorationPipeline.restore() — the exact path /api/process-stream uses —
and collects for each case:
  <case>_restored.png   full restored output (4x, 2K-normalized)
  <case>_report.json    the T-2 Restoration Report (honesty layer)
  gallery.html          before/after gallery + report summaries

Outputs land in git-ignored test_artifacts/user_cases/. The gallery merges
report JSONs already on disk, so batched runs (one case per invocation)
still produce a complete gallery.

Usage:
  "C:/Program Files/Python312/python.exe" scripts/run_user_cases_report.py [case ...]
  (no args = all case8..case12)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from ai.pipeline.restoration_pipeline import RestorationPipeline  # noqa: E402
from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAMAGED_DIR = os.path.join(REPO_ROOT, "tests", "restoration_dataset", "damaged")
OUT_DIR = os.path.join(REPO_ROOT, "test_artifacts", "user_cases")

CASES = [
    "case8_speckle_lena_margins.png",
    "case9_teared_sepia.jpg",
    "case10_spiderweb_lips.jpg",
    "case11_minor_creases.jpg",
    "case12_foxing_spots.png",
]


def run_case(pipeline: RestorationPipeline, case_file: str) -> dict:
    path = os.path.join(DAMAGED_DIR, case_file)
    name = os.path.splitext(case_file)[0]
    bgr = cv2.imread(path)
    if bgr is None:
        return {"case": name, "error": "unreadable image"}
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[{name}] input {original.shape[1]}x{original.shape[0]}, restoring...", flush=True)

    started = time.perf_counter()
    restored = pipeline.restore(image_input=original.copy())
    elapsed = time.perf_counter() - started

    ctx: PipelineContext = pipeline.last_context
    report = ctx.image_metadata.get("restoration_report")

    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_restored.png"),
                cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    with open(os.path.join(OUT_DIR, f"{name}_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    # Before/after side-by-side strip (input resized to output height).
    h, w = restored.shape[:2]
    orig_resized = (
        cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
        if original.shape[:2] != (h, w) else original
    )
    strip = cv2.cvtColor(cv2.hconcat([orig_resized, restored]), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_side_by_side.png"), strip)

    print(
        f"[{name}] done in {elapsed:.1f}s -> {restored.shape[1]}x{restored.shape[0]}  "
        f"recovered={len((report or {}).get('recovered', []))} "
        f"reconstructed={len((report or {}).get('reconstructed', []))} "
        f"flagged={len((report or {}).get('flagged', []))}",
        flush=True,
    )
    return {"case": name, "elapsed_s": round(elapsed, 1), "report": report}


def _embed_preview(png_path: str, max_w: int = 900, quality: int = 82) -> str:
    """Return a data-URI JPEG preview of a PNG, or a relative path on failure.

    The sandboxed HTML preview serves only the registered file, so a gallery
    meant to be previewed must carry its images inline; full-size PNGs stay
    on disk next to the gallery for direct viewing.
    """
    try:
        img = cv2.imread(png_path)
        if img is None:
            return png_path
        h, w = img.shape[:2]
        if w > max_w:
            nh = max(1, int(h * max_w / w))
            img = cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return png_path
        import base64
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return png_path


def build_gallery(results: list) -> None:
    """Write gallery.html from this run's results PLUS report JSONs already on
    disk, so batched runs (one case per invocation) still get a full gallery."""
    by_name = {os.path.splitext(r["case"])[0]: r for r in results}
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith("_report.json"):
            name = fn[: -len("_report.json")]
            if name not in by_name:
                try:
                    with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as fp:
                        by_name[name] = {
                            "case": name,
                            "elapsed_s": "earlier run",
                            "report": json.load(fp),
                        }
                except (OSError, json.JSONDecodeError):
                    continue

    cards = []
    for name in sorted(by_name):
        res = by_name[name]
        if "error" in res:
            cards.append(
                f"<section class='card'><h2>{name}</h2>"
                f"<p class='err'>FAILED: {res['error']}</p></section>"
            )
            continue
        rep = res.get("report") or {}

        def ul(key, _rep=rep):
            items = _rep.get(key, [])
            if not items:
                return "<p class='muted'>—</p>"
            return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

        cards.append(f"""
<section class='card'>
  <h2>{name} <span class='time'>({res['elapsed_s']}s)</span></h2>
  <div class='imgs'>
    <figure><img src='{_embed_preview(os.path.join(OUT_DIR, name + '_side_by_side.png'))}'><figcaption>Before &rarr; After</figcaption></figure>
    <figure><img src='{_embed_preview(os.path.join(OUT_DIR, name + '_restored.png'))}'><figcaption>Restored output (preview — full PNG on disk)</figcaption></figure>
  </div>
  <h3 class='rec'>Repaired (original pixels recovered)</h3>{ul('recovered')}
  <h3 class='recon'>Reconstructed by AI (plausible, not original)</h3>{ul('reconstructed')}
  <h3 class='flag'>Notes</h3>{ul('flagged')}
</section>""")

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>PixRevive — user case outputs</title>
<style>
body{{font-family:system-ui,'Segoe UI',sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#faf9f7;color:#333}}
h1{{font-size:1.4rem}} .muted{{color:#999}}
.card{{background:#fff;border:1px solid #e2e0dc;border-radius:10px;padding:1.2rem 1.4rem;margin:1.2rem 0}}
.imgs{{display:flex;gap:1rem;flex-wrap:wrap}} figure{{margin:0;max-width:48%}} img{{width:100%;border-radius:6px;border:1px solid #eee}}
figcaption{{font-size:.8rem;color:#777;margin-top:.3rem}}
h3{{font-size:.9rem;margin:.8rem 0 .2rem}} h3.rec{{color:#047857}} h3.recon{{color:#4338ca}} h3.flag{{color:#b45309}}
ul{{margin:.2rem 0;padding-left:1.2rem}} li{{font-size:.85rem}}
.time{{font-weight:400;color:#999;font-size:.8rem}} .err{{color:#b91c1c}}
</style></head><body>
<h1>PixRevive — user case outputs ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})</h1>
{''.join(cards)}
</body></html>"""
    gallery = os.path.join(OUT_DIR, "gallery.html")
    with open(gallery, "w", encoding="utf-8") as fp:
        fp.write(html)
    print(f"[OK] gallery: {gallery}", flush=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if "--gallery-only" in sys.argv[1:]:
        build_gallery([])  # rebuild from report JSONs already on disk
        return
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    case_files = [f for f in CASES if not wanted or any(f.startswith(w) for w in wanted)]
    if not case_files:
        print("No matching cases.")
        sys.exit(1)

    pipeline = RestorationPipeline()
    pipeline._models_ready.wait(timeout=600)
    pipeline._colorizer_ready.wait(timeout=600)

    results = []
    for case_file in case_files:
        try:
            results.append(run_case(pipeline, case_file))
        except Exception as exc:
            print(f"[{case_file}] FAILED: {exc}", flush=True)
            results.append({"case": os.path.splitext(case_file)[0], "error": str(exc)})

    build_gallery(results)


if __name__ == "__main__":
    main()
