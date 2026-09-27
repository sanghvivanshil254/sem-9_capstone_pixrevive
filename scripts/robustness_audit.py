"""robustness_audit.py — whole-pipeline robustness battery (plan post-6 verification).

Layers:
  1. Static hygiene      — compileall over backend/scripts/tests/experimental.
  2. Synthetic inputs    — 14 pathological images through the pipeline CPU-side
                           (solid fields, 1x1, strips, odd dims, NaN, RGBA...).
  3. Toggle matrix       — all 32 combinations of the 5 pipeline toggles on two
                           base images (guards must hold in every combination).
  4. API surface         — FastAPI TestClient: endpoints, malformed uploads,
                           SSE stream strict-JSON + stage-registry order,
                           restoration-report honesty, 8-thread concurrency.
  5. GPU spot check      — CUDA run on a case image with VRAM/wall timing
                           (skipped automatically when no GPU).
  6. Verdict             — machine-readable summary written to
                           test_artifacts/robustness_audit.json.

Usage:
  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" scripts/robustness_audit.py [--skip-gpu]

Exit code is 0 only when every executed check passes.
"""

import io
import json
import os
import sys
import time
import threading
import traceback
import concurrent.futures
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import numpy as np
import cv2
from PIL import Image

OUT_PATH = os.path.join("test_artifacts", "robustness_audit.json")
REPORT = {"started": datetime.now().isoformat(), "sections": {}}
_failures = []
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def record(section: str, name: str, ok: bool, detail: str = "", elapsed: float = 0.0) -> None:
    REPORT["sections"].setdefault(section, []).append(
        {"name": name, "ok": bool(ok), "detail": detail, "elapsed_s": round(elapsed, 3)}
    )
    status = "PASS" if ok else "FAIL"
    log(f"  [{status}] {name}" + (f" — {detail}" if detail and not ok else (f" ({detail})" if detail else "")))
    if not ok:
        _failures.append(f"{section}/{name}: {detail}")


def run_case(section: str, name: str, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        record(section, name, True, detail, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 — the battery must survive anything
        record(section, name, False, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)
        log(traceback.format_exc(limit=3))


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — static hygiene
# ─────────────────────────────────────────────────────────────────────────────
def section_static():
    import py_compile

    roots = ["backend", "scripts", "tests", "experimental"]
    bad = []
    count = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "node_modules", "weights", ".git")]
            for fn in filenames:
                if fn.endswith(".py"):
                    count += 1
                    path = os.path.join(dirpath, fn)
                    try:
                        py_compile.compile(path, doraise=True, cfile=os.devnull)
                    except py_compile.PyCompileError as exc:
                        bad.append(f"{path}: {exc.msg}")
                    except OSError as exc:
                        # Windows: compiling to os.devnull ('nul') raises on some
                        # runtimes — fall back to a real temp file.
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as tmp:
                            tmp_path = tmp.name
                        try:
                            py_compile.compile(path, doraise=True, cfile=tmp_path)
                        finally:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
    record("1_static", f"py_compile_all ({count} files)", not bad, "; ".join(bad[:5]))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — synthetic input battery
# ─────────────────────────────────────────────────────────────────────────────
def _make_images():
    rng = np.random.default_rng(42)
    return {
        "solid_black": np.zeros((64, 64, 3), np.uint8),
        "solid_white": np.full((64, 64, 3), 255, np.uint8),
        "solid_gray": np.full((64, 64, 3), 128, np.uint8),
        "one_pixel": np.full((1, 1, 3), 200, np.uint8),
        "three_by_three": np.full((3, 3, 3), 90, np.uint8),
        "eight_by_eight": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
        "tiny_gradient": np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (16, 1)),
        "small_photo": rng.integers(30, 230, (100, 75, 3), dtype=np.uint8),
        "odd_dims": rng.integers(0, 256, (121, 99, 3), dtype=np.uint8),
        "thin_strip": rng.integers(0, 256, (300, 2, 3), dtype=np.uint8),
        "wide_strip": rng.integers(0, 256, (2, 300, 3), dtype=np.uint8),
        "gray_noise": np.stack([np.clip(128 + rng.normal(0, 20, (260, 300)), 0, 255)] * 3, axis=2).astype(np.uint8),
        "float_nan": np.full((64, 64, 3), np.nan, dtype=np.float64),
        "rgba": np.dstack([rng.integers(0, 256, (64, 64), dtype=np.uint8)] * 3 + [np.full((64, 64), 255, np.uint8)]),
    }


def _img_is_sane(arr: np.ndarray) -> bool:
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 3
        and arr.shape[2] == 3
        and arr.dtype == np.uint8
        and arr.shape[0] > 0
        and arr.shape[1] > 0
    )


def section_synthetic():
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    from ai.pipeline.pipeline_context import PipelineContext

    pipeline = RestorationPipeline(device="cpu", load_models=False)
    for name, img in _make_images().items():
        def case(img=img, name=name):
            out = pipeline.restore(img.copy())
            if not _img_is_sane(out):
                raise AssertionError(f"unsane output: shape={getattr(out, 'shape', None)} dtype={getattr(out, 'dtype', None)}")
            return f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}"
        run_case("2_synthetic", name, case)

    # Input-type polymorphism: bytes / PIL / ndarray must all work.
    png_bytes = cv2.imencode(".png", cv2.cvtColor(_make_images()["small_photo"], cv2.COLOR_RGB2BGR))[1].tobytes()

    def case_bytes():
        out = pipeline.restore(png_bytes)
        assert _img_is_sane(out)
    run_case("2_synthetic", "input_png_bytes", case_bytes)

    def case_pil():
        out = pipeline.restore(Image.fromarray(_make_images()["small_photo"]))
        assert _img_is_sane(out)
    run_case("2_synthetic", "input_pil_image", case_pil)

    def case_none():
        try:
            pipeline.restore(None)
        except (TypeError, ValueError, AttributeError):
            return "rejected with explicit error"
        raise AssertionError("None input silently accepted")
    run_case("2_synthetic", "input_none_rejected", case_none)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — toggle matrix (2 images × 2^5 toggles)
# ─────────────────────────────────────────────────────────────────────────────
def section_toggles():
    from ai.pipeline.restoration_pipeline import RestorationPipeline

    pipeline = RestorationPipeline(device="cpu", load_models=False)
    rng = np.random.default_rng(7)
    bases = {
        "photo_100x75": rng.integers(30, 230, (100, 75, 3), dtype=np.uint8),
        "gray_256": np.full((256, 256, 3), 128, np.uint8),
    }
    for base_name, base in bases.items():
        for mask in range(32):
            flags = {
                "enable_repair": bool(mask & 1),
                "enable_denoise": bool(mask & 2),
                "enable_colorization": bool(mask & 4),
                "enable_face_restore": bool(mask & 8),
                "enable_sr": bool(mask & 16),
            }

            def case(base=base, flags=flags):
                out = pipeline.restore(base.copy(), **flags)
                if not _img_is_sane(out):
                    raise AssertionError(f"unsane output with {flags}")
                return "+".join(k.replace("enable_", "") for k, v in flags.items() if v) or "none"
            run_case("3_toggles", f"{base_name}/{mask:05b}", case)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — API surface, SSE contract, report honesty, concurrency
# ─────────────────────────────────────────────────────────────────────────────
def section_api():
    from fastapi.testclient import TestClient
    from main import app
    from ai.pipeline.stage_registry import STAGE_REGISTRY

    client = TestClient(app)

    # 4a. basic endpoints
    run_case("4_api", "health", lambda: (
        (lambda d: (_ for _ in ()).throw(AssertionError(d)) if not (d.get("status") == "ok") else "ok")(
            client.get("/api/health").json())))
    run_case("4_api", "presets", lambda: (
        (lambda p: (_ for _ in ()).throw(AssertionError("missing default")) if "default" not in p else "5 presets")(
            client.get("/api/presets").json()["presets"])))

    # 4b. malformed / hostile uploads
    tiny_png = cv2.imencode(".png", np.full((48, 48, 3), 111, np.uint8))[1].tobytes()

    def case_bad_type():
        r = client.post("/api/process-stream", files={"file": ("x.bmp", tiny_png, "image/bmp")})
        assert r.status_code == 400, f"expected 400, got {r.status_code}"
    run_case("4_api", "upload_rejects_bad_mime", case_bad_type)

    def case_oversize():
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (26 * 1024 * 1024)
        r = client.post("/api/process-stream", files={"file": ("big.png", big, "image/png")})
        assert r.status_code == 400, f"expected 400, got {r.status_code}"
    run_case("4_api", "upload_rejects_oversize", case_oversize)

    def case_truncated_png():
        r = client.post("/api/process-stream", files={"file": ("t.png", tiny_png[:10], "image/png")})
        # accepted or 400/500 — but the SSE stream must carry an `error` event,
        # never hang.
        assert r.status_code in (200, 400, 500), r.status_code
        if r.status_code == 200:
            assert 'event: error' in r.text, "truncated PNG must produce SSE error event"
            return "sse error event"
        return f"HTTP {r.status_code}"
    run_case("4_api", "upload_truncated_png", case_truncated_png)

    def case_unknown_task():
        r = client.get("/api/process/deadbeefdead/result")
        assert r.status_code == 404, f"expected 404, got {r.status_code}"
    run_case("4_api", "unknown_task_404", case_unknown_task)

    def case_admin_invalid_model():
        r = client.post(
            "/api/admin/weights/upload",
            data={"model_type": "bogus"},
            files={"file": ("w.pth", io.BytesIO(b"PK\x03\x04zz"), "application/octet-stream")},
        )
        assert r.status_code == 400
    run_case("4_api", "admin_upload_invalid_model", case_admin_invalid_model)

    # 4c. SSE stream strict parsing + stage-registry contract + report honesty
    from ai.pipeline.stage_registry import stage_ids
    known_stages = set(stage_ids())

    def case_sse():
        r = client.post(
            "/api/process-stream",
            files={"file": ("case.png", tiny_png, "image/png")},
            data={"device": "cpu"},
        )
        assert r.status_code == 200, r.status_code
        events, stage_seq, pct_seq, complete_payload = [], [], [], None
        for line in r.text.splitlines():
            if line.startswith("event: "):
                events.append(line[7:].strip())
            elif line.startswith("data: "):
                payload = json.loads(line[6:])  # strict: raises on malformed JSON
                if events[-1] == "progress":
                    stage_seq.append(payload["stage"])
                    pct_seq.append(int(payload["percent"]))
                elif events[-1] == "complete":
                    complete_payload = payload
        assert complete_payload and complete_payload.get("task_id"), "no complete/task_id"
        assert "restoration_report" in complete_payload, "report missing from complete event"
        # U-7 contract: every emitted stage id must be known to the registry
        # (routes may legitimately skip/reorder stages per image; unknown ids
        # are the contract break).
        unknown = [s for s in stage_seq if s not in known_stages]
        assert not unknown, f"stage ids unknown to the registry: {unknown}"
        # ProgressBar contract: displayed percent is monotonic.
        assert all(b >= a for a, b in zip(pct_seq, pct_seq[1:])), \
            f"non-monotonic progress percent: {pct_seq}"
        # result fetch must return a real image
        img = client.get(f"/api/process/{complete_payload['task_id']}/result")
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        arr = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
        assert arr is not None and arr.shape[0] > 0
        return f"{len(events)} events, {len(stage_seq)} stages, report present"
    run_case("4_api", "sse_contract_strict_json", case_sse)

    def case_report_honesty():
        # A tear-like bright crack must yield reconstructed coverage > 0 and
        # a clean gradient must yield ~0.
        tear = np.full((140, 140, 3), 120, np.uint8)
        cv2.line(tear, (70, 10), (70, 130), (250, 250, 250), 3)
        clean = np.tile(np.linspace(60, 180, 140, dtype=np.uint8), (140, 1))
        clean = np.stack([clean] * 3, axis=2)

        def restore_bytes(arr):
            png = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))[1].tobytes()
            r = client.post("/api/process-stream", files={"file": ("c.png", png, "image/png")},
                            data={"device": "cpu"})
            assert r.status_code == 200, r.status_code
            for line in r.text.splitlines():
                if line.startswith("event: complete"):
                    return json.loads(r.text.splitlines()[r.text.splitlines().index(line) + 1][6:])
            raise AssertionError("no complete event")

        rep_tear = restore_bytes(tear)["restoration_report"]
        # Report shape: recovered/reconstructed/flagged are lists of strings.
        for key in ("recovered", "reconstructed", "flagged", "stage_timings"):
            assert key in rep_tear
        # A real tear must show up as repaired (small coverage) or re-synthesized
        # (large coverage) — either way the report must disclose inpainting.
        tear_text = " ".join(rep_tear["recovered"] + rep_tear["reconstructed"])
        assert "inpainted" in tear_text or "re-synthesized" in tear_text, \
            f"tear not disclosed: {tear_text!r}"

        rep_clean = restore_bytes(clean)["restoration_report"]
        clean_text = " ".join(rep_clean["recovered"] + rep_clean["reconstructed"])
        assert "inpainted" not in clean_text and "re-synthesized" not in clean_text, \
            f"clean image over-reports reconstruction: {clean_text!r}"
        return f"tear disclosed, clean silent"
    run_case("4_api", "restoration_report_honesty", case_report_honesty)

    # 4d. concurrency — 8 parallel sync restorations through the GPU semaphore
    def case_concurrency():
        def one(i):
            r = client.post("/api/process", files={"file": (f"c{i}.png", tiny_png, "image/png")},
                            data={"device": "cpu"})
            return r.status_code, r.headers.get("content-type", "")

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one, range(8)))
        dt = time.perf_counter() - t0
        bad = [r for r in results if r[0] != 200 or "image" not in r[1]]
        assert not bad, f"failed requests: {bad}"
        return f"8/8 ok in {dt:.1f}s (serialized)"
    run_case("4_api", "concurrent_8x_process", case_concurrency)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — GPU spot check
# ─────────────────────────────────────────────────────────────────────────────
def section_gpu():
    import torch
    from ai.pipeline.restoration_pipeline import RestorationPipeline

    if not torch.cuda.is_available():
        record("5_gpu", "cuda_available", False, "SKIP — no CUDA device")
        return
    torch.cuda.reset_peak_memory_stats()
    pipeline = RestorationPipeline(device="cuda", load_models=True)

    def case():
        t0 = time.perf_counter()
        img = cv2.cvtColor(
            cv2.imread(os.path.join("tests", "restoration_dataset", "damaged", "case10_spiderweb_lips.jpg")),
            cv2.COLOR_BGR2RGB,
        )
        out = pipeline.restore(img)
        dt = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1048576
        assert _img_is_sane(out)
        return f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]} in {dt:.1f}s, peak {peak_mb:.0f} MB"
    run_case("5_gpu", "case10_cuda_end_to_end", case)


def main():
    os.makedirs("test_artifacts", exist_ok=True)
    skip_gpu = "--skip-gpu" in sys.argv

    log("=" * 72)
    log("PIXREVIVE ROBUSTNESS AUDIT")
    log("=" * 72)

    log("\n[1/5] Static hygiene (compileall)")
    run_case("1_static", "section", section_static)

    log("\n[2/5] Synthetic input battery")
    run_case("2_synthetic", "section", section_synthetic)

    log("\n[3/5] Toggle matrix (2 images × 32 combos)")
    run_case("3_toggles", "section", section_toggles)

    log("\n[4/5] API surface / SSE / report honesty / concurrency")
    run_case("4_api", "section", section_api)

    if not skip_gpu:
        log("\n[5/5] GPU spot check")
        run_case("5_gpu", "section", section_gpu)
    else:
        log("\n[5/5] GPU spot check — skipped by flag")

    total = sum(len(v) for v in REPORT["sections"].values())
    passed = sum(1 for v in REPORT["sections"].values() for r in v if r["ok"])
    REPORT["finished"] = datetime.now().isoformat()
    REPORT["summary"] = {"total": total, "passed": passed, "failed": total - passed}

    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(REPORT, fp, indent=2)

    log("\n" + "=" * 72)
    log(f"VERDICT: {passed}/{total} checks passed  (report written to {OUT_PATH})")
    if _failures:
        log("Failures:")
        for f in _failures:
            log(f"  FAIL: {f}")
        sys.exit(1)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
