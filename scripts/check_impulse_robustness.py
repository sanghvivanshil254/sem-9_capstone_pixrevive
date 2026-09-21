"""
check_impulse_robustness.py — Regression suite for impulse-noise handling.

What this guards
----------------
The failure this suite exists to prevent: impulse noise reaching the damage
detectors, being mistaken for emulsion flakes and crack starts, merged by
morphological closing into frame-filling blobs, and repainted by LaMa. The
restoration comes back smeared and crackled, with colorization inventing pastel
hues for the flat inpainted regions.

The property that has to hold is *density invariance*. Damage detection should
report the same damage whether or not the scan happens to be speckled, because
the damage is the same either way. Detector coverage plotted against impulse
density should be a flat line. Before the fix it was not remotely flat — on a
clean synthetic control, final mask coverage went 10.05% -> 96.28% -> 99.62% as
density went 0 -> 0.005 -> 0.010, collapsing from five components into one.

Six checks, each an independent regression:

  1. no-op on clean          the denoiser must not touch an image with no impulses
  2. thin structure survives 1px lines are not noise, including their endpoints
  3. impulse recovery        PSNR must improve, and the output must track the input
  4. density invariance      detector coverage must be flat in impulse density
  5. crack fallback bounded  the classical crack detector must not mask the frame
  6. router impulse stage    NoiseRouter's impulse branch is a no-op on clean input

Scope: the classical detection branch only. The learned U-Net crack path is
switched off explicitly (`use_defect_segmenter=False`), because the classical
branch is what changed and, being pure thresholding, it is the branch most easily
fooled by impulse noise. That also keeps the suite fast and deterministic, and
lets it run on a machine with no model weights at all — the torch stub in
impulse_testkit engages only when torch is genuinely absent. Validating the
shipped learned path end to end means running the real pipeline.

Run (system Python 3.12, per CLAUDE.md):
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/check_impulse_robustness.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Tuple

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from impulse_testkit import (  # noqa: E402
    add_channel_impulse,
    add_salt_pepper,
    changed_fraction,
    component_count,
    coverage,
    detail_energy,
    flat_gradient,
    install_torch_stub,
    photographic,
    psnr,
    real_sources,
    thin_structure,
)

_STUBBED = install_torch_stub()

from ai.analysis.quality_analyzer import QualityAnalyzer  # noqa: E402
from ai.damage.mask_refiner import MaskRefiner  # noqa: E402
from ai.damage.tearmark_detector import TearmarkDetector  # noqa: E402
from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.preprocessing.impulse_denoiser import ImpulseDenoiser  # noqa: E402

# Densities swept by the invariance check. 0.08 is well past anything a real
# scan produces; it is there because the pre-fix behaviour was not merely wrong
# at high density but *discontinuous* — tearmark coverage climbed to 37.45% and
# then collapsed to 0.00% when an internal safety valve tripped.
_DENSITIES = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)

# ── tolerances ────────────────────────────────────────────────────────────────
# A clean image has no impulses, so anything flagged is a false positive. Not
# exactly zero: a synthetic gradient quantised to 8 bits has banding steps, and
# an antialiased circle edge has genuinely isolated pixels.
_MAX_CLEAN_FLAGGED = 0.001          # 0.1% of pixels

# Fraction of known 1px structure pixels that must come through unmodified.
_MIN_STRUCTURE_SURVIVAL = 0.90

# High-frequency energy that must survive on a clean image. The pre-fix filter
# scored 0.422 here, which is what "the face went waxy" looks like as a number.
_MIN_CLEAN_DETAIL = 0.98

# Minimum PSNR improvement over the noisy input, at every density.
_MIN_PSNR_GAIN = 3.0

# Minimum spread in *output* PSNR between the lightest and heaviest noise tested.
# This is the check that the filter responds to its input at all: the pre-fix
# filter returned ~28.5 dB for every density from 0 to 0.10, a spread of ~1 dB.
_MIN_PSNR_SPREAD = 8.0              # dB

# Detector coverage may drift this much (in absolute percentage points of the
# frame) from its value on the same image with no impulses added.
_MAX_COVERAGE_DRIFT = 0.04          # 4 percentage points

# The classical crack fallback's own ceiling, mirrored from enhanced_detector.
_MAX_CRACK_COVERAGE = 0.20


class Report:
    """Accumulates pass/fail across checks and prints a readable log."""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.checks = 0

    def section(self, title: str) -> None:
        print()
        print("=" * 92)
        print(title)
        print("=" * 92)

    def expect(self, ok: bool, label: str, detail: str) -> bool:
        self.checks += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} — {detail}")
        if not ok:
            self.failures.append(f"{label}: {detail}")
        return ok

    def summary(self) -> int:
        print()
        print("=" * 92)
        if self.failures:
            print(f"FAILED — {len(self.failures)} of {self.checks} checks did not pass")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"OK — all {self.checks} checks passed")
        return 0


def _classical_detector():
    """One shared detector, built with the learned segmenter switched off.

    Two reasons this is a module-level singleton rather than per-call. First,
    constructing an EnhancedDamageDetector normally loads a 451 MB U-Net
    checkpoint, and check 4 would otherwise pay that cost once per fixture per
    density. Second, `use_defect_segmenter=False` is the only way to actually
    reach the classical crack branch: setting `.defect_segmenter = None` after
    construction does not work, because detect_thin_crack_network lazily rebuilds
    it and the weights are present in this repo.

    The classical branch is the subject under test here — its thresholds are what
    changed, and being pure thresholding it is the branch most easily fooled by
    impulse noise. The learned branch is exercised by running the real pipeline.
    """
    global _DETECTOR
    if _DETECTOR is None:
        from ai.damage.enhanced_detector import EnhancedDamageDetector
        _DETECTOR = EnhancedDamageDetector(use_defect_segmenter=False)
    return _DETECTOR


_DETECTOR = None


def _final_mask(rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Run the classical Stage-2 detector set and return the refined mask."""
    ctx = PipelineContext(rgb)
    ctx.image_metadata.update(QualityAnalyzer.analyze(rgb))

    det = _classical_detector()

    parts = {
        "flakes": det.detect_white_flakes(rgb, ctx),
        "wide": det.detect_wide_cracks(rgb, ctx),
        "thin": det.detect_thin_crack_network(rgb, ctx),
        "spots": det.detect_dark_spots(rgb, ctx),
    }
    tearmark = TearmarkDetector().detect(ctx)
    parts["tearmark"] = tearmark

    combined = np.zeros(rgb.shape[:2], np.uint8)
    for m in ("flakes", "wide", "thin", "spots"):
        combined = cv2.bitwise_or(combined, parts[m])
    ctx.damage_masks["tearmark"] = tearmark
    return MaskRefiner.refine(combined, ctx), parts


# ── check 1 ───────────────────────────────────────────────────────────────────

def check_clean_is_noop(rep: Report) -> None:
    rep.section("1. No-op on clean input — every flagged pixel is a false positive")
    fixtures = [
        ("flat_gradient", flat_gradient()),
        ("thin_structure", thin_structure()[0]),
        ("photographic", photographic()),
    ]
    for name, rgb in fixtures:
        out, field = ImpulseDenoiser.apply(rgb)
        changed = changed_fraction(rgb, out)
        retention = detail_energy(out) / max(detail_energy(rgb), 1e-6)
        rep.expect(
            field.density <= _MAX_CLEAN_FLAGGED,
            f"{name}: flagged fraction",
            f"{field.density:.4%} (ceiling {_MAX_CLEAN_FLAGGED:.2%}), "
            f"changed {changed:.4%}, sigma {field.sigma:.2f}, T {field.threshold:.1f}",
        )
        rep.expect(
            retention >= _MIN_CLEAN_DETAIL,
            f"{name}: detail retention",
            f"{retention:.1%} (floor {_MIN_CLEAN_DETAIL:.0%})",
        )


# ── check 2 ───────────────────────────────────────────────────────────────────

def check_thin_structure_survives(rep: Report) -> None:
    rep.section("2. Thin 1px structure is not noise — including line endpoints")
    img, structure = thin_structure()
    out, field = ImpulseDenoiser.apply(img)

    struct = structure > 0
    intact = np.all(out[struct] == img[struct], axis=-1)
    survival = float(np.mean(intact)) if struct.any() else 1.0
    rep.expect(
        survival >= _MIN_STRUCTURE_SURVIVAL,
        "clean structure survival",
        f"{survival:.1%} of {int(struct.sum())} structure pixels untouched "
        f"(floor {_MIN_STRUCTURE_SURVIVAL:.0%}), flagged {field.density:.3%}",
    )

    # And it must still survive while impulses around it are being removed.
    noisy = add_salt_pepper(img, 0.02, seed=5)
    # Only judge structure pixels the noise did not itself overwrite.
    untouched_by_noise = struct & np.all(noisy == img, axis=-1)
    out_n, _ = ImpulseDenoiser.apply(noisy)
    intact_n = np.all(out_n[untouched_by_noise] == img[untouched_by_noise], axis=-1)
    survival_n = float(np.mean(intact_n)) if untouched_by_noise.any() else 1.0
    rep.expect(
        survival_n >= _MIN_STRUCTURE_SURVIVAL,
        "structure survival at density 0.02",
        f"{survival_n:.1%} of {int(untouched_by_noise.sum())} uncorrupted "
        f"structure pixels untouched (floor {_MIN_STRUCTURE_SURVIVAL:.0%})",
    )


# ── check 3 ───────────────────────────────────────────────────────────────────

def check_impulse_recovery(rep: Report) -> None:
    rep.section("3. Impulse recovery — PSNR must improve, and the output quality "
                "must track the input's")
    clean = photographic()
    print(f"  {'density':>8} {'flagged%':>9} {'changed%':>9} {'PSNR in':>9} "
          f"{'PSNR out':>9} {'gain dB':>9} {'detail':>8}")
    rows: List[Tuple[float, float, float]] = []   # (density, psnr_out, gain)
    for d in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        noisy = add_salt_pepper(clean, d, seed=3)
        out, field = ImpulseDenoiser.apply(noisy)
        p_in, p_out = psnr(clean, noisy), psnr(clean, out)
        rows.append((d, p_out, p_out - p_in))
        print(f"  {d:>8.3f} {field.density * 100:>9.3f} "
              f"{changed_fraction(noisy, out) * 100:>9.3f} {p_in:>9.2f} {p_out:>9.2f} "
              f"{p_out - p_in:>+9.2f} "
              f"{detail_energy(out) / detail_energy(clean) * 100:>7.1f}%")

    for d, _, gain in rows:
        rep.expect(gain >= _MIN_PSNR_GAIN, f"PSNR gain at density {d:.3f}",
                   f"{gain:+.2f} dB (floor +{_MIN_PSNR_GAIN:.1f} dB)")

    # The pre-fix filter's signature failure was an output PSNR of ~28.5 dB at
    # *every* input density from 0 to 0.10 — it converged to the same degraded
    # image regardless of how much noise it was given, because it was filtering
    # content rather than noise. A filter that actually responds to its input
    # produces a markedly better result from a lightly corrupted image than from
    # a heavily corrupted one. Note this is deliberately a check on output PSNR
    # and not on the dB *gain*: a filter that removes a constant fraction of
    # impulses has a constant gain by construction, so a rising gain is not the
    # property that distinguishes a working filter from a broken one.
    spread = rows[0][1] - rows[-1][1]
    rep.expect(
        spread >= _MIN_PSNR_SPREAD,
        "output quality tracks input quality",
        f"PSNR out {rows[0][1]:.2f} dB at density {rows[0][0]:.3f} vs "
        f"{rows[-1][1]:.2f} dB at {rows[-1][0]:.3f} — spread {spread:.2f} dB "
        f"(floor {_MIN_PSNR_SPREAD:.1f} dB)",
    )
    rep.expect(
        all(rows[i][1] >= rows[i + 1][1] - 1.0 for i in range(len(rows) - 1)),
        "output PSNR is non-increasing in density",
        " -> ".join(f"{p:.1f}" for _, p, _ in rows),
    )

    # Per-channel impulses: a luminance-only detector misses these.
    noisy_c = add_channel_impulse(clean, 0.02, seed=4)
    out_c, field_c = ImpulseDenoiser.apply(noisy_c)
    gain_c = psnr(clean, out_c) - psnr(clean, noisy_c)
    rep.expect(
        gain_c >= _MIN_PSNR_GAIN,
        "per-channel impulse recovery at density 0.02",
        f"{gain_c:+.2f} dB, flagged {field_c.density:.3%}, "
        f"per-channel {[f'{c:.3%}' for c in field_c.channels]}",
    )


# ── check 4 ───────────────────────────────────────────────────────────────────

def check_density_invariance(rep: Report) -> None:
    rep.section("4. Density invariance — detector coverage must not track impulse "
                "density")
    fixtures: List[Tuple[str, np.ndarray]] = [
        ("flat_gradient", flat_gradient()),
        ("thin_structure", thin_structure()[0]),
    ]
    # Real photographs add coverage the synthetics cannot: genuine film grain,
    # JPEG blocking, real damage. Capped at three because this check is the
    # expensive one — the whole classical detector set runs once per density per
    # image — and the suite has to stay cheap enough to run on every change.
    fixtures.extend(real_sources(REPO)[:3])

    for name, rgb in fixtures:
        print(f"\n  {name}  {rgb.shape[1]}x{rgb.shape[0]}")
        print(f"  {'density':>8} {'flakes%':>8} {'thin%':>8} {'tearmk%':>8} "
              f"{'FINAL%':>8} {'comps':>7}")
        baseline: float | None = None
        drifts: List[Tuple[float, float]] = []
        for d in _DENSITIES:
            noisy = add_salt_pepper(rgb, d, seed=1)
            cleaned, _ = ImpulseDenoiser.apply(noisy)   # Stage 0
            mask, parts = _final_mask(cleaned)
            cov = coverage(mask)
            if baseline is None:
                baseline = cov
            drifts.append((d, abs(cov - baseline)))
            print(f"  {d:>8.3f} {coverage(parts['flakes']) * 100:>8.2f} "
                  f"{coverage(parts['thin']) * 100:>8.2f} "
                  f"{coverage(parts['tearmark']) * 100:>8.2f} "
                  f"{cov * 100:>8.2f} {component_count(mask):>7}")

        worst_d, worst = max(drifts, key=lambda t: t[1])
        rep.expect(
            worst <= _MAX_COVERAGE_DRIFT,
            f"{name}: coverage drift",
            f"worst {worst * 100:.2f} pp at density {worst_d:.3f} "
            f"(ceiling {_MAX_COVERAGE_DRIFT * 100:.0f} pp, baseline "
            f"{(baseline or 0) * 100:.2f}%)",
        )


# ── check 5 ───────────────────────────────────────────────────────────────────

def check_crack_fallback_bounded(rep: Report) -> None:
    rep.section("5. Classical crack fallback must not mask the frame")
    det = _classical_detector()

    fixtures: List[Tuple[str, np.ndarray]] = [
        ("flat_gradient", flat_gradient()),
        ("thin_structure", thin_structure()[0]),
        ("photographic", photographic()),
    ]
    fixtures.extend(real_sources(REPO))

    for name, rgb in fixtures:
        ctx = PipelineContext(rgb)
        ctx.image_metadata.update(QualityAnalyzer.analyze(rgb))
        mask = det.detect_thin_crack_network(rgb, ctx)
        cov = coverage(mask)
        rep.expect(
            cov <= _MAX_CRACK_COVERAGE,
            f"{name}: crack coverage",
            f"{cov:.2%} (ceiling {_MAX_CRACK_COVERAGE:.0%})",
        )


# ── check 6 ───────────────────────────────────────────────────────────────────

def check_router_impulse_stage(rep: Report) -> None:
    rep.section("6. NoiseRouter's impulse branch — no-op on clean, effective on noisy")
    from ai.postprocessing.noise_router import NoiseClassifier, _reduce_impulse

    clean = photographic()
    out = _reduce_impulse(clean, NoiseClassifier.classify(clean, {}))
    changed = changed_fraction(clean, out)
    retention = detail_energy(out) / max(detail_energy(clean), 1e-6)
    rep.expect(
        changed <= _MAX_CLEAN_FLAGGED,
        "_reduce_impulse on clean: pixels changed",
        f"{changed:.4%} (ceiling {_MAX_CLEAN_FLAGGED:.2%})",
    )
    rep.expect(
        retention >= _MIN_CLEAN_DETAIL,
        "_reduce_impulse on clean: detail retention",
        f"{retention:.1%} (floor {_MIN_CLEAN_DETAIL:.0%})",
    )

    noisy = add_salt_pepper(clean, 0.02, seed=9)
    out_n = _reduce_impulse(noisy, NoiseClassifier.classify(noisy, {}))
    gain = psnr(clean, out_n) - psnr(clean, noisy)
    rep.expect(
        gain >= _MIN_PSNR_GAIN,
        "_reduce_impulse on noisy: PSNR gain",
        f"{gain:+.2f} dB (floor +{_MIN_PSNR_GAIN:.1f} dB)",
    )


def main() -> int:
    print("Impulse-noise robustness regression suite")
    print(f"repo: {REPO}")
    print(f"torch: {'stubbed (classical CV only)' if _STUBBED else 'real'}")

    rep = Report()
    checks: List[Callable[[Report], None]] = [
        check_clean_is_noop,
        check_thin_structure_survives,
        check_impulse_recovery,
        check_density_invariance,
        check_crack_fallback_bounded,
        check_router_impulse_stage,
    ]
    for fn in checks:
        try:
            fn(rep)
        except Exception as exc:      # a crashing check is a failing check
            import traceback
            rep.section(f"{fn.__name__} raised")
            traceback.print_exc()
            rep.expect(False, fn.__name__, f"raised {type(exc).__name__}: {exc}")

    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())
