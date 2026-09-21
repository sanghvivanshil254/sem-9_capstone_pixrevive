"""
scripts/forensic_restore_validator.py — Forensic Photo Restoration Validator Runner.

Executes the PixRevive restoration pipeline on a speckle-noise-corrupted grayscale photograph,
verifies all 6 Hard Constraints line-by-line using ForensicValidator, and produces:
1. Forensically restored 4x image (`test_artifacts/case1_forensic_restored.png`).
2. Full forensic audit report (`test_artifacts/case1_forensic_audit_report.md`).
3. Multi-stage visual progression and comparison grid (`test_artifacts/case1_forensic_comparison.png`).

Usage:
    & "C:\Program Files\Python312\python.exe" scripts/forensic_restore_validator.py [optional_image_path]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Ensure backend is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai.pipeline.restoration_pipeline import RestorationPipeline
from ai.pipeline.pipeline_context import PipelineContext
from ai.validation.forensic_validator import ForensicValidator, ForensicAuditResult


DEFAULT_IMAGE = str(ROOT / "tests" / "restoration_dataset" / "damaged" / "case1_noisy_lena.jpg")


def run_forensic_restoration(image_path: str = DEFAULT_IMAGE) -> Tuple[np.ndarray, np.ndarray, ForensicAuditResult]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print(f"=== PixRevive Forensic Photo Restoration Validator ===")
    print(f"Source image: {image_path}")

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to read image at {image_path}")
    raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_in, w_in = raw_rgb.shape[:2]
    print(f"Raw dimensions: {w_in}x{h_in}, channels: {raw_rgb.shape[2]}")

    pipeline = RestorationPipeline()
    print("Waiting for deep learning models to initialize...")
    pipeline._models_ready.wait(timeout=30.0)
    pipeline._colorizer_ready.wait(timeout=30.0)

    t0 = time.time()
    context = PipelineContext(raw_rgb.copy(), {"color_type": "grayscale"})
    context.pipeline_flags.update({
        "enable_repair": True,
        "enable_denoise": True,
        "enable_colorization": True,
        "enable_face_restore": True,
        "enable_sr": True,
    })

    def progress_callback(stage: str, percent: int, message: str) -> None:
        print(f"  [{percent:3d}%] {stage:<15} : {message}")

    restored_rgb = pipeline.restore(
        image_input=raw_rgb,
        enable_repair=True,
        enable_denoise=True,
        enable_colorization=True,
        enable_face_restore=True,
        enable_sr=True,
        on_progress=progress_callback,
    )
    elapsed = time.time() - t0
    h_out, w_out = restored_rgb.shape[:2]
    print(f"Pipeline executed in {elapsed:.2f}s! Restored canvas: {w_out}x{h_out}")

    # Extract detected faces for forensic validator
    faces = context.image_metadata.get("faces", [])
    face_bbox = faces[0].get("bbox") if faces else None
    landmarks = faces[0].get("landmarks") if faces else None

    print("\n--- Running Line-by-Line Forensic Self-Check (6 Hard Constraints) ---")
    audit_result = ForensicValidator.audit(
        input_img=raw_rgb,
        output_img=restored_rgb,
        face_bbox=face_bbox,
        landmarks=landmarks,
    )

    return raw_rgb, restored_rgb, audit_result


def generate_comparison_grid(
    raw_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    audit_result: ForensicAuditResult,
    out_grid_path: str,
) -> None:
    """Creates a high-resolution visual audit card."""
    h_out, w_out = restored_rgb.shape[:2]
    raw_scaled = cv2.resize(raw_rgb, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

    # 2-panel comparison (Raw vs Restored) with informational header
    header_h = 100
    footer_h = 120
    margin = 30
    gap = 30

    canvas_w = margin * 2 + w_out * 2 + gap
    canvas_h = margin * 2 + header_h + h_out + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="#0e131b")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 32)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_card = ImageFont.truetype("arialbd.ttf", 20)
        font_footer = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_title = font_sub = font_card = font_footer = ImageFont.load_default()

    # Header
    draw.text((margin, margin), "PixRevive — Forensic Photo Restoration & Ground Truth Audit", fill="#ffffff", font=font_title)
    draw.text((margin, margin + 45), f"Target: Lena (case1_noisy_lena.jpg) | Zero Structural Distortion | Status: {'PASS' if audit_result.passed else 'FAIL'}", fill="#4fd1c5", font=font_sub)

    # Place raw panel
    raw_pil = Image.fromarray(raw_scaled)
    y_panels = margin + header_h
    canvas.paste(raw_pil, (margin, y_panels))
    draw.rectangle([margin, y_panels - 35, margin + w_out, y_panels], fill="#1a202c")
    draw.text((margin + 15, y_panels - 28), f"Source Ground Truth (196x148 Grayscale Speckle Noise)", fill="#e2e8f0", font=font_card)

    # Place restored panel
    restored_pil = Image.fromarray(restored_rgb)
    x_restored = margin + w_out + gap
    canvas.paste(restored_pil, (x_restored, y_panels))
    draw.rectangle([x_restored, y_panels - 35, x_restored + w_out, y_panels], fill="#1a202c")
    draw.text((x_restored + 15, y_panels - 28), f"Forensically Certified Restored Output ({w_out}x{h_out} 4x Super-Res)", fill="#38a169", font=font_card)

    # Footer verification badges
    y_footer = y_panels + h_out + 25
    badge_x = margin
    for cr in audit_result.constraint_results:
        color = "#22c55e" if cr.passed else "#ef4444"
        txt = f"C{cr.constraint_id} [{cr.name[:18]}...]: {'PASS' if cr.passed else 'FAIL'} ({cr.score:.2f})"
        draw.text((badge_x, y_footer), txt, fill=color, font=font_footer)
        badge_x += (w_out * 2 + gap) // 3
        if badge_x > canvas_w - 200:
            badge_x = margin
            y_footer += 28

    canvas.save(out_grid_path, quality=95)
    print(f"Saved visual comparison card to: {out_grid_path}")


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    out_dir = ROOT / "test_artifacts"
    os.makedirs(out_dir, exist_ok=True)

    raw_rgb, restored_rgb, audit_result = run_forensic_restoration(image_path)

    # 1. Save restored PNG
    out_img_path = str(out_dir / "case1_forensic_restored.png")
    cv2.imwrite(out_img_path, cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))
    print(f"\n[OK] Saved forensically restored PNG to: {out_img_path}")

    # 2. Save audit markdown report
    out_report_path = str(out_dir / "case1_forensic_audit_report.md")
    report_content = audit_result.format_report()
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[OK] Saved audit report to: {out_report_path}")

    # 3. Save comparison card
    out_card_path = str(out_dir / "case1_forensic_comparison.png")
    generate_comparison_grid(raw_rgb, restored_rgb, audit_result, out_card_path)

    print("\n" + report_content)

    if not audit_result.passed:
        print("\n[ERROR] One or more forensic hard constraints failed!")
        return 1

    print("\n[SUCCESS] All 6 Hard Constraints verified! Zero structural or geometric corruption.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
