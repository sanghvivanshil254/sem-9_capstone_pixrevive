#!/usr/bin/env python3
"""corpus_semantic_gate.py — Phase V0 go/no-go gate (plan section 4, V0 exit).

Runs the report-only semantic battery across the fixture corpus:
  * pristine/  -> expect ZERO findings (false-positive gate)
  * damaged/   -> findings allowed; counts recorded for V1 tuning
  * teared.jpg -> the standing user case

Evidence masks come from the project's own Stage-2 damage detectors run
offline (EnhancedDamageDetector + TearmarkDetector via the pipeline's real
fracture-masking stage, minus the U-Net gatekeeper when its weights are
unavailable), matching what Stage 13 sees in production. Faces come from the
same YuNet the pipeline uses.

Usage:
  PYTHONPATH=backend python scripts/corpus_semantic_gate.py [--quick]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "backend"))

from ai.pipeline.stages.damage_stages import DamageStagesMixin  # noqa: E402
from ai.validation.semantic_validator import run_semantic_validation  # noqa: E402
from ai.validation.faceid_embedder import FaceIDEmbedder  # noqa: E402  (YuNet shared)


def _faces_for(bgr: np.ndarray) -> list:
    emb = FaceIDEmbedder()
    face = emb._detect_largest(bgr)
    if face is None:
        return []
    x, y, w, h = [float(v) for v in face[:4]]
    return [{"bbox": [x, y, x + w, y + h]}]


def _evidence_mask(bgr: np.ndarray) -> np.ndarray:
    """Offline Stage-2 mask: the pipeline's real damage stage, model-free path.

    Uses the mixin's fracture-masking logic via a bare PipelineContext stub so
    the numbers reflect the production detectors, not a synthetic proxy.
    """
    from ai.pipeline.pipeline_context import PipelineContext

    ctx = PipelineContext(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mixin = DamageStagesMixin.__new__(DamageStagesMixin)
    mixin._lama = None
    mixin.inpainting_engine = None
    from ai.damage.enhanced_detector import EnhancedDamageDetector
    mixin.enhanced_detector = EnhancedDamageDetector(use_defect_segmenter=False)
    try:
        mixin._stage_2_fracture_masking(ctx, None)
    except Exception as exc:
        print(f"  [warn] offline Stage-2 failed on image: {exc}")
        ctx.damage_masks["final"] = None
    m = ctx.damage_masks.get("final")
    return ((m > 0).astype(np.uint8) * 255 if m is not None
            else np.zeros(bgr.shape[:2], np.uint8))


def _eval(path: str) -> dict:
    bgr = cv2.imread(path)
    if bgr is None:
        return {"file": path, "error": "unreadable"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ev = _evidence_mask(bgr)
    cov = float(np.count_nonzero(ev > 0)) / ev.size * 100
    faces = _faces_for(bgr)
    block = run_semantic_validation(rgb, ev, faces)
    return {"file": os.path.basename(path), "evidence_cov_pct": round(cov, 2),
            "faces": len(faces), "summary": block["summary"],
            "findings": block["findings"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="subset run (gate A + case9/case3/teared)")
    args = ap.parse_args()

    root = os.path.join("tests", "restoration_dataset")
    # The repo ships only the damaged/ corpus (clean/ & expected/ are empty
    # dirs, never populated — verified in git history). Production-correct
    # FP gate: "pristine" == NO lost evidence, i.e. the battery must stay
    # silent on ANY image when the evidence mask is empty (Gate A). This is
    # exactly what Stage 13 sees on a fully-repaired canvas.
    damaged = sorted(glob.glob(os.path.join(root, "damaged", "*")))
    teared = os.path.join("test_artifacts", "sources", "teared.jpg")
    if args.quick:
        damaged = [p for p in damaged
                   if any(k in p for k in ("case9", "case3", "case1_lena."))]

    results = {"gate_a": [], "damaged": [], "teared": None}
    for p in damaged:
        r = _eval(p)
        # Gate A: same image, empty evidence mask — findings here are pure
        # false positives of the evidence conditioning.
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = _faces_for(bgr)
        block = run_semantic_validation(rgb, np.zeros(bgr.shape[:2], np.uint8),
                                        faces)
        results["gate_a"].append({"file": r["file"],
                                  "findings": block["findings"]})
        results["damaged"].append(r)
    if os.path.exists(teared):
        results["teared"] = _eval(teared)

    os.makedirs("test_artifacts", exist_ok=True)
    with open("test_artifacts/semantic_v0_corpus.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    fp = sum(len(r["findings"]) for r in results["gate_a"])
    n = len(results["gate_a"])
    print(f"GATE A (empty evidence => silent): {n} images, {fp} FP "
          f"-> GATE {'PASS' if fp == 0 else 'FAIL'}")
    for r in results["gate_a"]:
        if r["findings"]:
            print("  FP:", r["file"], json.dumps(r["findings"])[:160])
    dmg = sum(len(r.get("findings") or []) for r in results["damaged"])
    print(f"DAMAGED: {len(results['damaged'])} images, {dmg} findings")
    for r in results["damaged"]:
        for f in r.get("findings") or []:
            print(f"  {r['file']}: {f['rule']} [{f['evidence_state']}]")
    t = results["teared"]
    if t:
        print(f"TEARED: coverage {t.get('evidence_cov_pct')}%, faces {t.get('faces')}, "
              f"findings {len(t.get('findings') or [])}")
        for f in t.get("findings") or []:
            print(f"  {f['rule']} [{f['evidence_state']}] {f['verdict'][:80]}")


if __name__ == "__main__":
    main()
