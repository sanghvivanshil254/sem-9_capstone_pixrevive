#!/usr/bin/env python3
"""smoke_faceid.py — H4 live-path proof in one fresh process (backend STOPPED).

Sequence:
  1. profile() tiers the host (real resource gate, no overrides)
  2. loads the SD1.5 inpaint pipeline through the reimaginer's own loader
  3. installs FaceIDAdapter on the REAL UNet (strict .bin load)
  4. embeds case9's face with the ArcFace embedder (512-d)
  5. projects the embed to 4 tokens, CONCATs to text embeds, runs a short
     generation through the LoRA/SDPA attention path
  6. verifies: non-black output, evidence composite intact, SFace verdict
     recorded, and the re-imagined face's ArcFace embedding similarity to
     the input identity (self-retrieval sanity check)

Usage:  PYTHONPATH=backend python scripts/smoke_faceid.py [--steps 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "backend"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    import torch

    from ai.restoration import resource_profiler, reimaginer
    from ai.restoration.faceid_adapter import FaceIDAdapter
    from ai.validation import faceid_embedder
    from ai.validation.face_identity_gate import SFaceIdentityGate

    t0 = time.time()
    tier, reason, meas = resource_profiler.profile()
    print(f"[1] tier={tier.name} runnable={tier.runnable} reason={reason}",
          flush=True)
    if not tier.runnable:
        print("FATAL: resource gate refused — is the backend stopped?",
              file=sys.stderr)
        raise SystemExit(2)

    reimaginer._STATE.tier = tier
    kind = "sd15"
    pipe = reimaginer._load_pipeline(kind)
    print(f"[2] pipeline loaded ({time.time() - t0:.0f}s)", flush=True)

    # _load_pipeline installs FaceID itself BEFORE the CPU-offload snapshot
    # (a post-offload processor swap leaves new weights un-hooked on CPU —
    # 'mat2 is on cpu' crash, smoke-measured 2026-09-27). Assert it happened;
    # do NOT reinstall here.
    adapter = reimaginer._STATE.faceid
    ok = adapter is not None and adapter.installed
    print(f"[3] FaceID install: {ok}", flush=True)
    if not ok:
        raise SystemExit(3)

    fixture = os.path.join("tests", "restoration_dataset", "damaged",
                           "case9_teared_sepia.jpg")
    bgr = cv2.imread(fixture)
    assert bgr is not None, fixture
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    emb = faceid_embedder.FaceIDEmbedder()
    id_embed = emb.embed(bgr)
    assert id_embed is not None, "ArcFace embedder returned None on case9"
    print(f"[4] id_embed dim={id_embed.shape[0]} norm={np.linalg.norm(id_embed):.4f}",
          flush=True)

    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    y0, y1 = int(h * 0.20), int(h * 0.80)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    mask[y0:y1, x0:x1] = 255

    t1 = time.time()
    gen = reimaginer._generate(rgb, mask, kind, seed=1234, size_px=512,
                               steps=args.steps, guidance=7.5,
                               ip_image=None, id_embed=id_embed)
    print(f"[5] generation ok in {time.time() - t1:.0f}s "
          f"(conditioning used: {reimaginer._STATE.conditioning_used})",
          flush=True)
    black = float((gen.max(axis=2) == 0).mean())
    print(f"    black share: {black:.3f} (NaN-poison threshold "
          f"{reimaginer._BLACK_SHARE})", flush=True)

    composite = reimaginer._composite_evidence(rgb, gen, mask)
    evidence_ok = bool(np.array_equal(composite[mask == 0], rgb[mask == 0]))
    print(f"[6] evidence composite bytewise intact: {evidence_ok}", flush=True)

    gate = SFaceIdentityGate()
    verdict = gate.check(rgb, composite, max_faces=0)
    print(f"[7] SFace verdict: action={verdict.get('action')} "
          f"similarity={verdict.get('similarity')}", flush=True)

    # Self-retrieval sanity: how close is the re-imagined face's ArcFace
    # embedding to the input identity? (informative, not a gate)
    sim = None
    gen_bgr = cv2.cvtColor(gen, cv2.COLOR_RGB2BGR)
    gen_embed = emb.embed(gen_bgr)
    if gen_embed is not None:
        sim = float(np.dot(id_embed, gen_embed))
    print(f"[8] ArcFace self-retrieval (input vs generated): {sim}",
          flush=True)

    # Faceless-input pass: on a FaceID pipe this must NOT crash — the null
    # identity-token path replaces the historical NoneType-embeds failure
    # (regression for the 2026-09-27 crash class).
    reimaginer._STATE.conditioning_used = None
    gen2 = reimaginer._generate(rgb, mask, kind, seed=77, size_px=512,
                                steps=args.steps, guidance=7.5,
                                ip_image=None, id_embed=None)
    print(f"[9] faceless pass ok (conditioning used: "
          f"{reimaginer._STATE.conditioning_used}, black "
          f"{float((gen2.max(axis=2) == 0).mean()):.3f})", flush=True)

    reimaginer.unload_reimagine_models()
    out = {
        "install": ok,
        "conditioning_used": reimaginer._STATE.conditioning_used,
        "faceless_pass_ok": True,
        "black_share": round(black, 4),
        "evidence_intact": evidence_ok,
        "sface_action": verdict.get("action"),
        "sface_similarity": verdict.get("similarity"),
        "arcface_self_retrieval": sim,
        "steps": args.steps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs("test_artifacts", exist_ok=True)
    with open("test_artifacts/smoke_faceid.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    cv2.imwrite("test_artifacts/smoke_faceid.png",
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(json.dumps(out, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
