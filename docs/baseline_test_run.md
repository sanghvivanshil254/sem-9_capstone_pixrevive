# PixRevive — Baseline Test Run (Phase 0.3)

Frozen pre-change record for the restoration-v2 work. Raw output:
[baseline_pytest_output.txt](baseline_pytest_output.txt).

| | |
|---|---|
| Date | 2026-09-22 |
| Commit | `c558e25` — "Initial baseline commit of PixRevive restoration system" |
| Interpreter | `C:\Program Files\Python312\python.exe` (system 3.12.10, no venv) |
| Pytest | `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe`, env `PYTHONPATH=backend` |
| Result | **108 passed, 6 failed** (114 collected), 29 warnings, **207.45 s** |

All 6 failures are pre-existing and confined to one module — they are the accepted
baseline, not regressions introduced by the plan:

```
tests/test_margin_robust_noise.py::TestContentRegion::test_robust_sigma_sees_through_margin
tests/test_margin_robust_noise.py::TestMarginRobustAnalysis::test_additive_noise_image_stays_on_denoise_semantics
tests/test_margin_robust_noise.py::TestMarginRobustAnalysis::test_case8_grain_field_flagged
tests/test_margin_robust_noise.py::TestMarginRobustAnalysis::test_case8_no_bogus_compression_flags
tests/test_margin_robust_noise.py::TestMarginRobustAnalysis::test_case8_noise_score_is_not_diluted
tests/test_margin_robust_noise.py::TestGrainFieldGate::test_speckle_image_gets_tiny_mask_not_5pct
```

Gate for later phases: **never fewer than 108 passing, never a new failure outside
`test_margin_robust_noise.py`** without an explicit plan note.

## Slowest tests (top 10)

| Dur | Test |
|---|---|
| 33.6 s | setup — `test_forensic_restoration.py::test_case1_noisy_lena_satisfies_all_six_hard_constraints` (model init) |
| 26.8 s | call — same (end-to-end pipeline) |
| 25.4 s | `test_restoration_regressions.py::test_pipeline_process_method_returns_png_bytes` |
| 24.3 s | `test_restoration_regressions.py::test_noisy_lowres_portrait_face_damage_guard` |
| 22.4 s | `test_restoration_regressions.py::test_noise_adaptive_stage_5_denoising` |
| 13.0 s | `test_margin_robust_noise.py::test_speckle_image_gets_tiny_mask_not_5pct` |
| 13.0 s | `test_orchestrator_phase4.py::test_orchestrator_manifest_generation` |
| 7.6 s | `test_restoration_regressions.py::test_deferred_face_restoration_logic_on_small_canvas` |
| 6.4 s | `test_orchestrator_phase4.py::test_orchestrator_handles_rollback_gracefully` |
| 5.9 s | `test_orchestrator_phase4.py::test_restoration_pipeline_delegates_to_orchestrator` |

## Notes

- Baseline was taken **before** Phase 1 deletions; nothing in the junkyard purge
  (Phase 1) can excuse a change in these numbers except by fixing a failure.
- Re-run command:
  ```bash
  PYTHONPATH=backend "C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe" tests/ -v
  ```
- Phase 0.4 (progress-stream stage-order reference) still pending — needs a live
  backend on port 8000.
