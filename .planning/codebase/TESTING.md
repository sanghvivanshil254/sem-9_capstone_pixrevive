# Testing Patterns

**Analysis Date:** 2026-09-19

## Test Framework

**Runner:**
- Backend: Pytest via system Python (`C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe`).
- Frontend / End-to-End: Playwright (`@playwright/test` v1.62.1).
- Config: `playwright.config.js` at repository root.

**Assertion Library:**
- Python standard library `unittest.TestCase` assertions (`self.assertTrue`, `self.assertEqual`, `self.assertLess`).
- Playwright assertions (`await expect(page).toHaveTitle(...)`, `await expect(locator).toBeVisible()`).

**Run Commands:**
```powershell
# Run backend test suite (PowerShell)
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/ -v

# Run single test file
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_restoration_regressions.py -v

# Run color refiner regression tests
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_color_refiner.py -v

# Run end-to-end browser tests (Playwright)
npm run test:e2e

# Run Playwright in interactive UI mode
npm run test:e2e:ui
```

## Test File Organization

**Location:**
- Backend tests live in `tests/` directory at the project root.
- E2E tests live in `e2e/` directory at the project root.
- Diagnostic and performance verification scripts live in `test_artifacts/`.

**Structure:**
```
tests/
├── restoration_dataset/
│   └── damaged/
│       ├── case1_noisy_lena.jpg          # Gaussian noise on flat grayscale
│       ├── case2_dead_tree_landscape.jpg # Landscape macro-edges
│       ├── case3_sepia_flaked_portrait.jpg# Emulsion flaking on sepia portrait
│       ├── case4_diagonal_tear_portrait.jpg# Diagonal paper fracture
│       ├── case5_spiderweb_creases_portrait.jpg# Spiderweb facial cracks + open smile
│       └── case6_severe_3d_torn_photo.png# Severe 3D paper tear & curl
├── test_admin_router.py                  # Administrative endpoint security tests
├── test_color_refiner.py                 # Color bleed, banding, and skin protection tests
├── test_compat.py                        # TorchVision functional_tensor shim verification
├── test_facial_landmarks.py              # RetinaFace 5-landmark coordinate accuracy
├── test_forensic_restoration.py          # 6 Hard Constraints forensic auditor tests
├── test_identity_preservation.py         # Face identity, SSIM, and landmark drift tests
├── test_image_router.py                  # Process and result endpoint tests
├── test_progress_stream.py               # SSE streaming monotonicity tests
└── test_restoration_regressions.py       # Core 24-point restoration regression suite
```

## Testing Patterns & Best Practices

1. **Strict Python Runtime Enforcement:**
   - Tests MUST be run with system Python 3.12 and `PYTHONPATH=backend`. Virtual environments are strictly prohibited.
2. **Deterministic Geometric Normalization:**
   - Identity tests resize restored outputs back to input bounding boxes before SSIM or landmark distance calculations (`IdentityValidator.validate_context`).
3. **Mocking External and Heavy Deep Models:**
   - API streaming tests (`test_progress_stream.py`) mock `ImageService.process_image` to emit progress events without incurring multi-second neural inference latency during unit test passes.
4. **Synthetic Degradation Fixtures:**
   - Tests programmatically inject controlled Gaussian noise, impulse salt-and-pepper spikes, artificial boundary cracks, and color casts to verify specific algorithm stages in isolation.
