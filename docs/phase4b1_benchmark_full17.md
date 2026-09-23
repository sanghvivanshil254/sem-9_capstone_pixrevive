# PixRevive benchmark report

Run: 2026-09-23T06:07:59+00:00
Previous run: 2026-09-22T19:25:55+00:00

| Case | in damage% | in NIQE | out NIQE | Δ NIQE vs prev | SSIM vs input |
|---|---|---|---|---|---|
| case10_spiderweb_lips | 1.98 | 4.751 | 6.351 | 🟡 +0.09 | 0.7065 |
| case11_minor_creases | 1.37 | 5.619 | 5.537 | 🔴 +0.13 | 0.8023 |
| case12_foxing_spots | 0.30 | 15.721 | 11.788 | 🟢 +0.00 | 0.8957 |
| case1_lena | 3.02 | 4.21 | 3.584 | 🟢 -0.00 | 0.3225 |
| case1_lena_cropped | 1.82 | 3.686 | 3.47 | 🟡 +0.01 | 0.753 |
| case1_noisy_lena | 2.81 | 7.337 | 5.19 | 🟢 -0.01 | 0.5481 |
| case2_dead_tree_landscape | 2.55 | — | 2.875 | 🟢 +0.00 | 0.6177 |
| case3_sepia_flaked_portrait | 5.03 | 5.165 | 6.461 | 🟡 +0.07 | 0.2716 |
| case4_diagonal_tear_portrait | 1.32 | 5.581 | 5.457 | 🟡 +0.02 | 0.8015 |
| case5_spiderweb_creases_portrait | 1.98 | 4.715 | 6.273 | 🔴 +0.18 | 0.7025 |
| case6_severe_3d_torn_photo | 1.86 | 5.541 | 7.542 | 🟢 +0.00 | 0.6879 |
| case7_lena | 3.02 | 4.21 | 3.584 | 🟢 +0.00 | 0.3225 |
| case8_speckle_lena_margins | 3.00 | 4.201 | 3.525 | 🟢 -0.01 | 0.3215 |
| case9_teared_sepia | 5.05 | 5.106 | 5.486 | 🟢 -0.08 | 0.2873 |
| screenshot_portrait_1 | 0.89 | 7.914 | 7.982 | 🟡 +0.00 | 0.8346 |
| screenshot_portrait_2 | 1.24 | 4.694 | 3.95 | 🟢 +0.00 | 0.8864 |
| screenshot_portrait_3 | 0.89 | 6.561 | 7.437 | 🟡 +0.14 | 0.8978 |

**VERDICT: REGRESSION** — NIQE worsened >2% on: case11_minor_creases, case5_spiderweb_creases_portrait
