# PixRevive benchmark report

Run: 2026-09-22T19:25:55+00:00
Previous run: 2026-09-22T16:29:46+00:00

| Case | in damage% | in NIQE | out NIQE | Δ NIQE vs prev | SSIM vs input |
|---|---|---|---|---|---|
| case10_spiderweb_lips | 1.98 | 4.751 | 6.263 | 🟢 -0.26 | 0.7064 |
| case11_minor_creases | 1.37 | 5.619 | 5.406 | 🟡 +0.06 | 0.8023 |
| case12_foxing_spots | 0.30 | 15.721 | 11.788 | 🔴 +0.29 | 0.8957 |
| case1_lena | 3.02 | 4.21 | 3.586 | 🟢 -0.23 | 0.3225 |
| case1_lena_cropped | 1.82 | 3.686 | 3.462 | 🟢 -0.40 | 0.7529 |
| case1_noisy_lena | 2.81 | 7.337 | 5.205 | 🟢 -0.20 | 0.5481 |
| case2_dead_tree_landscape | 2.55 | — | 2.875 | 🟢 -0.35 | 0.6177 |
| case3_sepia_flaked_portrait | 5.03 | 5.165 | 6.393 | 🟡 +0.12 | 0.2716 |
| case4_diagonal_tear_portrait | 1.32 | 5.581 | 5.438 | 🟢 -0.18 | 0.8014 |
| case5_spiderweb_creases_portrait | 1.98 | 4.715 | 6.09 | 🟢 -0.20 | 0.7024 |
| case6_severe_3d_torn_photo | 1.86 | 5.541 | 7.542 | 🔴 +0.32 | 0.6879 |
| case7_lena | 3.02 | 4.21 | 3.584 | 🟢 -0.28 | 0.3227 |
| case8_speckle_lena_margins | 3.00 | 4.201 | 3.539 | 🟢 -0.43 | 0.3215 |
| case9_teared_sepia | 5.05 | 5.106 | 5.564 | 🟢 -0.76 | 0.2875 |
| screenshot_portrait_1 | 0.89 | 7.914 | 7.981 | 🟢 -0.11 | 0.8345 |
| screenshot_portrait_2 | 1.24 | 4.694 | 3.95 | 🟢 +0.00 | 0.8864 |
| screenshot_portrait_3 | 0.89 | 6.561 | 7.296 | 🟢 -0.08 | 0.8979 |

**VERDICT: REGRESSION** — NIQE worsened >2% on: case12_foxing_spots, case6_severe_3d_torn_photo
