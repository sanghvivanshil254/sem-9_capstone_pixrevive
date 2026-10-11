"""
Unit tests for FacialSkinToneProtector and ColorArtifactReducer.

Verifies:
- Correction of unnatural green/cyan casts in facial skin zones.
- Multi-face processing with landmark-gated protection of eyes and mouth.
- Edge case handling (no faces, degenerate bboxes, neutral/natural images).
"""

import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.colorization.color_refiner import (
    FacialSkinToneProtector,
    ColorArtifactReducer,
    BandingReducer,
)


def _create_synthetic_portrait(
    width: int = 256,
    height: int = 256,
    face_bbox: list = None,
    landmarks: list = None,
    skin_color_rgb: tuple = (120, 210, 140),  # Unnatural green/cyan skin cast
    eye_color_rgb: tuple = (20, 20, 30),      # Dark pupil/iris
    mouth_color_rgb: tuple = (180, 50, 70),   # Natural red lips
) -> tuple[np.ndarray, dict]:
    """Creates a synthetic image with a face and optional 5 landmarks."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 180  # Neutral gray background

    if face_bbox is None:
        face_bbox = [64, 48, 192, 208]

    x0, y0, x1, y1 = face_bbox
    fw = x1 - x0
    fh = y1 - y0
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    # Draw face oval with unnatural cast
    cv2.ellipse(img, (cx, cy), (int(fw * 0.40), int(fh * 0.48)), 0, 0, 360, skin_color_rgb, -1)

    if landmarks is None:
        # 5 default landmarks: [right eye, left eye, nose, left mouth, right mouth]
        landmarks = [
            [cx - int(fw * 0.18), cy - int(fh * 0.12)],
            [cx + int(fw * 0.18), cy - int(fh * 0.12)],
            [cx, cy + int(fh * 0.05)],
            [cx - int(fw * 0.14), cy + int(fh * 0.22)],
            [cx + int(fw * 0.14), cy + int(fh * 0.22)],
        ]

    # Draw eyes
    for ex, ey in landmarks[:2]:
        cv2.circle(img, (int(ex), int(ey)), max(4, int(fw * 0.06)), eye_color_rgb, -1)

    # Draw mouth
    mx = int((landmarks[3][0] + landmarks[4][0]) // 2)
    my = int((landmarks[3][1] + landmarks[4][1]) // 2)
    cv2.ellipse(img, (mx, my), (int(fw * 0.15), int(fh * 0.06)), 0, 0, 360, mouth_color_rgb, -1)

    face_dict = {
        "bbox": face_bbox,
        "confidence": 0.95,
        "landmarks": landmarks,
    }

    return img, face_dict


def test_skin_tone_protection_corrects_green_cast():
    """Unnatural greenish cast inside face oval should be guided towards warm tones."""
    img, face = _create_synthetic_portrait()

    # Pre-condition: sample skin cheek area has green cast (a* < 128 in LAB)
    lab_before = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cx = (face["bbox"][0] + face["bbox"][2]) // 2
    cy = (face["bbox"][1] + face["bbox"][3]) // 2
    cheek_y, cheek_x = cy + 10, cx + 20

    a_before = lab_before[cheek_y, cheek_x, 1]
    assert a_before < 128, f"Synthetic test setup expected greenish tint (a* < 128), got {a_before}"

    # Apply protection
    corrected = FacialSkinToneProtector.protect_skin_tones(img, faces=[face], strength=0.80)

    lab_after = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
    a_after = lab_after[cheek_y, cheek_x, 1]
    b_after = lab_after[cheek_y, cheek_x, 2]

    # a* channel should have increased towards warm red/melanin (> a_before)
    assert a_after > a_before, f"Expected a* to increase towards warm skin tone, got {a_after} vs {a_before}"
    assert b_after >= 128, f"Expected b* to stay warm/yellow, got {b_after}"


def test_skin_tone_protection_preserves_eyes_and_mouth():
    """Eyes and mouth landmarks should remain protected from skin tone clamping."""
    img, face = _create_synthetic_portrait()

    corrected = FacialSkinToneProtector.protect_skin_tones(img, faces=[face], strength=0.80)

    # Eyes check
    eye_x, eye_y = int(face["landmarks"][0][0]), int(face["landmarks"][0][1])
    # The pupil/iris center should remain intact or change much less than skin
    diff_eye = np.abs(corrected[eye_y, eye_x].astype(int) - img[eye_y, eye_x].astype(int))
    assert np.max(diff_eye) <= 5, f"Eye center should not be altered by skin tone protection: diff={diff_eye}"

    # Mouth check
    mouth_x = int((face["landmarks"][3][0] + face["landmarks"][4][0]) // 2)
    mouth_y = int((face["landmarks"][3][1] + face["landmarks"][4][1]) // 2)
    diff_mouth = np.abs(corrected[mouth_y, mouth_x].astype(int) - img[mouth_y, mouth_x].astype(int))
    assert np.max(diff_mouth) <= 5, f"Mouth center should not be altered: diff={diff_mouth}"


def test_multi_face_protection():
    """Both faces in a multi-face portrait should receive protection."""
    h, w = 300, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 180

    face1_bbox = [40, 40, 180, 220]
    face2_bbox = [280, 40, 420, 220]

    # Draw both faces with greenish cast
    for bbox in [face1_bbox, face2_bbox]:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        cv2.ellipse(img, (cx, cy), (40, 60), 0, 0, 360, (110, 200, 130), -1)

    faces = [
        {"bbox": face1_bbox, "landmarks": None},
        {"bbox": face2_bbox, "landmarks": None},
    ]

    lab_before = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c1x, c1y = (face1_bbox[0] + face1_bbox[2]) // 2, (face1_bbox[1] + face1_bbox[3]) // 2
    c2x, c2y = (face2_bbox[0] + face2_bbox[2]) // 2, (face2_bbox[1] + face2_bbox[3]) // 2
    a1_before = lab_before[c1y, c1x, 1]
    a2_before = lab_before[c2y, c2x, 1]

    corrected = FacialSkinToneProtector.protect_skin_tones(img, faces=faces, strength=0.70)
    lab_after = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)

    # Check center of face 1 and face 2
    assert lab_after[c1y, c1x, 1] > a1_before + 20, "Face 1 chrominance was not significantly improved"
    assert lab_after[c2y, c2x, 1] > a2_before + 20, "Face 2 chrominance was not significantly improved"



def test_edge_cases_and_graceful_fallbacks():
    """Empty faces, missing bbox, or natural images should not error or degrade."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    # No faces
    res_empty = FacialSkinToneProtector.protect_skin_tones(img, faces=[])
    assert np.array_equal(res_empty, img)

    # Degenerate bbox
    res_degen = FacialSkinToneProtector.protect_skin_tones(img, faces=[{"bbox": [10, 10, 12, 12]}])
    assert np.array_equal(res_degen, img)

    # Integrated ColorArtifactReducer
    res_reducer = ColorArtifactReducer.reduce_artifacts(img, enable_skin_protection=True, faces=[])
    assert res_reducer.shape == img.shape
