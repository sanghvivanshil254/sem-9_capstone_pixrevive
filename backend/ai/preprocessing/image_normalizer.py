import io
import numpy as np
import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

class ImageNormalizer:
    # Set a safe limit to prevent Out-Of-Memory (OOM) attacks (e.g. ~10,000 x 10,000)
    MAX_IMAGE_PIXELS = 100000000

    @classmethod
    def normalize(cls, image_bytes: bytes) -> np.ndarray:
        """
        Safely loads image bytes, corrects orientation, drops alpha channel,
        converts to RGB, and returns a numpy array.
        Raises ValueError for corrupted, unsupported, or overly large images.
        """
        Image.MAX_IMAGE_PIXELS = cls.MAX_IMAGE_PIXELS
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Verify the image is fully readable
            img.verify()
        except UnidentifiedImageError:
            raise ValueError("Unsupported or corrupted image format.")
        except Exception as e:
            raise ValueError(f"Error opening image: {str(e)}")

        # reopen after verify() because verify() can leave the file in an unstable state
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Error reopening image: {str(e)}")

        # EXIF Orientation correction
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass # Ignore if exif transpose fails

        # Convert to RGB safely
        if img.mode != 'RGB':
            try:
                # If transparent, composite over a white background instead of turning black
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGBA')
                    background = Image.new('RGBA', img.size, (255, 255, 255))
                    img = Image.alpha_composite(background, img)
                    
                img = img.convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to convert image to RGB: {str(e)}")

        return np.array(img)

    @classmethod
    def detect_and_crop_borders(
        cls,
        img: np.ndarray,
        white_thresh: float = 242.0,
        black_thresh: float = 15.0,
        max_std: float = 6.0,
        min_margin: int = 6,
    ) -> tuple[np.ndarray, bool, tuple[int, int, int, int]]:
        """
        Detect solid/uniform scanner framing margins (white or black canvas mats)
        and cleanly crop to the genuine photo content bounds.
        
        Returns:
            (cropped_img, has_border, (y_min, y_max, x_min, x_max))
        """
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return img, False, (0, h, 0, w)

        if img.ndim == 3:
            gray = (
                0.299 * img[:, :, 0].astype(np.float32) +
                0.587 * img[:, :, 1].astype(np.float32) +
                0.114 * img[:, :, 2].astype(np.float32)
            )
        else:
            gray = img.astype(np.float32)

        # Check top border
        y_min = 0
        while y_min < h - 12:
            row = gray[y_min, :]
            m, s = float(np.mean(row)), float(np.std(row))
            if (m >= white_thresh or m <= black_thresh) and s <= max_std:
                y_min += 1
            else:
                break

        # Check bottom border
        y_max = h
        while y_max > y_min + 12:
            row = gray[y_max - 1, :]
            m, s = float(np.mean(row)), float(np.std(row))
            if (m >= white_thresh or m <= black_thresh) and s <= max_std:
                y_max -= 1
            else:
                break

        # Check left border
        x_min = 0
        while x_min < w - 12:
            col = gray[:, x_min]
            m, s = float(np.mean(col)), float(np.std(col))
            if (m >= white_thresh or m <= black_thresh) and s <= max_std:
                x_min += 1
            else:
                break

        # Check right border
        x_max = w
        while x_max > x_min + 12:
            col = gray[:, x_max - 1]
            m, s = float(np.mean(col)), float(np.std(col))
            if (m >= white_thresh or m <= black_thresh) and s <= max_std:
                x_max -= 1
            else:
                break

        has_border = (
            y_min >= min_margin or
            (h - y_max) >= min_margin or
            x_min >= min_margin or
            (w - x_max) >= min_margin
        )

        cropped_ratio = ((y_max - y_min) * (x_max - x_min)) / float(h * w)
        if has_border and cropped_ratio >= 0.20:
            return img[y_min:y_max, x_min:x_max].copy(), True, (y_min, y_max, x_min, x_max)

        return img, False, (0, h, 0, w)

    # ------------------------------------------------------------------ T-4 --
    # Auto-deskew (Phase 4B.1). Skewed scans waste detector accuracy (edges
    # stop being axis-aligned) and leave wedges of mat after any crop. Both
    # problems are cheapest to fix here, BEFORE detect_and_crop_borders: the
    # rotation wedges land in the margins and the crop then removes them.

    MIN_SKEW_DEG = 0.3     # plan T-4 gate: skip when |angle| < 0.3°
    MAX_SKEW_DEG = 8.0     # never rotate by a wild amount on ambiguous evidence

    @classmethod
    def estimate_skew_angle(cls, img: np.ndarray) -> float:
        """Estimate the dominant skew of a scan from long straight structures.

        Long line segments (photo borders, page edges, text baselines) are the
        only reliable skew evidence; short edges describe content, not scan
        geometry. Horizontal and vertical families both vote: a scan rotated
        by θ puts true-horizontal content at θ and true-vertical content at
        90+θ, which fold to the same evidence.

        Returns the CORRECTION angle in degrees in [-45, 45) — the value to
        pass to cv2.getRotationMatrix2D to undo the skew (a scan warped with
        rotation parameter +t measures -t; applying the returned angle
        restores axis alignment) — or 0.0 when the evidence is absent or
        scattered (natural photos have edges in every direction — the
        estimator must stay inert on them, the same lesson as Stage 0's
        autocorrelation gate).
        """
        h, w = img.shape[:2]
        if h < 60 or w < 60:
            return 0.0

        # Scale-invariant angle: work on a bounded canvas for speed.
        scale = 1024.0 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Median-based Canny bounds (robust across exposure).
        v = float(np.median(gray))
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lower, upper)

        # Standard (exhaustive) Hough transform, NOT the probabilistic variant:
        # HoughLinesP's random point sampling can return zero segments at
        # boundary angles on perfectly clean input (observed at 2.5deg on a
        # synthetic scan), which would make the gate flaky. HoughLines is
        # deterministic, and its accumulator count is a direct measurement of
        # the collinear pixel support — the natural weight for each line vote.
        # theta resolution 0.5deg; a line must be supported by >=120 edge px
        # (short content edges stay in the pool; the dominance test below
        # decides whether they matter).
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 360.0, threshold=120)
        if lines is None or len(lines) == 0:
            return 0.0

        # Fold every line's direction to [-45, 45): the horizontal family
        # (near 0) and the vertical family (near ±90) then vote for the same
        # skew, and genuine diagonal content (near ±45) lands where the
        # MAX_SKEW_DEG range check rejects it.
        #   dir_deg = normal_angle - 90 is the line direction in degrees.
        raw = []
        for rho_ang, theta in lines[:200, 0]:  # sorted by votes desc
            dir_deg = np.degrees(float(theta)) - 90.0
            a = (dir_deg + 90.0) % 180.0 - 90.0   # fold to [-90, 90)
            if a > 45.0:
                a -= 90.0
            elif a < -45.0:
                a += 90.0                          # fold to [-45, 45)
            raw.append((a, float(rho_ang)))

        total_weight = sum(l for _, l in raw)
        if total_weight <= 0:
            return 0.0

        # Weighted 1deg histogram over [-45, 45).
        bins = np.zeros(90, dtype=np.float64)
        for a, l in raw:
            bins[int(np.clip(a + 45.0, 0, 89))] += l
        peak_angle = float(np.argmax(bins)) - 45.0

        # Dominance test: the peak family (±2deg) must carry most of the
        # long-line vote weight. Duplicate Hough peaks for one physical line
        # are evidence mass, not padding — a stronger line legitimately votes
        # harder, and a distinct-lines count requirement is unmeetable on
        # sparse scans (a synthetic with 3 bars yields 1-2 physical lines).
        # Scattered orientation (natural scenes) fails the ratio and the
        # estimator stays inert.
        near = [(a, l) for a, l in raw if abs(a - peak_angle) <= 2.0]
        near_weight = sum(l for _, l in near)
        if near_weight / total_weight < 0.35:
            return 0.0

        # Length-weighted circular mean of the near-peak votes (sub-degree
        # refinement over the 1deg histogram).
        rad = np.radians([a for a, _ in near])
        wts = np.array([l for _, l in near])
        mean_rad = np.arctan2(float(np.sum(wts * np.sin(rad))),
                              float(np.sum(wts * np.cos(rad))))
        return float(np.degrees(mean_rad))

    @classmethod
    def deskew(cls, img: np.ndarray) -> tuple[np.ndarray, float]:
        """T-4: affine-correct scan skew. Returns (image, applied_angle).

        Gated: the image is returned BITWISE unchanged unless the dominant
        skew is measurable and within [MIN_SKEW_DEG, MAX_SKEW_DEG]. Rotation
        fills the revealed wedges with the scanner mat colour (white); the
        following border crop then trims them away.
        """
        angle = cls.estimate_skew_angle(img)
        if angle == 0.0 or abs(angle) < cls.MIN_SKEW_DEG or abs(angle) > cls.MAX_SKEW_DEG:
            return img, 0.0

        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            img, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated, angle

    # ------------------------------------------------------------------ T-5 --
    # Flat-field illumination correction (Phase 4B.2). Scanners light the pane
    # unevenly: the lamp falls off toward the edges, so the same emulsion
    # density records darker near the borders and downstream contrast gates
    # (damage thresholds, variance clusters) measure geometry, not material.
    # The classical fix is a background estimate + divide; the hard part is the
    # GATE, because natural content is full of smooth gradients (a portrait's
    # dark hair beside a lit cheek) that a naive "illumination varies" test
    # cannot tell from a lamp defect.
    #
    # The gate was chosen by probe (scripts/_probe_flatfield.py, deleted after
    # the phase): fit a quadratic field to the morphological-closing envelope
    # of 17 real dataset photos plus synthetic vignettes/tilts/content bands,
    # and measure candidate discriminators. Results that killed the obvious
    # candidates:
    #   * per-channel field correlation — dead: EVERYTHING measures ~1.0
    #     (portrait channels are mutually correlated; even a half-black band
    #     is channel-neutral). Not a lamp signature after all.
    #   * quadratic fit R^2 — dead: real photos span 0.09-0.98, no gap.
    #   * raw field amplitude M — dead: content reaches M=0.91 (case2), a
    #     0.25-strength vignette only 0.16.
    # The survivor is SYMMETRIC CORNER FALLOFF: a lamp/vignette defect puts
    # ALL FOUR corners of the envelope down together (probe: vignette corner
    # spread 0.011, corner deficit 0.67), while content gradients are
    # spatially lopsided (every real photo spread >= 0.27; a dark content band
    # spread 1.30). Gating on symmetric deficit keeps the stage inert on the
    # entire real corpus; the cost is documented below.

    MIN_CORNER_DEFICIT = 0.20   # fire only when corners drop >=20% below center
    MAX_CORNER_SPREAD = 0.12    # and only when all four drop TOGETHER
    MAX_CORRECTION = 2.0        # never brighten a pixel by more than 2x

    @classmethod
    def estimate_illumination_deficit(cls, img: np.ndarray) -> dict:
        """Measure symmetric corner falloff of the image's background envelope.

        Returns {"corner_deficit", "corner_spread", "field"} where the first
        two gate the correction and `field` is the low-order background field
        (None when unmeasurable). On the real corpus these read deficit 0.0-0.6
        with spread 0.27-0.63 (asymmetric => content, not lamp); a synthetic
        vignette reads deficit 0.2-0.67 with spread <= 0.01.
        """
        h, w = img.shape[:2]
        L = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        k = max(31, min(151, (min(h, w) // 8) | 1))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        envelope = cv2.morphologyEx(L, cv2.MORPH_CLOSE, se)

        b = max(8, min(h, w) // 40)
        corners = [
            float(np.mean(envelope[:b, :b])), float(np.mean(envelope[:b, -b:])),
            float(np.mean(envelope[-b:, :b])), float(np.mean(envelope[-b:, -b:])),
        ]
        center = float(np.mean(
            envelope[h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b]
        ))
        denom = max(center, 1e-6)
        corner_deficit = (center - float(np.mean(corners))) / denom
        corner_spread = (max(corners) - min(corners)) / denom

        # Low-order field for the correction: quadratic surface fit to the
        # envelope (downsampled to 128x128 for a stable, tiny least-squares).
        small = cv2.resize(envelope, (128, 128), interpolation=cv2.INTER_AREA)
        ys, xs = np.mgrid[0:128, 0:128]
        xn, yn = xs / 127.0, ys / 127.0
        A = np.column_stack([
            np.ones(128 * 128), xn.ravel(), yn.ravel(),
            (xn * xn).ravel(), (yn * yn).ravel(), (xn * yn).ravel(),
        ])
        z = small.ravel().astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        field = cv2.resize(
            (A @ coef).reshape(128, 128).astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        return {
            "corner_deficit": corner_deficit,
            "corner_spread": corner_spread,
            "field": field,
        }

    @classmethod
    def correct_illumination(cls, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """T-5: flatten scanner illumination falloff. Returns (image, info).

        Gated: the image is returned BITWISE unchanged unless the background
        envelope shows a SYMMETRIC corner deficit >= 20% (all four corners
        down together, spread <= 12%) — the signature of a lamp/vignette
        defect. Content gradients (portraits, landscapes, bands) are
        asymmetric and never trigger. The correction divides each channel by
        the fitted quadratic field, clamped so no pixel is brightened by more
        than MAX_CORRECTION (2x) — preserving absolute exposure rather than
        equalizing it away.

        Known, accepted limitation: one-sided lamp falloff is asymmetric and
        reads like content; this gate will not fire on it. False-negative is
        the safe direction (no content damage); false-positive is the
        destructive one.
        """
        info: dict = {"applied": False}
        try:
            m = cls.estimate_illumination_deficit(img)
            deficit = m["corner_deficit"]
            spread = m["corner_spread"]
            info.update({
                "corner_deficit": round(float(deficit), 3),
                "corner_spread": round(float(spread), 3),
            })
            if deficit < cls.MIN_CORNER_DEFICIT or spread > cls.MAX_CORNER_SPREAD:
                return img, info

            field = m["field"]
            gain = np.clip(field.max() / np.maximum(field, 1e-6), 1.0, cls.MAX_CORRECTION)
            corrected = np.clip(
                img.astype(np.float32) * gain[..., None], 0, 255
            ).astype(np.uint8)
            info.update({"applied": True, "max_gain": round(float(gain.max()), 3)})
            return corrected, info
        except Exception as exc:  # pragma: no cover - defensive
            info["error"] = str(exc)
            return img, info

