import io
import numpy as np
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

