import numpy as np

def calculate_psnr(img1: np.ndarray, img2: np.ndarray):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between two images.
    Returns None if scikit-image is not installed.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        return float(psnr(img1, img2, data_range=255))
    except ImportError:
        return None

def calculate_ssim(img1: np.ndarray, img2: np.ndarray):
    """
    Calculates Structural Similarity Index (SSIM) between two images.
    Returns None if scikit-image is not installed.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        multichannel = True if len(img1.shape) == 3 else False
        channel_axis = -1 if multichannel else None
        return float(ssim(img1, img2, channel_axis=channel_axis, data_range=255))
    except ImportError:
        return None
