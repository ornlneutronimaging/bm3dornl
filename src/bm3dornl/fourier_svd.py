"""Remove vertical streaks with frequency detection and matrix decomposition."""

import numpy as np
import logging

try:
    from .bm3d_rust import (
        fourier_svd_removal_rust,
        fourier_svd_removal_rust_f64
    )
except ImportError:
    # During development/test, might not be installed yet
    logging.warning("bm3d_rust module not found. Fourier-SVD will fail.")
    fourier_svd_removal_rust = None
    fourier_svd_removal_rust_f64 = None

logger = logging.getLogger(__name__)

def fourier_svd_removal(
    sinogram: np.ndarray,
    fft_alpha: float = 1.0,
    notch_width: float = 2.0,
) -> np.ndarray:
    """
    Remove vertical streaks with Fourier-SVD processing.

    The first stage detects vertical-frequency energy with a fast Fourier
    transform (FFT). The second stage uses rank-one singular value decomposition
    (SVD) to estimate and subtract the streak pattern.

    Parameters
    ----------
    sinogram : np.ndarray
        Two-dimensional sinogram. Float32 and float64 use matching backends.
        Other dtypes are processed as float32.
    fft_alpha : float, optional
        Weight given to detected vertical-frequency energy. Higher values make
        that energy influence the removal threshold more strongly. Set 0.0 to
        use fixed thresholds. Default: 1.0.
    notch_width : float, optional
        Gaussian notch width in frequency bins. Larger values include more
        frequencies away from the vertical axis. Default: 2.0.

    Returns
    -------
    np.ndarray
        Destriped sinogram with the input shape and dtype. Other dtypes
        are converted back to their original dtype.
    """
    if sinogram.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {sinogram.shape}")

    input_dtype = sinogram.dtype

    # Check dimensions
    rows, cols = sinogram.shape
    if rows < 10 or cols < 10:
        logger.warning("Image too small for Fourier-SVD streak removal. Returning input.")
        return sinogram.copy()

    # Dispatch based on dtype
    if input_dtype == np.float32:
        if fourier_svd_removal_rust is None:
            raise ImportError("bm3d_rust backend not available")
        return fourier_svd_removal_rust(sinogram, fft_alpha, notch_width)

    elif input_dtype == np.float64:
        if fourier_svd_removal_rust_f64 is None:
            raise ImportError("bm3d_rust backend not available")
        return fourier_svd_removal_rust_f64(sinogram, fft_alpha, notch_width)

    else:
        # Auto-convert other types to float32
        logger.info(f"Converting input from {input_dtype} to float32 for processing")
        sino_f32 = sinogram.astype(np.float32)
        result = fourier_svd_removal_rust(sino_f32, fft_alpha, notch_width)
        return result.astype(input_dtype)
