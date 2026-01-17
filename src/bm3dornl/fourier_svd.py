"""
Fourier-SVD Streak Removal

A two-stage algorithm for removing vertical streak artifacts from sinograms:

Stage 1: FFT-Guided Energy Detection
- Apply FFT and isolate vertical frequencies (Fy ≈ 0) using Gaussian notch filter
- Compute per-column energy profile from isolated streak spectrum
- Use energy profile to spatially modulate removal threshold

Stage 2: Rank-1 SVD with Magnitude Gating
- Extract first principal component via power iteration
- Median filter the v-vector to separate baseline from streak detail
- Apply soft magnitude gating: 1 / (1 + (|v| / threshold)^exponent)
- Reconstruct streak as rank-1 outer product and subtract from input

Parameters:
- fft_alpha: Controls FFT energy influence on threshold modulation (default: 1.0)
- notch_width: Gaussian notch filter width in frequency domain (default: 2.0)
"""

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
    Fourier-SVD Streak Removal.

    A two-stage algorithm combining FFT-based energy detection with rank-1 SVD
    for removing vertical streak artifacts from sinograms.

    Benchmark Performance:
    - Speed: ~2.6x faster than BM3D.
    - Low SNR: Superior limits (cleaner noise floor).
    - High SNR: Excellent structure preservation (avoids wall attack).

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram (2D array). Supported types: float32, float64.
    fft_alpha : float, optional
        FFT-guided trust factor for adaptive thresholding. Higher values increase
        sensitivity to vertical energy in the frequency domain. Set to 0.0 to disable
        FFT-guided gating and use fixed thresholds. Default: 1.0.
    notch_width : float, optional
        Width of the Gaussian notch filter in frequency domain (in pixels). Controls
        the selectivity of the vertical frequency isolation. Larger values accept
        more off-axis frequencies. Default: 2.0.

    Returns
    -------
    np.ndarray
        Destriped sinogram (same shape/type).
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
