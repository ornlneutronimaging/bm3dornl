"""Utility functions for diagnostics and advanced use."""

import numpy as np
import logging

try:
    from .bm3d_rust import (
        estimate_noise_sigma_rust,
        estimate_noise_sigma_rust_f64,
    )
except ImportError:
    logging.warning("bm3d_rust module not found. Noise estimation will fail.")
    estimate_noise_sigma_rust = None
    estimate_noise_sigma_rust_f64 = None

logger = logging.getLogger(__name__)


def estimate_noise_sigma(sinogram: np.ndarray) -> float:
    """
    Estimate the amplitude of vertical streak noise in a sinogram.

    This implements the estimator from Mäkinen et al. (2021). It first smooths
    vertically with a Gaussian width of ``height / 12``. This preserves streaks
    while suppressing pixel noise. A Daubechies-3 high-pass filter then isolates
    changes between columns.

    The result is 1.4826 times the median absolute deviation (MAD). MAD is the
    median distance from the filtered values' median. The result measures vertical
    streak amplitude, not pixel-level standard deviation. Vertical smoothing
    removes most independent pixel noise. Its estimate is therefore much
    smaller than that noise's standard deviation.

    The BM3D pipeline uses this estimator when ``sigma_random`` is at or
    below 1e-6, for example 0.0. Streak
    modes estimate after subtracting the streak profile. Generic mode estimates
    on normalized input. Use this function to inspect streak strength or choose
    ``sigma_random`` manually.

    Parameters
    ----------
    sinogram : np.ndarray
        Two-dimensional sinogram. Float32 and float64 use matching backends.
        Other dtypes are converted to float32.

    Returns
    -------
    float
        Estimated amplitude of the vertical streak noise.

    Examples
    --------
    >>> import numpy as np
    >>> from bm3dornl.utils import estimate_noise_sigma
    >>> rng = np.random.default_rng(42)
    >>> clean = np.ones((256, 512), dtype=np.float32)
    >>> streaks = rng.normal(0.0, 0.2, size=(1, 512)).astype(np.float32)
    >>> sigma = estimate_noise_sigma(clean + streaks)  # true streak sigma: 0.2
    >>> 0.15 < sigma < 0.25
    True
    """
    if sinogram.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {sinogram.shape}")

    input_dtype = sinogram.dtype

    if input_dtype == np.float32:
        if estimate_noise_sigma_rust is None:
            raise ImportError("bm3d_rust backend not available")
        return float(estimate_noise_sigma_rust(sinogram))

    elif input_dtype == np.float64:
        if estimate_noise_sigma_rust_f64 is None:
            raise ImportError("bm3d_rust backend not available")
        return float(estimate_noise_sigma_rust_f64(sinogram))

    else:
        # Auto-convert other types to float32
        logger.info(f"Converting input from {input_dtype} to float32 for processing")
        sino_f32 = sinogram.astype(np.float32)
        if estimate_noise_sigma_rust is None:
            raise ImportError("bm3d_rust backend not available")
        return float(estimate_noise_sigma_rust(sino_f32))


def compute_cdf(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the cumulative distribution function of an image.

    Useful for comparing intensity distributions before and after
    ring-artifact removal.

    Parameters
    ----------
    img : np.ndarray
        The input image.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The sorted CDF values and the corresponding probabilities.
    """
    cdf_org_sorted = np.sort(img.flatten())
    p_org = 1.0 * np.arange(len(cdf_org_sorted)) / (len(cdf_org_sorted) - 1)
    return cdf_org_sorted, p_org
