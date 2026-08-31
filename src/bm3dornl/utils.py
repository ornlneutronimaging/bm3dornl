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
    Estimate the standard deviation of vertical streak noise in a sinogram.

    This implements the sigma estimation from Makinen et al. (2021). The image
    is smoothed with a tall vertical Gaussian (sigma = height / 12), which
    passes column-wise (vertical streak) structure but suppresses pixel-level
    random noise, then high-pass filtered horizontally (Daubechies-3) to
    isolate column-to-column variation. The scaled median absolute deviation
    (1.4826 * MAD) of the filtered result is returned.

    The returned value is therefore the amplitude of vertical streaks — the
    sinogram signature of ring artifacts — not the pixel-level standard
    deviation of the image. For purely independent (i.i.d.) pixel noise the
    vertical smoothing removes most of what the filter measures, and the
    result lands far below the pixel-level sigma (roughly 8x smaller for a
    256-row image; the taller the image, the stronger the suppression).

    This is the same estimator the BM3D pipeline runs internally to fill in
    ``sigma_random`` when it is set to 0.0 (in the streak-removal modes that
    estimate is taken after the streak profile has been subtracted; in
    generic mode, on the normalized input). As a standalone diagnostic it is
    useful for judging streak strength or for choosing ``sigma_random``
    manually.

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram (2D array). Supported types: float32, float64.

    Returns
    -------
    float
        Estimated standard deviation of the vertical streak noise.

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
