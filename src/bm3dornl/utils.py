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
    Estimate noise standard deviation using MAD-based robust estimation.

    This implements sigma estimation from Makinen et al. (2021), optimized for
    detecting vertical streak noise in sinograms. The image is filtered to isolate
    vertical streaks (vertical Gaussian + horizontal High-pass), then MAD (Median
    Absolute Deviation) is computed and scaled.

    This is primarily a diagnostic tool for advanced users who want to understand
    the noise characteristics of their data or tune denoising parameters.

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram (2D array). Supported types: float32, float64.

    Returns
    -------
    float
        Estimated noise standard deviation (sigma).

    Examples
    --------
    >>> import numpy as np
    >>> from bm3dornl.utils import estimate_noise_sigma
    >>> sinogram = np.random.randn(256, 512).astype(np.float32) * 0.1
    >>> sigma = estimate_noise_sigma(sinogram)
    >>> print(f"Estimated sigma: {sigma:.4f}")  # Should be close to 0.1
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
