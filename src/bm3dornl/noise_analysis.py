"""Module for noise analysis."""

#!/usr/bin/env python3
import numpy as np


def get_exact_noise_variance(patches: np.ndarray) -> np.ndarray:
    """Estimate the noise variance of each patch in the input array.

    Parameters
    ----------
    patches : np.ndarray
        Input array with shape (N, H, W), where N is the number of patches and H, W are the patch dimensions.

    Returns
    -------
    np.ndarray
        Array with shape (N, H, W) containing the estimated noise variance of each patch.
    """
    patches_psd = np.abs(np.fft.fftn(patches, axes=(-2, -1))) ** 2
    patches_psd_mean = np.mean(patches_psd, axis=0)
    return patches_psd * patches_psd_mean[np.newaxis, ...]
