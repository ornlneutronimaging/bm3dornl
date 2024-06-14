#!/usr/bin/env python3
"""Module for noise analysis."""

import numpy as np
from bm3dornl.signal import hadamard_transform


def estimate_noise_psd(noisy_image: np.ndarray) -> np.ndarray:
    """
    Estimate the noise power spectral density (PSD) from a given noisy image.

    Parameters
    ----------
    noisy_image : np.ndarray
        The noisy image from which to estimate the noise PSD.

    Returns
    -------
    np.ndarray
        The estimated noise PSD.
    """
    # Compute the Fourier transform of the noisy image
    noisy_image_fft = np.fft.fft2(noisy_image)

    # Estimate the noise power spectral density (PSD)
    noise_psd = np.abs(noisy_image_fft) ** 2 / noisy_image.size

    return noise_psd


def get_exact_noise_variance_fft(patches: np.ndarray) -> np.ndarray:
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


def get_exact_noise_variance(transformed_patches: np.ndarray) -> np.ndarray:
    """Estimate the noise variance of each transformed patch in the input array.

    Parameters
    ----------
    transformed_patches : np.ndarray
        Input array with shape (N, H, W), where N is the number of patches and H, W are the patch dimensions.

    Returns
    -------
    np.ndarray
        Array with shape (N, H, W) containing the estimated noise variance of each transformed patch.
    """
    patches_psd = np.abs(transformed_patches) ** 2
    patches_psd_mean = np.mean(patches_psd, axis=0)
    return patches_psd * patches_psd_mean[np.newaxis, ...]


def get_exact_noise_variance_hadamard(patches: np.ndarray) -> np.ndarray:
    """
    Estimate the noise variance of each patch in the input array using the Hadamard transform.

    Parameters
    ----------
    patches : np.ndarray
        Input array with shape (N, H, W), where N is the number of patches and H, W are the patch dimensions.

    Returns
    -------
    np.ndarray
        Array with shape (N, H, W) containing the estimated noise variance of each patch.
    """
    hadamard_patches = np.array([hadamard_transform(patch) for patch in patches])
    patches_psd = np.abs(hadamard_patches) ** 2
    patches_psd_mean = np.mean(patches_psd, axis=0)
    variances = patches_psd * patches_psd_mean[np.newaxis, ...]
    return variances
