#!/usr/bin/env python3
"""Utility functions for BM3DORNL."""

import numpy as np
from numba import njit
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from scipy.ndimage import zoom


@njit
def is_within_threshold(
    ref_patch: np.ndarray, cmp_patch: np.ndarray, intensity_diff_threshold: float
) -> bool:
    """
    Determine if the Euclidean distance between two patches is less than a specified threshold.

    This function computes the Euclidean distance between two patches and checks if it is less than the provided
    intensity difference threshold. It is optimized with Numba's JIT in nopython mode to ensure high performance.

    Parameters
    ----------
    ref_patch : np.ndarray
        The reference patch as a flattened array of intensities.
    cmp_patch : np.ndarray
        The comparison patch as a flattened array of intensities.
    intensity_diff_threshold : float
        The threshold below which the Euclidean distance between the patches is considered sufficiently small
        for the patches to be deemed similar.

    Returns
    -------
    bool
        True if the Euclidean distance between `ref_patch` and `cmp_patch` is less than `intensity_diff_threshold`;
        otherwise, False.

    Example:
    --------
    >>> ref_patch = np.array([1, 2, 3])
    >>> cmp_patch = np.array([1, 2, 5])
    >>> threshold = 2.5
    >>> is_within_threshold(ref_patch, cmp_patch, threshold)
    False
    """
    return np.linalg.norm(ref_patch - cmp_patch) <= intensity_diff_threshold


def estimate_background_intensity(
    tomostack: np.ndarray,
    quantile: float = 0.2,  # 20% quantile, can estimate from visual inspection
) -> float:
    """
    Estimate the background intensity from the tomostack.

    Parameters
    ----------
    tomostack : np.ndarray
        The tomostack to estimate the background intensity.
    quantile : float, optional
        The quantile to estimate the background intensity, by default 0.05.

    Returns
    -------
    float
        The estimated background intensity.
    """
    # if tomostack is 3D, average along axis 1 (the row axis)
    if tomostack.ndim == 3:
        tomostack = np.mean(tomostack, axis=1)
    return np.quantile(tomostack, quantile)


def downscale_2d_horizontal(array: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Downscale a 2D array horizontally by the given factor,
    mimicking the original horizontal binning approach.

    Parameters
    ----------
    array : np.ndarray
        The 2D array to downscale horizontally.
    scale_factor : int
        The factor by which to downscale the array horizontally.

    Returns
    -------
    np.ndarray
        The horizontally downscaled 2D array.
    """
    _, cols = array.shape

    # Create and apply the Gaussian convolution kernel
    kernel = gaussian(scale_factor, std=scale_factor / 2).reshape(1, -1)
    kernel /= kernel.sum()  # Normalize the kernel
    binned = convolve2d(array, kernel, mode="same")

    # Calculate the size of the downscaled array
    new_cols = (cols + scale_factor - 1) // scale_factor

    # Subsample at bin centers
    start = (scale_factor - 1) // 2
    return binned[:, start::scale_factor][:, :new_cols]


def upscale_2d_horizontal(
    input_array: np.ndarray,
    scale_factor: int,
    original_width: int,
    use_iterative_refinement: bool = False,
    refinement_iterations: int = 3,
) -> np.ndarray:
    """
    Upscale a 2D array horizontally using RectBivariateSpline interpolation, with optional iterative refinement.

    Parameters
    ----------
    input_array : np.ndarray
        The 2D input array to upscale horizontally.
    scale_factor : int
        The factor by which to upscale horizontally.
    original_width : int
        The width of the original array before downscaling.
    use_iterative_refinement : bool, optional
        Whether to use iterative residual correction (default is False).
    refinement_iterations : int, optional
        Number of refinement iterations if using iterative refinement (default is 3).

    Returns
    -------
    np.ndarray
        The horizontally upscaled 2D array.
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive integer.")

    input_array = input_array.astype(np.float64)  # Ensure consistent type
    zoom_factor = original_width / input_array.shape[1]
    upscaled_array = zoom(input_array, (1, zoom_factor), order=3)

    if use_iterative_refinement:
        for _ in range(refinement_iterations):
            # Downscale the current upscaled array
            downscaled = downscale_2d_horizontal(upscaled_array, scale_factor)

            # Calculate the residual between the original input and the downscaled version
            residual = input_array - downscaled

            # Upscale the residual and add it back to the upscaled array
            upscaled_residual = zoom(residual, (1, zoom_factor), order=3)
            upscaled_array += upscaled_residual

    return upscaled_array
