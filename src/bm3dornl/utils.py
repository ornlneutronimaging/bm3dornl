#!/usr/bin/env python3
"""Utility functions for BM3DORNL."""

import numpy as np
from numba import njit
from scipy.signal import convolve2d
from scipy.interpolate import RectBivariateSpline


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


def create_array(base_arr: np.ndarray, h: int, dim: int):
    """
    Create a padded and convolved array used in both binning and debinning.

    Parameters
    ----------
    base_arr: np.ndarray
        Input array
    h: int
        bin count
    dim: int
        bin dimension (0 or 1)

    Returns
    -------
    np.ndarray
        The resulting array
    """
    mod_pad = h - ((base_arr.shape[dim] - 1) % h) - 1
    if dim == 0:
        pads = ((0, mod_pad), (0, 0))
        kernel = np.ones((h, 1))
    else:
        pads = ((0, 0), (0, mod_pad))
        kernel = np.ones((1, h))

    return convolve2d(np.pad(base_arr, pads, "symmetric"), kernel, "same", "fill")


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

    # Create and apply the convolution kernel
    kernel = np.ones((1, scale_factor)) / scale_factor
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
    rows, cols = input_array.shape
    x = np.arange(cols)
    y = np.arange(rows)
    x_upscaled = np.linspace(0, cols - 1, original_width)

    # Initial upscaling
    interp_func = RectBivariateSpline(y, x, input_array)
    upscaled_array = interp_func(y, x_upscaled)

    if use_iterative_refinement:
        for _ in range(refinement_iterations):
            # Downscale the current estimate
            downscaled = downscale_2d_horizontal(upscaled_array, scale_factor)

            # Calculate residual
            residual = input_array - downscaled

            # Upscale the residual
            residual_interp = RectBivariateSpline(y, x, residual)
            upscaled_residual = residual_interp(y, x_upscaled)

            # Add the upscaled residual to the current estimate
            upscaled_array += upscaled_residual

    return upscaled_array
