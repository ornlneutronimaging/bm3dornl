#!/usr/bin/env python3
"""Utility functions for BM3DORNL."""

import numpy as np
from numba import njit
from scipy.signal import convolve2d
from scipy.interpolate import interp1d


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
    base_arr: Input array
           h: bin count
         dim: bin dimension (0 or 1)
    Returns
    -------
    resulting array
    """
    mod_pad = h - ((base_arr.shape[dim] - 1) % h) - 1
    if dim == 0:
        pads = ((0, mod_pad), (0, 0))
        kernel = np.ones((h, 1))
    else:
        pads = ((0, 0), (0, mod_pad))
        kernel = np.ones((1, h))

    return convolve2d(np.pad(base_arr, pads, "symmetric"), kernel, "same", "fill")


def horizontal_binning(z: np.ndarray, h: int = 2, dim: int = 1) -> np.ndarray:
    """
    Horizontal binning of the image Z

    Parameters
    ----------
    Z : np.ndarray
        The image to be binned.
    h : int
        devisor or binning
    dim : direction
    Returns
    -------
    np.ndarray
        binned image

    """

    if h > 1:
        h_half = h // 2
        z_bin = create_array(z, h, dim)

        # get coordinates of bin centres
        if dim == 0:
            z_bin = z_bin[
                h_half + ((h % 2) == 1) : z_bin.shape[dim] - h_half + 1 : h, :
            ]
        else:
            z_bin = z_bin[
                :, h_half + ((h % 2) == 1) : z_bin.shape[dim] - h_half + 1 : h
            ]

        return z_bin

    return z


def horizontal_debinning(
    z: np.ndarray, size: int, h: int, n_iter: int, dim: int = 1
) -> np.ndarray:
    """
    Horizontal debinning of the image Z into the same shape as Z_target.

    Parameters
    ----------
    z : np.ndarray
        The image to be debinned.
    size: target size (original size before binning) for the second dimension
    h: binning factor (original divisor)
    n_iter: number of iterations
    dim: dimension for binning (0 or 1)

    Returns
    -------
    np.ndarray
        The debinned image.
    """
    if h <= 1:
        return np.copy(z)

    h_half = h // 2

    if dim == 0:
        base_arr = np.ones((size, 1))
    else:
        base_arr = np.ones((1, size))

    n_counter = create_array(base_arr, h, dim)

    # coordinates of bin counts
    x1c = np.arange(h_half + ((h % 2) == 1), (z.shape[dim]) * h, h)
    x1 = np.arange(h_half + 1 - ((h % 2) == 0) / 2, (z.shape[dim]) * h, h)

    # coordinates of image pixels
    ix1 = np.arange(1, size + 1)

    y_j = 0

    for jj in range(max(1, n_iter)):
        # residual
        if jj > 0:
            r_j = z - horizontal_binning(y_j, h, dim)
        else:
            r_j = z

        # interpolation
        if dim == 0:
            interp = interp1d(
                x1,
                r_j / n_counter[x1c, :],
                kind="cubic",
                fill_value="extrapolate",
                axis=0,
            )
        else:
            interp = interp1d(
                x1, r_j / n_counter[:, x1c], kind="cubic", fill_value="extrapolate"
            )
        y_j = y_j + interp(ix1)

    return y_j


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
