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


def horizontal_binning(Z: np.ndarray, fac: int = 2, dim: int = 1) -> np.ndarray:
    """
    Horizontal binning of the image Z

    Parameters
    ----------
    Z : np.ndarray
        The image to be binned.
    fac : int
        binning factor
    dim : direction X=0, Y=1
    Returns
    -------
    np.ndarray
        binned image

    """

    if fac > 1:
        fac_half = fac // 2
        binned_zs = create_array(Z, fac, dim)

        # get coordinates of bin centres
        if dim == 0:
            binned_zs = binned_zs[
                fac_half + ((fac % 2) == 1) : binned_zs.shape[dim] - fac_half + 1 : fac,
                :,
            ]
        else:
            binned_zs = binned_zs[
                :,
                fac_half + ((fac % 2) == 1) : binned_zs.shape[dim] - fac_half + 1 : fac,
            ]

        return binned_zs

    return Z


def horizontal_debinning(
    Z: np.ndarray, size: int, fac: int, n_iter: int, dim: int = 1
) -> np.ndarray:
    """
    Horizontal debinning of the image Z into the same shape as Z_target.

    Parameters
    ----------
    Z : np.ndarray
        The image to be debinned.
    size: target size (original size before binning) for the second dimension
    fac: binning factor (original divisor)
    n_iter: number of iterations
    dim: dimension for binning (Y = 0 or X = 1)

    Returns
    -------
    np.ndarray
        The debinned image.
    """
    if fac <= 1:
        return np.copy(Z)

    fac_half = fac // 2

    if dim == 0:
        base_array = np.ones((size, 1))
    else:
        base_array = np.ones((1, size))

    n_counter = create_array(base_array, fac, dim)

    # coordinates of bin counts
    x1c = np.arange(fac_half + ((fac % 2) == 1), (Z.shape[dim]) * fac, fac)
    x1 = np.arange(fac_half + 1 - ((fac % 2) == 0) / 2, (Z.shape[dim]) * fac, fac)

    # coordinates of image pixels
    ix1 = np.arange(1, size + 1)

    interpolated_image = 0

    for j in range(max(1, n_iter)):
        # residual
        if j > 0:
            residual = Z - horizontal_binning(interpolated_image, fac, dim)
        else:
            residual = Z

        # interpolation
        if dim == 0:
            interp = interp1d(
                x1,
                residual / n_counter[x1c, :],
                kind="cubic",
                fill_value="extrapolate",
                axis=0,
            )
        else:
            interp = interp1d(
                x1, residual / n_counter[:, x1c], kind="cubic", fill_value="extrapolate"
            )
        interpolated_image = interpolated_image + interp(ix1)

    return interpolated_image


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
