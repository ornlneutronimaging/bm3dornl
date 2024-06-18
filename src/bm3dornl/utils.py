#!/usr/bin/env python3
"""Utility functions for BM3DORNL."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from numba import njit


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


def horizontal_binning(Z: np.ndarray, k: int = 0) -> list[np.ndarray]:
    """
    Horizontal binning of the image Z into a list of k images.

    Parameters
    ----------
    Z : np.ndarray
        The image to be binned.
    k : int
        Number of iterations to bin the image by half.

    Returns
    -------
    list[np.ndarray]
        List of k images.

    Example
    -------
    >>> Z = np.random.rand(64, 64)
    >>> binned_zs = horizontal_binning(Z, 3)
    >>> len(binned_zs)
    4
    """
    binned_zs = [Z]
    for _ in range(k):
        sub_z0 = Z[:, ::2]
        sub_z1 = Z[:, 1::2]
        # make sure z0 and z1 have the same shape
        if sub_z0.shape[1] > sub_z1.shape[1]:
            sub_z0 = sub_z0[:, :-1]
        elif sub_z0.shape[1] < sub_z1.shape[1]:
            sub_z1 = sub_z1[:, :-1]
        # average z0 and z1
        Z = (sub_z0 + sub_z1) * 0.5
        binned_zs.append(Z)
    return binned_zs


def horizontal_debinning(original_image: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Horizontal debinning of the image Z into the same shape as Z_target.

    Parameters
    ----------
    original_image : np.ndarray
        The image to be debinned.
    target : np.ndarray
        The target image to match the shape.

    Returns
    -------
    np.ndarray
        The debinned image.

    Example
    -------
    >>> Z = np.random.rand(64, 64)
    >>> target = np.random.rand(64, 128)
    >>> debinned_z = horizontal_debinning(Z, target)
    >>> debinned_z.shape
    (64, 128)
    """
    # Original dimensions
    original_height, original_width = original_image.shape
    # Target dimensions
    new_height, new_width = target.shape

    # Original grid
    original_x = np.arange(original_width)
    original_y = np.arange(original_height)

    # Target grid
    new_x = np.linspace(0, original_width - 1, new_width)
    new_y = np.linspace(0, original_height - 1, new_height)

    # Spline interpolation
    spline = RectBivariateSpline(original_y, original_x, original_image)
    interpolated_image = spline(new_y, new_x)

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
