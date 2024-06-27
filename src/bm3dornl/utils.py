#!/usr/bin/env python3
"""Utility functions for BM3DORNL."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from numba import njit
from typing import Tuple, List
from scipy.signal import convolve2d
from scipy.interpolate import interp1d


@njit
def find_candidate_patch_ids(
    signal_patches: np.ndarray, ref_index: int, cut_off_distance: Tuple
) -> List[int]:
    """
    Identify candidate patch indices that are within the specified Manhattan distance from a reference patch.

    This function computes a list of indices for patches that are within a given row and column distance from
    the reference patch specified by `ref_index`. It only considers patches that have not been compared previously
    (i.e., patches that are ahead of the reference patch in the list, ensuring the upper triangle of the comparison matrix).

    Parameters
    ----------
    signal_patches : np.ndarray
        Array containing the positions of all signal patches. Each position is represented as (row_index, column_index).
    ref_index : int
        Index of the reference patch in `signal_patches` for which candidates are being sought.
    cut_off_distance : tuple
        A tuple (row_dist, col_dist) specifying the maximum allowed distances in the row and column directions.

    Returns
    -------
    list
        A list of integers representing the indices of the candidate patches in `signal_patches` that are within
        the `cut_off_distance` from the reference patch and are not previously compared (ensuring upper triangle).
    """
    num_patches = signal_patches.shape[0]
    ref_pos = signal_patches[ref_index]
    candidate_patch_ids = [ref_index]

    for i in range(ref_index + 1, num_patches):  # Ensure only checking upper triangle
        if (
            np.abs(signal_patches[i, 0] - ref_pos[0]) <= cut_off_distance[0]
            and np.abs(signal_patches[i, 1] - ref_pos[1]) <= cut_off_distance[1]
        ):
            candidate_patch_ids.append(i)

    return candidate_patch_ids


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


@njit
def get_signal_patch_positions(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    background_threshold: float = 0.1,
) -> np.ndarray:
    """Segment an image into signal patches.

    Parameters
    ----------
    image : np.ndarray
        The input image to be segmented into patches.
    patch_size : Tuple[int, int]
        The size of the patches to be extracted.
    stride : int
        The stride for patch extraction.
    background_threshold : float
        The threshold for determining background patches.

    Returns
    -------
    signal_patches : np.ndarray
        An array of positions of signal patches.

    NOTE
    ----
    Numba has issues with return a Tuple of np.ndarray, and since we only operates on signal patches,
    we will ignore the background patches for now.
    """
    i_height, i_width = image.shape
    p_height, p_width = patch_size

    signal_patches = []

    for r in range(0, i_height - p_height + 1, stride):
        for c in range(0, i_width - p_width + 1, stride):
            patch = image[r : r + p_height, c : c + p_width]
            patch_max = np.max(patch)
            if patch_max >= background_threshold:
                signal_patches.append((r, c))

    # deal with empty list
    # Note: raise error when couldn't find a single signal patch from the entire
    #       sinogram, which usually indicating a bad background estimation.
    if len(signal_patches) == 0:
        raise ValueError(
            "Couldn't find any signal patches in the image! Please check the background threshold."
        )

    return np.array(signal_patches)


@njit
def pad_patch_ids(
    candidate_patch_ids: np.ndarray,
    num_patches: int,
    mode: str = "circular",
) -> np.ndarray:
    """
    Pad the array of patch IDs to reach a specified length using different strategies.

    Parameters
    ----------
    candidate_patch_ids : np.ndarray
        Array of patch indices identified as candidates.
    num_patches : int
        Desired number of patches in the padded list.
    mode : str
        Padding mode, options are 'first', 'repeat_sequence', 'circular', 'mirror', 'random'.

    Returns
    -------
    np.ndarray
        Padded array of patch indices.
    """
    current_length = len(candidate_patch_ids)
    if current_length >= num_patches:
        return candidate_patch_ids[:num_patches]

    if mode == "first":
        padding = np.full((num_patches - current_length,), candidate_patch_ids[0])
    elif mode == "repeat_sequence":
        repeats = (num_patches // current_length) + 1
        padded = np.empty(num_patches, dtype=candidate_patch_ids.dtype)
        index = 0
        for _ in range(repeats):
            for candidate in candidate_patch_ids:
                if index >= num_patches:
                    break
                padded[index] = candidate
                index += 1
        return padded
    elif mode == "circular":
        padded = np.empty(num_patches, dtype=candidate_patch_ids.dtype)
        for i in range(num_patches):
            padded[i] = candidate_patch_ids[i % current_length]
        return padded
    elif mode == "mirror":
        extended = np.concatenate((candidate_patch_ids, candidate_patch_ids[::-1]))
        padded = extended[:num_patches]
        return padded
    elif mode == "random":
        padding = np.random.choice(candidate_patch_ids, num_patches - current_length)
    return np.concatenate((candidate_patch_ids, padding))


def horizontal_binning1(Z: np.ndarray, k: int = 0) -> list[np.ndarray]:
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


def horizontal_debinning1(original_image: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def create_array(base_arr: np.ndarray, h: int, dim: int):
    """
    Create a padded and convolved array used in both binning and debinning.
    :param base_arr: Input array
    :param h: bin count
    :param dim: bin dimension (0 or 1)
    :return: resulting array
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
