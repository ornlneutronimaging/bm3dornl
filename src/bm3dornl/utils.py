#!/usr/bin/env python3
"""Utility functions for BM3DORNL."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from numba import jit
from typing import Tuple, List
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter


@jit(nopython=True)
def find_candidate_patch_ids(
    signal_patches: np.ndarray, ref_index: int, cut_off_distance: Tuple
) -> List:
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


@jit(nopython=True)
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


@jit(nopython=True)
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
        padded = np.tile(candidate_patch_ids, repeats)[:num_patches]
        return padded
    elif mode == "circular":
        extended = np.tile(candidate_patch_ids, ((num_patches // current_length) + 1))[
            :num_patches
        ]
        return extended
    elif mode == "mirror":
        mirror_length = min(current_length, num_patches - current_length)
        mirrored_part = candidate_patch_ids[:mirror_length][::-1]
        return np.concatenate([candidate_patch_ids, mirrored_part])
    elif mode == "random":
        random_padded = np.random.choice(candidate_patch_ids, num_patches, replace=True)
        return random_padded
    else:
        raise ValueError("Unknown padding mode specified.")

    return np.concatenate([candidate_patch_ids, padding])


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


def estimate_noise_std(
    noisy_image: np.ndarray,
    noise_free_image: np.ndarray,
) -> float:
    """
    Estimate the noise standard deviation from a pair of noisy and noise-free images.

    Parameters
    ----------
    noisy_image : np.ndarray
        A 2D NumPy array representing the noisy image.
    noise_free_image : np.ndarray
        A 2D NumPy array representing the noise-free image.

    Returns
    -------
    float
        The estimated noise standard deviation.
    """
    # Rescale both images to [0, 255]
    noisy_image_min, noisy_image_max = noisy_image.min(), noisy_image.max()
    noise_free_image_min, noise_free_image_max = (
        noise_free_image.min(),
        noise_free_image.max(),
    )

    noisy_image = (
        (noisy_image - noisy_image_min) / (noisy_image_max - noisy_image_min) * 255
    )
    noise_free_image = (
        (noise_free_image - noise_free_image_min)
        / (noise_free_image_max - noise_free_image_min)
        * 255
    )

    # calculate the noise standard deviation
    return np.std(np.abs(noisy_image - noise_free_image))


def estimate_noise_free_sinogram(
    sinogram: np.ndarray,
    background_estimate: float,
    sigma_gaussian: float = 5.0,
) -> np.ndarray:
    """
    Estimate noise-free sinogram from noisy sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        Noisy sinogram.
    background_estimate : float
        Background estimate value.
    sigma_gaussian : float, optional
        Standard deviation of the 1D Gaussian filter, by default 5.0.

    Returns
    -------
    np.ndarray
        Noise-free sinogram.
    """
    # Perform the hard-thresholding using FFT
    sinogram_fft_shifted = fftshift(fft2(sinogram))
    mask = np.ones_like(sinogram_fft_shifted)
    crow = sinogram_fft_shifted.shape[0] // 2
    mask[crow] = (
        0  # this will suppress all vertical streaks, and some features (demerit)
    )
    sinogram_fft_shifted *= mask
    sinogram_filtered = ifft2(ifftshift(sinogram_fft_shifted)).real

    # Renormalize the sinogram to [0, 1] as the hard threshold mess up the intensity distribution
    sinogram_filtered -= sinogram_filtered.min()
    sinogram_filtered /= sinogram_filtered.max()

    # Now reapply the background
    sinogram_filtered[sinogram < background_estimate] = 0

    sino_blurred = gaussian_filter(sinogram, sigma=sigma_gaussian)
    scale_profile = np.sum(sinogram_filtered, axis=0) / np.sum(sino_blurred, axis=0)
    sinogram_filtered /= scale_profile + 1e-8

    # renormalize the sinogram to [0, 1]
    sinogram_filtered -= sinogram_filtered.min()
    sinogram_filtered /= sinogram_filtered.max()

    return sinogram_filtered
