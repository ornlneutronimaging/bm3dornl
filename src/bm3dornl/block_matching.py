#!/usr/bin/env python3
"""Block matching to build hyper block from single sinogram."""

import numpy as np
from typing import Tuple, List
from numba import njit, prange
# from bm3dornl.utils import (
#     # pad_patch_ids,
#     # find_candidate_patch_ids,
# )


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


@njit
def get_patch_numba(
    image: np.ndarray, position: Tuple[int, int], patch_size: Tuple[int, int]
) -> np.ndarray:
    """
    Retrieve a patch from the image at the specified position.

    Parameters
    ----------
    image : np.ndarray
        The image from which the patch is to be extracted.
    position : tuple
        The row and column indices of the top-left corner of the patch.
    patch_size : tuple
        The size of the patch to be extracted.

    Returns
    -------
    np.ndarray
        The patch extracted from the image.
    """
    i, j = position
    return image[i : i + patch_size[0], j : j + patch_size[1]]


@njit(parallel=True)
def compute_variance_weighted_distance_matrix(
    transformed_patches: np.ndarray,
    variances: np.ndarray,
    patch_positions: np.ndarray,
    cut_off_distance: Tuple[int, int],
) -> np.ndarray:
    """
    Compute the distance matrix for the given signal patches in the frequency domain, considering noise variances.

    Parameters
    ----------
    transformed_patches : np.ndarray
        The transformed (frequency domain) patches for the signal blocks.
    variances : np.ndarray
        The variances for the transformed patches.
    patch_positions : np.ndarray
        The positions of the signal patches.
    cut_off_distance : tuple
        The maximum Manhattan distance for considering candidate patches.

    Returns
    -------
    np.ndarray
        The computed distance matrix.
    """
    num_patches = len(patch_positions)
    distance_matrix = np.zeros((num_patches, num_patches), dtype=float)

    for ref_patch_id in prange(num_patches):
        ref_patch = transformed_patches[ref_patch_id]
        ref_variance = variances[ref_patch_id]
        # note:
        # find_candidate_patch_ids will add the ref_patch_id to the list of candidate patch ids
        candidate_patch_ids = find_candidate_patch_ids(
            patch_positions, ref_patch_id, cut_off_distance
        )
        for neighbor_patch_id in candidate_patch_ids:
            neighbor_patch_id = int(neighbor_patch_id)
            neighbor_patch = transformed_patches[neighbor_patch_id]
            neighbor_variance = variances[neighbor_patch_id]
            # Compute weighted L2 norm distance between patches in the frequency domain
            val_diff = 0.0
            for i in range(ref_patch.shape[0]):
                for j in range(ref_patch.shape[1]):
                    variance_avg = (ref_variance[i, j] + neighbor_variance[i, j]) / 2
                    val_diff += (
                        (ref_patch[i, j] - neighbor_patch[i, j]) ** 2
                    ) / variance_avg
            val_diff = val_diff**0.5
            val_diff = max(val_diff, 1e-8)
            distance_matrix[ref_patch_id, neighbor_patch_id] = val_diff
            distance_matrix[neighbor_patch_id, ref_patch_id] = val_diff

    return distance_matrix


@njit(parallel=True)
def compute_distance_matrix_no_variance(
    transformed_patches: np.ndarray,
    patch_positions: np.ndarray,
    cut_off_distance: Tuple[int, int],
) -> np.ndarray:
    """
    Compute the distance matrix for the given signal patches without considering noise variances.

    Parameters
    ----------
    transformed_patches : np.ndarray
        The transformed (frequency domain) patches for the signal blocks.
    patch_positions : np.ndarray
        The positions of the signal patches.
    cut_off_distance : tuple
        The maximum Manhattan distance for considering candidate patches.

    NOTE
    ----
    The reason behind not reusing compute_variance_weighted_distance_matrix with a unitary variance matrix is that
    we want to minimize the number of operations and memory usage for the case of no noise variance.
    """
    num_patches = len(patch_positions)
    distance_matrix = np.zeros((num_patches, num_patches), dtype=float)

    for ref_patch_id in prange(num_patches):
        ref_patch = transformed_patches[ref_patch_id]
        candidate_patch_ids = find_candidate_patch_ids(
            patch_positions, ref_patch_id, cut_off_distance
        )
        for neighbor_patch_id in candidate_patch_ids:
            neighbor_patch_id = int(neighbor_patch_id)
            neighbor_patch = transformed_patches[neighbor_patch_id]
            # compute L2 norm distance between patches
            val_diff = 0.0
            for i in range(ref_patch.shape[0]):
                for j in range(ref_patch.shape[1]):
                    val_diff += (ref_patch[i, j] - neighbor_patch[i, j]) ** 2
            val_diff = val_diff**0.5
            # check if the distance is within the threshold
            val_diff = max(val_diff, 1e-8)
            distance_matrix[ref_patch_id, neighbor_patch_id] = val_diff
            distance_matrix[neighbor_patch_id, ref_patch_id] = val_diff

    return distance_matrix


@njit(parallel=True)
def form_hyper_blocks_from_distance_matrix(
    distance_matrix: np.ndarray,
    patch_positions: np.ndarray,
    patch_size: Tuple[int, int],
    num_patches_per_group: int,
    image: np.ndarray,
    variances: np.ndarray,
    padding_mode: str = "circular",
):
    """
    Form hyper blocks (groups of patches) from the distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix containing distances between patches.
    patch_positions : np.ndarray
        The positions of the patches.
    patch_size : tuple
        The size of each patch (height, width).
    num_patches_per_group : int
        The number of patches in each group.
    image : np.ndarray
        The input image from which patches are extracted.
    variances : np.ndarray
        The variances of the patches in the transform domain.
    padding_mode : str, optional
        The padding mode for the hyper block, by default "circular".

    Returns
    -------
    block : np.ndarray
        The hyper blocks of patches.
    positions : np.ndarray
        The positions of the patches in the hyper blocks.
    variance_blocks : np.ndarray
        The variances of the patches in the hyper blocks.
    """
    group_size = len(distance_matrix)
    block = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )
    positions = np.empty((group_size, num_patches_per_group, 2), dtype=np.int32)
    variance_blocks = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )

    for i in prange(group_size):
        row = distance_matrix[i]
        candidate_patch_ids = np.where(row > 0)[0]
        candidate_patch_val = row[candidate_patch_ids]
        candidate_patch_ids = candidate_patch_ids[np.argsort(candidate_patch_val)]
        padded_patch_ids = pad_patch_ids(
            candidate_patch_ids, num_patches_per_group, mode=padding_mode
        )

        for j in range(num_patches_per_group):
            idx = int(padded_patch_ids[j])
            patch = get_patch_numba(image, patch_positions[idx], patch_size)
            block[i, j] = patch
            positions[i, j] = patch_positions[idx]
            variance_blocks[i, j] = variances[idx]

    return block, positions, variance_blocks


@njit(parallel=True)
def form_hyper_blocks_from_two_images(
    distance_matrix: np.ndarray,
    patch_positions: np.ndarray,
    patch_size: Tuple[int, int],
    num_patches_per_group: int,
    image1: np.ndarray,
    image2: np.ndarray,
    variances1: np.ndarray,
    padding_mode: str = "circular",
):
    """
    Form hyper blocks (groups of patches) from the distance matrix for two images.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix containing distances between patches.
    patch_positions : np.ndarray
        The positions of the patches.
    patch_size : tuple
        The size of each patch (height, width).
    num_patches_per_group : int
        The number of patches in each group.
    image1 : np.ndarray
        The first input image from which patches are extracted.
    image2 : np.ndarray
        The second input image from which patches are extracted.
    variances1 : np.ndarray
        The variances of the patches in the first image in the transform domain.
    padding_mode : str, optional
        The padding mode for the hyper block, by default "circular".

    Returns
    -------
    block1 : np.ndarray
        The hyper blocks of patches for the first image.
    block2 : np.ndarray
        The hyper blocks of patches for the second image.
    positions : np.ndarray
        The positions of the patches in the hyper blocks.
    variance_blocks1 : np.ndarray
        The variances of the patches in the hyper blocks for the first image.
    """
    group_size = len(distance_matrix)
    block1 = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )
    block2 = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )
    positions = np.empty((group_size, num_patches_per_group, 2), dtype=np.int32)
    variance_blocks1 = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )

    for i in prange(group_size):
        row = distance_matrix[i]
        candidate_patch_ids = np.where(row > 0)[0]
        candidate_patch_val = row[candidate_patch_ids]
        candidate_patch_ids = candidate_patch_ids[np.argsort(candidate_patch_val)]
        padded_patch_ids = pad_patch_ids(
            candidate_patch_ids, num_patches_per_group, mode=padding_mode
        )

        for j in range(num_patches_per_group):
            idx = int(padded_patch_ids[j])
            patch1 = get_patch_numba(image1, patch_positions[idx], patch_size)
            patch2 = get_patch_numba(image2, patch_positions[idx], patch_size)
            block1[i, j] = patch1
            block2[i, j] = patch2
            positions[i, j] = patch_positions[idx]
            variance_blocks1[i, j] = variances1[idx]

    return block1, block2, positions, variance_blocks1
