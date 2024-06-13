#!/usr/bin/env python3
"""Functions for aggregating hyper patch block into a single image."""

import numpy as np
from typing import Tuple
from numba import njit, prange


@njit(parallel=True)
def aggregate_patches(
    estimate_denoised_image: np.ndarray,
    weights: np.ndarray,
    hyper_block: np.ndarray,
    hyper_block_index: np.ndarray,
):
    """
    Aggregate patches into the denoised image matrix and update the corresponding weights matrix.

    Parameters
    ----------
    estimate_denoised_image : np.ndarray
        The 2D numpy array where the aggregate result of the denoised patches will be stored.
    weights : np.ndarray
        The 2D numpy array that counts the contributions of the patches to the cells of the `estimate_denoised_image`.
    hyper_block : np.ndarray
        A 4D numpy array of patches to be aggregated. Shape is (num_blocks, num_patches_per_block, patch_height, patch_width).
    hyper_block_index : np.ndarray
        A 3D numpy array containing the top-left indices (row, column) for each patch in the `hyper_block`.
        Shape is (num_blocks, num_patches_per_block, 2).

    Notes
    -----
    This is an old function to be deprecated.
    """
    num_blocks, num_patches, ph, pw = hyper_block.shape
    for i in prange(num_blocks):
        for p in range(num_patches):
            patch = hyper_block[i, p]
            i_pos, j_pos = hyper_block_index[i, p]
            for ii in range(ph):
                for jj in range(pw):
                    estimate_denoised_image[i_pos + ii, j_pos + jj] += patch[ii, jj]
                    weights[i_pos + ii, j_pos + jj] += 1


@njit(parallel=True)
def aggregate_block_to_image(
    image_shape: Tuple[int, int],
    hyper_blocks: np.ndarray,
    hyper_block_indices: np.ndarray,
    variance_blocks: np.ndarray,
) -> np.ndarray:
    """
    Aggregate patches into the denoised image matrix and update the corresponding weights matrix using smart weighting.

    Parameters
    ----------
    image_shape : tuple
        The shape of the image to be denoised.
    hyper_blocks : np.ndarray
        A 4D numpy array of patches to be aggregated. Shape is (num_blocks, num_patches_per_block, patch_height, patch_width).
    hyper_block_indices : np.ndarray
        A 3D numpy array containing the top-left indices (row, column) for each patch in the `hyper_blocks`.
        Shape is (num_blocks, num_patches_per_block, 2).
    variance_blocks : np.ndarray
        A 4D numpy array of the variances for each patch. Shape is the same as `hyper_blocks`.

    Returns
    -------
    np.ndarray
        The denoised image.
    """
    estimate_denoised_image = np.zeros(image_shape)
    weights = np.zeros(image_shape)

    num_blocks, num_patches, ph, pw = hyper_blocks.shape

    for i in prange(num_blocks):
        for p in range(num_patches):
            patch = hyper_blocks[i, p]
            variance = variance_blocks[i, p]
            weight = 1 / (variance + 1e-8)  # Small epsilon to avoid division by zero
            i_pos, j_pos = hyper_block_indices[i, p]
            for ii in range(ph):
                for jj in range(pw):
                    estimate_denoised_image[i_pos + ii, j_pos + jj] += (
                        patch[ii, jj] * weight[ii, jj]
                    )
                    weights[i_pos + ii, j_pos + jj] += weight[ii, jj]

    # Normalize the denoised image by the sum of weights
    estimate_denoised_image /= weights

    return estimate_denoised_image
