#!/usr/bin/env python3
"""Functions for aggregating hyper patch block into a single image."""

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
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
    This function uses Numba's JIT compilation with parallel execution to speed up the aggregation of image patches.
    Each thread handles a block of patches independently, reducing computational time significantly on multi-core processors.
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
