"""Denoising functions using CuPy for GPU acceleration."""

import numpy as np
import cupy as cp


def memory_cleanup():
    """Clear the memory cache for CuPy and synchronize the default stream."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()


def shrinkage_hyper_blocks(
    hyper_blocks: np.ndarray, variance_blocks: np.ndarray, threshold_factor: float = 3
) -> np.ndarray:
    """
    Apply shrinkage to hyper blocks using their variances.

    Parameters
    ----------
    hyper_blocks : np.ndarray
        Hyper blocks of image patches with shape (num_blocks, num_patches_per_block, patch_height, patch_width).
    variance_blocks : np.ndarray
        Hyper blocks of noise variances with shape (num_blocks, num_patches_per_block, patch_height, patch_width).
    threshold_factor : float
        Factor to determine the shrinkage threshold.

    Returns
    -------
    np.ndarray
        Array of denoised hyper blocks with the same shape as the input hyper blocks.
    """
    # Send data to GPU
    denoised_hyper_blocks = cp.asarray(hyper_blocks)
    variance_blocks = cp.asarray(variance_blocks)

    # Compute the threshold for shrinkage
    threshold = threshold_factor * cp.sqrt(variance_blocks)

    # Transform the hyper blocks to the frequency domain
    denoised_hyper_blocks = cp.fft.fftn(denoised_hyper_blocks, axes=(-3, -2, -1))

    # Apply shrinkage (hard thresholding) in the frequency domain
    denoised_hyper_blocks = cp.where(
        np.abs(denoised_hyper_blocks) > threshold, denoised_hyper_blocks, 0
    )

    # Apply inverse 3D FFT to obtain the denoised hyper blocks
    denoised_hyper_blocks = cp.fft.ifftn(denoised_hyper_blocks, axes=(-3, -2, -1)).real

    # Send data back to CPU
    hyper_blocks = denoised_hyper_blocks.get()

    # Clear memory cache
    del denoised_hyper_blocks, variance_blocks, threshold
    memory_cleanup()

    return hyper_blocks
