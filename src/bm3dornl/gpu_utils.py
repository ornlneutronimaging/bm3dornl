#!/usr/bin/env python3
"""CuPy utility functions for GPU acceleration."""

import numpy as np
import cupy as cp
from cupyx.scipy.linalg import hadamard


def hard_thresholding(
    hyper_block: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Apply shrinkage operation to a block of image patches on GPU using CuPy.

    This function transforms the block of patches into the frequency domain using FFT,
    applies hard thresholding to attenuate small coefficients, and then transforms the
    patches back to the spatial domain to acquire a noise-free estimate.

    Parameters
    ----------
    hyper_block : cp.ndarray
        A 4D CuPy array containing groups of stack of 2D image patches.
        The shape of `hyper_block` should be (group, n_patches, patch_height, patch_width).
    threshold : float
        The threshold value for hard thresholding. Coefficients with absolute values below
        this threshold will be set to zero.

    Returns
    -------
    denoised_block : np.ndarray
        A 4D CuPy array of the same shape as `hyper_block`, containing the denoised patches.

    Notes
    -----
    1. This function uses GPU acceleration to improve the performance of the FFT-based denoising process.
    2. FFT cache are manually cleared to release memory after each iteration, avoid potential CUDA out of memory error.
    """
    # Send data to the GPU
    hyper_block = cp.asarray(hyper_block)

    # Transform the patch block to the frequency domain using rfft
    hyper_block = cp.fft.rfft2(hyper_block, axes=(1, 2, 3))

    # Apply hard thresholding
    hyper_block[cp.abs(hyper_block) < threshold] = 0

    # Transform the block back to the spatial domain using irFFT
    hyper_block = cp.fft.irfft2(hyper_block, axes=(1, 2, 3))

    # Send data back to the CPU
    denoised_block = hyper_block.get()
    del hyper_block

    # release fft cache
    cp.fft.config._get_plan_cache().clear()

    return denoised_block


def wiener_hadamard(hyper_block: np.ndarray, sigma_squared: float):
    """
    Wiener filter using the Hadamard transform, implemented with CuPy for GPU acceleration.

    This function handles both 3D and 4D inputs where patches are square and of size 2^n x 2^n.

    Parameters
    ----------
    hyper_block : cp.ndarray
        A 3D or 4D array containing groups of image patches in the **spatial** domain.
    sigma_squared : float
        The noise variance.

    Returns
    -------
    np.ndarray
        An array of the same shape as `patch_block`, containing the denoised patches.
    """
    # Send data to the GPU
    hyper_block = cp.asarray(hyper_block)

    # Get the size of the patches
    n = hyper_block.shape[-1]  # Assuming square patches
    H = hadamard(n)

    # Flatten 4D to 3D if necessary
    original_shape = hyper_block.shape
    if hyper_block.ndim == 4:
        hyper_block = hyper_block.reshape(-1, n, n)

    # Hadamard transform
    hyper_block = cp.einsum("ij,kjl->kil", H, hyper_block)
    hyper_block = cp.einsum("ijk,kl->ijl", hyper_block, H)

    # Calculate mean and variance across the patches dimension
    local_mean = cp.mean(hyper_block, axis=0, keepdims=True)
    local_variance = cp.var(hyper_block, axis=0, keepdims=True)

    # Apply Wiener filter
    hyper_block = (1 - sigma_squared / (local_variance + 1e-8)) * (
        hyper_block - local_mean
    ) + local_mean
    mask = cp.broadcast_to(local_variance < sigma_squared, hyper_block.shape)
    hyper_block[mask] = 0

    # Inverse Hadamard transform
    hyper_block = cp.einsum("ij,kjl->kil", H, hyper_block)
    hyper_block = cp.einsum("ijk,kl->ijl", hyper_block, H) / (n * n)

    # Reshape back if it was 4D
    if original_shape != hyper_block.shape:
        hyper_block = hyper_block.reshape(original_shape)

    # Send data back to the CPU
    denoised_block = hyper_block.get()

    # release memory
    del hyper_block

    return denoised_block


def memory_cleanup():
    """Clear the memory cache for CuPy and synchronize the default stream."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
