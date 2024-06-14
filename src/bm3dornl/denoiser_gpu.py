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
    hyper_blocks = cp.asnumpy(denoised_hyper_blocks)

    # Clear memory cache
    del denoised_hyper_blocks, variance_blocks, threshold
    memory_cleanup()

    return hyper_blocks


def collaborative_wiener_filtering(
    hyper_blocks: np.ndarray,
    variance_blocks: np.ndarray,
    noisy_hyper_blocks: np.ndarray,
) -> np.ndarray:
    """
    Apply Wiener filtering to hyper blocks using their variances and the noisy image patches.

    Parameters
    ----------
    hyper_blocks : np.ndarray
        Hyper blocks of denoised image patches with shape (num_blocks, num_patches_per_block, patch_height, patch_width).
    variance_blocks : np.ndarray
        Hyper blocks of noise variances with shape (num_blocks, num_patches_per_block, patch_height, patch_width).
    noisy_hyper_blocks : np.ndarray
        Hyper blocks of noisy image patches with shape (num_blocks, num_patches_per_block, patch_height, patch_width).

    Returns
    -------
    np.ndarray
        Array of denoised hyper blocks after Wiener filtering.
    """
    # Send data to the GPU
    hyper_blocks = cp.array(hyper_blocks)
    variance_blocks = cp.array(variance_blocks)
    noisy_hyper_blocks = cp.array(noisy_hyper_blocks)

    # Transform the denoised hyper blocks and the noisy hyper blocks to the frequency domain
    hyper_blocks_fft = cp.fft.fftn(hyper_blocks, axes=(-3, -2, -1))
    noisy_hyper_blocks_fft = cp.fft.fftn(noisy_hyper_blocks, axes=(-3, -2, -1))

    # Estimate the power spectral density (PSD) of the signal and noise
    signal_psd = cp.abs(hyper_blocks_fft) ** 2
    noise_psd = variance_blocks

    # Compute Wiener filter coefficients
    wiener_filter = signal_psd / (signal_psd + noise_psd + 1e-8)

    # Apply Wiener filtering
    filtered_blocks_fft = wiener_filter * noisy_hyper_blocks_fft

    # Apply inverse 3D FFT to obtain the denoised hyper blocks
    filtered_blocks = cp.fft.ifftn(filtered_blocks_fft, axes=(-3, -2, -1)).real

    # Send the data back to the CPU
    hyper_blocks = cp.asnumpy(filtered_blocks)

    # Clean up the GPU memory
    del (
        variance_blocks,
        noisy_hyper_blocks,
        hyper_blocks_fft,
        noisy_hyper_blocks_fft,
        signal_psd,
        noise_psd,
        wiener_filter,
        filtered_blocks_fft,
    )
    memory_cleanup()

    return hyper_blocks
