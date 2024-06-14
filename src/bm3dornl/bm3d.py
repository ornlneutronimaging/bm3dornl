#!/usr/bin/env python3
"""Denoising functions using CuPy for GPU acceleration."""

import numpy as np
from typing import Tuple, Callable
from .block_matching import (
    get_signal_patch_positions,
    get_patch_numba,
    compute_variance_weighted_distance_matrix,
    compute_distance_matrix_no_variance,
    form_hyper_blocks_from_distance_matrix,
    form_hyper_blocks_from_two_images,
)
from .noise_analysis import (
    estimate_noise_psd,
    get_exact_noise_variance_fft,
    get_exact_noise_variance_hadamard,
)
from .denoiser_gpu import (
    shrinkage_fft,
    shrinkage_hadamard,
    collaborative_wiener_filtering,
    collaborative_hadamard_filtering,
)
from .aggregation import (
    aggregate_block_to_image,
    aggregate_denoised_block_to_image,
)
from .signal import (
    fft_transform,
    hadamard_transform,
)


def shrinkage_via_hardthresholding(
    sinogram: np.ndarray,
    patch_size: Tuple[int, int],
    num_patches_per_group: int,
    padding_mode: str,
    transformed_patches: np.ndarray,
    noise_variance: np.ndarray,
    patch_positions: np.ndarray,
    cut_off_distance: Tuple[int, int],
    shrinkage_factor: float,
    shrinkage_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
):
    """Apply BM3D-ORNL denoising to a sinogram using hard thresholding.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram to be denoised.
    patch_size : Tuple[int, int]
        The size of the patches.
    num_patches_per_group : int
        The number of patches per group.
    padding_mode : str
        The padding mode for the patches.
    transformed_patches : np.ndarray
        The transformed patches, [block_num, patch_num, patch_height, patch_width]
    noise_variance : np.ndarray
        The noise variance, [block_num, patch_num, patch_height, patch_width]
    patch_positions : np.ndarray
        The patch positions.
    cut_off_distance : Tuple[int, int]
        The cut-off distance.
    shrinkage_factor : float
        The shrinkage factor.
    shrinkage_func : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        The shrinkage function.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    # Compute the variance-weighted distance matrix
    distance_matrix = compute_variance_weighted_distance_matrix(
        transformed_patches=transformed_patches,
        variances=noise_variance,
        patch_positions=patch_positions,
        cut_off_distance=cut_off_distance,
    )
    # Construct the hyper block
    blocks, positions, blocks_var = form_hyper_blocks_from_distance_matrix(
        distance_matrix=distance_matrix,
        patch_positions=patch_positions,
        patch_size=patch_size,
        num_patches_per_group=num_patches_per_group,
        image=sinogram,
        variances=noise_variance,
        padding_mode=padding_mode,
    )
    # Hard thresholding via shrinkage
    denoised_blocks = shrinkage_func(
        hyper_blocks=blocks,
        variance_blocks=blocks_var,
        threshold_factor=shrinkage_factor,
    )
    # Reconstruct the denoised sinogram
    denoised_sinogram = aggregate_block_to_image(
        image_shape=sinogram.shape,
        hyper_blocks=denoised_blocks,
        hyper_block_indices=positions,
        variance_blocks=blocks_var,
    )
    # Rescale to [0, 1]
    denoised_sinogram = (denoised_sinogram - np.min(denoised_sinogram)) / (
        np.max(denoised_sinogram) - np.min(denoised_sinogram)
    )

    return denoised_sinogram


def collaborative_filtering(
    sinogram: np.ndarray,
    denoised_sinogram: np.ndarray,
    patch_size: Tuple[int, int],
    num_patches_per_group: int,
    padding_mode: str,
    noise_variance: np.ndarray,
    patch_positions: np.ndarray,
    cut_off_distance: Tuple[int, int],
    transform_func: Callable[[np.ndarray], np.ndarray],
    collaborative_filtering_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ],
) -> np.ndarray:
    """Apply BM3D-ORNL denoising to a sinogram using collaborative filtering.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram to be denoised.
    denoised_sinogram : np.ndarray
        The denoised sinogram from previous estimate.
    patch_size : Tuple[int, int]
        The size of the patches.
    num_patches_per_group : int
        The number of patches per group.
    padding_mode : str
        The padding mode for the patches.
    noise_variance : np.ndarray
        The noise variance, [block_num, patch_num, patch_height, patch_width]
    patch_positions : np.ndarray
        The patch positions.
    cut_off_distance : Tuple[int, int]
        The cut-off distance.
    transform_func : Callable[[np.ndarray], np.ndarray]
        The transform function.
    collaborative_filtering_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        The collaborative filtering function.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    # Retrive the signal patches for the denoised sinogram
    signal_patches_denoised = np.array(
        [get_patch_numba(denoised_sinogram, pos, patch_size) for pos in patch_positions]
    )
    # Transform the patches to the frequency domain
    transformed_patches_denoised = transform_func(signal_patches_denoised)
    # Compute the new distance matrix without noise variances
    distance_matrix = compute_distance_matrix_no_variance(
        transformed_patches=transformed_patches_denoised,
        patch_positions=patch_positions,
        cut_off_distance=cut_off_distance,
    )
    # Build blocks from two images
    blocks_org, block_denoised, positions, blocks_var = (
        form_hyper_blocks_from_two_images(
            distance_matrix=distance_matrix,
            patch_positions=patch_positions,
            patch_size=patch_size,
            num_patches_per_group=num_patches_per_group,
            image1=sinogram,
            image2=denoised_sinogram,
            variances1=noise_variance,
            padding_mode=padding_mode,
        )
    )
    # use collaborative Wiener filtering
    denoised_blocks = collaborative_filtering_func(
        hyper_blocks=block_denoised,
        variance_blocks=blocks_var,
        noisy_hyper_blocks=blocks_org,
    )
    # Reconstruct the denoised sinogram
    denoised_sinogram = aggregate_denoised_block_to_image(
        image_shape=sinogram.shape,
        denoised_patches=denoised_blocks,
        patch_positions=positions,
    )
    # Normalize the denoised sinogram
    denoised_sinogram = (denoised_sinogram - np.min(denoised_sinogram)) / (
        np.max(denoised_sinogram) - np.min(denoised_sinogram)
    )

    return denoised_sinogram


def global_wiener_filtering(sinogram: np.ndarray) -> np.ndarray:
    """Apply global Wiener filtering to the sinogram (2D image).

    Parameters
    ----------
    sinogram : np.ndarray
        The input sinogram to be denoised.
    transform_func : Callable[[np.ndarray], np.ndarray]
        The transform function to be applied to the sinogram.
    inverse_transform_func : Callable[[np.ndarray], np.ndarray]
        The inverse transform function to be applied to the denoised sinogram.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    # Transform the entire sinogram
    transformed_sinogram = np.fft.fft2(sinogram)
    # Estimate the global noise variance
    global_noise_variance = np.mean(np.abs(transformed_sinogram) ** 2)
    # Estimate the power spectral density (PSD) of the signal
    signal_psd = np.abs(transformed_sinogram) ** 2
    # Apply Wiener filtering directly to the transformed image
    wiener_filter = signal_psd / (signal_psd + global_noise_variance + 1e-8)
    denoised_sinogram = wiener_filter * transformed_sinogram
    # Inverse transform
    denoised_sinogram = np.fft.ifft2(denoised_sinogram).real
    denoised_sinogram = (denoised_sinogram - np.min(denoised_sinogram)) / (
        np.max(denoised_sinogram) - np.min(denoised_sinogram)
    )

    return denoised_sinogram


def global_fourier_thresholding(
    noisy_image: np.ndarray, noise_psd: np.ndarray, estimated_image: np.ndarray
) -> np.ndarray:
    """
    Apply global Fourier thresholding to the noisy image using the estimated noise-free image.

    Parameters
    ----------
    noisy_image : np.ndarray
        The original noisy image.
    noise_psd : np.ndarray
        The noise power spectral density (PSD).
    estimated_image : np.ndarray
        The estimated noise-free image.

    Returns
    -------
    np.ndarray
        The new noisy image after GFT.
    """
    # Compute the Fourier transforms of the noisy image and estimated noise-free image
    noisy_image_fft = np.fft.fft2(noisy_image)
    estimated_image_fft = np.fft.fft2(estimated_image)

    # Compute the power spectral density (PSD) of the estimated image
    estimated_psd = np.abs(estimated_image_fft) ** 2

    # Compute the threshold
    wiener = estimated_psd / (estimated_psd + noise_psd + 1e-8)

    # Apply the thresholding
    thresholded_image_fft = wiener * noisy_image_fft

    # Inverse Fourier transform to get the new noisy image
    new_noisy_image = np.fft.ifft2(thresholded_image_fft).real

    return new_noisy_image


def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    background_threshold: float = 0.0,
    cut_off_distance: Tuple[int, int] = (64, 64),
    num_patches_per_group: int = 32,
    padding_mode: str = "circular",
    method: str = "fft",
    shrinkage_factor: float = 3e-2,
) -> np.ndarray:
    """Remove ring artifacts from a sinogram using BM3D."""
    # Register function based on the method
    method = method.lower()
    if method == "fft":
        transform_func = fft_transform
        noise_variance_func = get_exact_noise_variance_fft
        shrinkage_func = shrinkage_fft
        collaborative_filtering_func = collaborative_wiener_filtering
    elif method == "hadamard":
        transform_func = hadamard_transform
        noise_variance_func = get_exact_noise_variance_hadamard
        shrinkage_func = shrinkage_hadamard
        collaborative_filtering_func = collaborative_hadamard_filtering
    else:
        raise ValueError(f"Unknown method: {method}")

    # record original dynamic range
    original_max = np.max(sinogram)
    original_min = np.min(sinogram)

    # Normalize the sinogram
    z = (sinogram - original_min) / (original_max - original_min)
    # estimate the global noise psd
    phi = estimate_noise_psd(z)

    # Step 1: use bm3d hard thresholding HT to estimate the noise-free sinogram
    #         HT(z, phi) -> yhat_ht
    # 1.1 Get the positions of the signal patches
    patch_positions = get_signal_patch_positions(
        image=z,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
    )
    # 1.2 Retrive the signal patches
    signal_patches = np.array(
        [get_patch_numba(z, pos, patch_size) for pos in patch_positions]
    )
    # 1.3 Compute the transformed patches
    transformed_patches = transform_func(signal_patches)
    # 1.4 Estimate the noise variance for transformed patches
    phi_fft = noise_variance_func(signal_patches)
    # 1.5 Estimate the noise-free sinogram via hard-thresholding
    yhat_ht = shrinkage_via_hardthresholding(
        sinogram=z,
        patch_size=patch_size,
        num_patches_per_group=num_patches_per_group,
        padding_mode=padding_mode,
        transformed_patches=transformed_patches,
        noise_variance=phi_fft,
        patch_positions=patch_positions,
        cut_off_distance=cut_off_distance,
        shrinkage_factor=shrinkage_factor,
        shrinkage_func=shrinkage_func,
    )

    # Step 2: use global fourier thresholding GFT to refine the noise-free sinogram
    #         GFT(z, phi, yhat_ht) -> yhat_gft_ht, phi_gft_ht
    z_gft_ht = global_fourier_thresholding(z, phi, yhat_ht)

    # Step 3: use hard thresholding again to refine the GFT result
    #         HT(z_gft_ht, phi_fft_gft_ht) -> yhat_ht_gft
    # 3.1 Retrive the signal patches
    signal_patches = np.array(
        [get_patch_numba(z_gft_ht, pos, patch_size) for pos in patch_positions]
    )
    # 3.2 Compute the transformed patches
    transformed_patches = transform_func(signal_patches)
    # 3.3 Estimate the noise variance for transformed patches
    phi_fft_gft_ht = noise_variance_func(signal_patches)
    # 3.4 Estimate the noise-free sinogram via hard-thresholding
    yhat_ht_gft = shrinkage_via_hardthresholding(
        sinogram=z_gft_ht,
        patch_size=patch_size,
        num_patches_per_group=num_patches_per_group,
        padding_mode=padding_mode,
        transformed_patches=transformed_patches,
        noise_variance=phi_fft_gft_ht,
        patch_positions=patch_positions,
        cut_off_distance=cut_off_distance,
        shrinkage_factor=shrinkage_factor,
        shrinkage_func=shrinkage_func,
    )

    # Step 4: use collaborative filtering to refine the hard-thresholding result
    #         WIE(z, phi, yhat_ht_gft) -> yhat_wie
    yhat_wie = collaborative_filtering(
        sinogram=z,
        denoised_sinogram=yhat_ht_gft,
        patch_size=patch_size,
        num_patches_per_group=num_patches_per_group,
        padding_mode=padding_mode,
        noise_variance=phi_fft,
        patch_positions=patch_positions,
        cut_off_distance=cut_off_distance,
        transform_func=transform_func,
        collaborative_filtering_func=collaborative_filtering_func,
    )

    # Step 5: use global Fourier thresholding to refine the collaborative filtering result
    #         GFT(z, phi, yhat_wie) -> zhat_gft_wie, phi_gft_wie
    z_gft_wie = global_fourier_thresholding(z, phi, yhat_wie)

    # Step 6: final step with collaborative filtering
    #         WIE(z_gft_wie, phi_fft_gft_wie, yhat_wie) -> yhat_final
    # 6.1 Retrive the signal patches
    signal_patches = np.array(
        [get_patch_numba(z_gft_wie, pos, patch_size) for pos in patch_positions]
    )
    # 6.2 Compute the transformed patches
    transformed_patches = transform_func(signal_patches)
    # 6.3 Estimate the noise variance for transformed patches
    phi_fft_gft_wie = noise_variance_func(signal_patches)
    # 6.4 Compute the final denoised sinogram
    yhat_final = collaborative_filtering(
        sinogram=z_gft_wie,
        denoised_sinogram=yhat_wie,
        patch_size=patch_size,
        num_patches_per_group=num_patches_per_group,
        padding_mode=padding_mode,
        noise_variance=phi_fft_gft_wie,
        patch_positions=patch_positions,
        cut_off_distance=cut_off_distance,
        transform_func=transform_func,
        collaborative_filtering_func=collaborative_filtering_func,
    )
    # rescale yhat_final to [0, 1]
    yhat_final = (yhat_final - np.min(yhat_final)) / (
        np.max(yhat_final) - np.min(yhat_final)
    )
    # Restore the original dynamic range
    yhat_final = yhat_final * (original_max - original_min) + original_min

    return yhat_final
