#!/usr/bin/env python3
"""Denoising functions using CuPy for GPU acceleration."""

import logging
import numpy as np
import cupy as cp
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
    get_exact_noise_variance,
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
from .utils import (
    downscale_2d_horizontal,
    upscale_2d_horizontal,
)

# NOTE: These default parameters are based on the parameter tuning study.
#       For more information, please refer to study/parameter_tuning/
default_block_matching_kwargs = {
    "patch_size": (7, 7),
    "stride": 2,
    "background_threshold": 0.0,
    "cut_off_distance": (40, 40),
    "num_patches_per_group": 64,
    "padding_mode": "circular",
}
default_filter_kwargs = {
    "filter_function": "fft",
    "shrinkage_factor": 4e-2,
}


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


def padded_piecewise_weighted_denoising(
    sinogram: np.ndarray,
    window_size: int = 50,
    step_size: int = 10,
    pad_size: int = None,
) -> np.ndarray:
    """
    Perform piecewise weighted denoising on a sinogram with padding using CuPy for GPU acceleration.

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram to be denoised.
    window_size : int, optional
        Size of the window used for piecewise denoising, by default 50.
    step_size : int, optional
        Step size for moving the window, by default 10.
    pad_size : int, optional
        Padding size added to the sinogram before denoising. If None, pad_size is set to 2 * window_size.

    Returns
    -------
    np.ndarray
        Denoised sinogram after piecewise weighted denoising.
    """
    if pad_size is None:
        pad_size = window_size * 2

    # Move data to GPU
    sinogram_gpu = cp.asarray(sinogram)

    # Pad the sinogram
    padded_sinogram = cp.pad(sinogram_gpu, ((pad_size, pad_size), (0, 0)), mode="wrap")

    rows, _ = padded_sinogram.shape
    new_img = cp.zeros_like(padded_sinogram, dtype=cp.float32)
    new_wgt = cp.zeros_like(padded_sinogram, dtype=cp.float32)

    for i in range(0, rows, step_size):
        end = min(i + window_size, rows)
        start = max(0, end - window_size)
        window = padded_sinogram[start:end, :]
        median = cp.median(window, axis=0)
        new_img[start:end, :] += window - median
        new_wgt[start:end, :] += 1

    # Avoid division by zero
    new_wgt = cp.maximum(new_wgt, 1e-10)
    denoised = new_img / new_wgt

    # Restore the overall intensity level
    denoised += cp.median(sinogram_gpu)

    # Remove padding
    denoised = denoised[pad_size:-pad_size, :]

    # Move result back to CPU
    return cp.asnumpy(denoised)


def estimate_noise_free_sinogram(sinogram: np.ndarray) -> np.ndarray:
    """
    Estimate noise-free sinogram from noisy sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        Noisy sinogram.

    Returns
    -------
    np.ndarray
        Noise-free sinogram.
    """
    # use piecewise weighted denoising
    denoised = padded_piecewise_weighted_denoising(sinogram)

    # normalize to [0, 1]
    denoised -= np.min(denoised)
    denoised /= np.max(denoised)

    return denoised


def bm3d_full(
    sinogram: np.ndarray,
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
) -> np.ndarray:
    """Remove ring artifacts from a sinogram using BM3D following the full six steps.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram to be denoised.
    block_matching_kwargs : dict
        The block matching parameters.
    filter_kwargs : dict
        The filter parameters.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    # Unpack the block matching parameters
    patch_size = block_matching_kwargs.get("patch_size", (8, 8))
    stride = block_matching_kwargs.get("stride", 3)
    background_threshold = block_matching_kwargs.get("background_threshold", 0.0)
    cut_off_distance = block_matching_kwargs.get("cut_off_distance", (64, 64))
    num_patches_per_group = block_matching_kwargs.get("num_patches_per_group", 32)
    padding_mode = block_matching_kwargs.get("padding_mode", "circular")
    # Unpack the filter parameters
    filter_function = filter_kwargs.get("filter_function", "fft")
    shrinkage_factor = filter_kwargs.get("shrinkage_factor", 3e-2)

    # Register function based on the method
    filter_function = filter_function.lower()
    if filter_function == "fft":
        transform_func = fft_transform
        shrinkage_func = shrinkage_fft
        collaborative_filtering_func = collaborative_wiener_filtering
    elif filter_function == "hadamard":
        transform_func = hadamard_transform
        shrinkage_func = shrinkage_hadamard
        collaborative_filtering_func = collaborative_hadamard_filtering
    else:
        raise ValueError(f"Unknown filter function: {filter_function}")

    # record original dynamic range
    original_max, original_min = np.max(sinogram), np.min(sinogram)
    # Normalize the sinogram
    z = (sinogram - original_min) / (original_max - original_min)

    # get the patch positions
    patch_positions = get_signal_patch_positions(
        image=z,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
    )
    # retrieve the signal patches
    signal_patches = np.array(
        [get_patch_numba(z, pos, patch_size) for pos in patch_positions]
    )

    # Based on the flow chart in Fig 13 of paper:
    #     doi: 10.1109/TIP.2020.3014721
    # Step 1: use bm3d hard thresholding HT to estimate the noise-free sinogram
    #         HT(z, phi) -> yhat_ht
    # 1.1 transform the patches
    transformed_patches = transform_func(signal_patches)
    # 1.2 estimate the noise variance for transformed patches
    # note: we don't know the noise variance, so we estimate it from the patches
    #       directly
    phi_fft = get_exact_noise_variance(transformed_patches)
    # 1.3 estimate the noise-free sinogram via hard-thresholding
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
    #         GFT(z, phi, yhat_ht) -> z_gft_ht
    # estimate the global noise psd
    phi = estimate_noise_psd(z)
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
    phi_fft_gft_ht = get_exact_noise_variance(transformed_patches)
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
    phi_fft_gft_wie = get_exact_noise_variance(transformed_patches)
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


def bm3d_lite(
    sinogram: np.ndarray,
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
    use_refiltering: bool = True,
) -> np.ndarray:
    """Remove ring artifacts from a sinogram using BM3D in simple and express mode.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram to be denoised.
    block_matching_kwargs : dict
        The block matching parameters.
    filter_kwargs : dict
        The filter parameters.
    use_refiltering : bool
        Whether to use refiltering.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    # Unpack the block matching parameters
    patch_size = block_matching_kwargs.get("patch_size", (8, 8))
    stride = block_matching_kwargs.get("stride", 3)
    background_threshold = block_matching_kwargs.get("background_threshold", 0.0)
    cut_off_distance = block_matching_kwargs.get("cut_off_distance", (64, 64))
    num_patches_per_group = block_matching_kwargs.get("num_patches_per_group", 32)
    padding_mode = block_matching_kwargs.get("padding_mode", "circular")
    # Unpack the filter parameters
    filter_function = filter_kwargs.get("filter_function", "fft")

    # Register function based on the method
    filter_function = filter_function.lower()
    if filter_function == "fft":
        transform_func = fft_transform
        collaborative_filtering_func = collaborative_wiener_filtering
    elif filter_function == "hadamard":
        transform_func = hadamard_transform
        collaborative_filtering_func = collaborative_hadamard_filtering
    else:
        raise ValueError(f"Unknown filter function: {filter_function}")

    # record original dynamic range
    original_max, original_min = np.max(sinogram), np.min(sinogram)

    # Normalize the sinogram
    z = (sinogram - original_min) / (original_max - original_min)
    # get the patch positions
    patch_positions = get_signal_patch_positions(
        image=z,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
    )
    # express mode = notch_filter + wiener_filtering
    # replace block matching hard thresholding with simple estimate via a notch filter
    yhat_ht_gft = estimate_noise_free_sinogram(z)
    # estimate noises
    eta = z - yhat_ht_gft
    # estimate the noise variance
    noise_patches = np.array(
        [get_patch_numba(eta, pos, patch_size) for pos in patch_positions]
    )
    # transform the noise patches
    transformed_noise_patches = transform_func(noise_patches)
    # estimate the noise variance for transformed patches
    phi_fft = np.abs(transformed_noise_patches) ** 2
    #
    yhat_final = collaborative_filtering(
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
    if use_refiltering:
        # simple mode = express mode + global refiltering + wiener filtering
        eta = z - yhat_final
        eta_psd = np.abs(np.fft.fft2(eta)) ** 2
        z = global_fourier_thresholding(z, eta_psd, yhat_final)
        #
        eta = z - yhat_final
        noise_patches = np.array(
            [get_patch_numba(eta, pos, patch_size) for pos in patch_positions]
        )
        transformed_noise_patches = transform_func(noise_patches)
        phi_fft = np.abs(transformed_noise_patches) ** 2
        yhat_final = collaborative_filtering(
            sinogram=z,
            denoised_sinogram=yhat_final,
            patch_size=patch_size,
            num_patches_per_group=num_patches_per_group,
            padding_mode=padding_mode,
            noise_variance=phi_fft,
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


def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    mode: str = "simple",  # express, simple, full
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
) -> np.ndarray:
    """Remove ring artifacts from a sinogram using BM3D.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram to be denoised.
    mode : str
        The denoising mode to use.
    block_matching_kwargs : dict
        The block matching parameters.
    filter_kwargs : dict
        The filter parameters.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    if mode == "full":
        return bm3d_full(
            sinogram=sinogram,
            block_matching_kwargs=block_matching_kwargs,
            filter_kwargs=filter_kwargs,
        )
    elif mode == "express":
        return bm3d_lite(
            sinogram=sinogram,
            block_matching_kwargs=block_matching_kwargs,
            filter_kwargs=filter_kwargs,
            use_refiltering=False,
        )
    elif mode == "simple":
        return bm3d_lite(
            sinogram=sinogram,
            block_matching_kwargs=block_matching_kwargs,
            filter_kwargs=filter_kwargs,
            use_refiltering=True,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_scale_adjusted_blockmatching_params(
    original_params: dict, scale_factor: int
) -> dict:
    """Scale the parameters based on the given factor.

    Parameters
    ----------
    original_params : dict
        The original parameters.
    scale_factor : int
        The scale factor.

    Returns
    -------
    dict
        The adjusted parameters.
    """
    adjusted_params = original_params.copy()

    # Adjust patch size
    # minimum patch size is 3x3
    adjusted_params["patch_size"] = tuple(
        max(3, int(x * scale_factor)) for x in original_params["patch_size"]
    )

    # Adjust stride
    # minimum stride is 1
    adjusted_params["stride"] = max(1, int(original_params["stride"] * scale_factor))

    # Adjust cut-off distance
    # minimum cut-off distance is 8
    adjusted_params["cut_off_distance"] = tuple(
        max(8, int(x / scale_factor)) for x in original_params["cut_off_distance"]
    )

    # Optionally adjust number of patches per group
    if scale_factor > 1:
        adjusted_params["num_patches_per_group"] = max(
            16, original_params["num_patches_per_group"] // scale_factor
        )

    return adjusted_params


def bm3d_ring_artifact_removal_ms(
    sinogram: np.ndarray,
    k: int = 3,
    mode: str = "simple",  # express, simple, full
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
    use_iterative_refinement: bool = True,
    refinement_iterations: int = 3,
    scale_factor_base: int = 2,
) -> np.ndarray:
    """
    Multiscale BM3D for streak removal

    Parameters
    ----------
    sinogram : np.ndarray
        The input sinogram to be denoised.
    k : int, optional
        The number of iterations for horizontal binning, by default 3
    mode : str
        The denoising mode to use.
    block_matching_kwargs : dict
        The block matching parameters.
    filter_kwargs : dict
        The filter parameters.
    use_iterative_refinement : bool, optional
        Whether to use iterative refinement in upscaling, by default True
    refinement_iterations : int, optional
        Number of refinement iterations if using iterative refinement, by default 3
    scale_factor_base : int, optional
        The base scale factor for binning, by default 2

    Returns
    -------
    np.ndarray
        The denoised sinogram.

    References
    ----------
    [1] ref: `Collaborative Filtering of Correlated Noise <https://doi.org/10.1109/TIP.2020.3014721>`_
    [2] ref: `Ring artifact reduction via multiscale nonlocal collaborative filtering of spatially correlated noise <https://doi.org/10.1107/S1600577521001910>`_
    """
    # step 0: initialize
    sino_star = np.array(sinogram)

    if k == 0:
        # single pass
        return bm3d_ring_artifact_removal(
            sino_star,
            mode=mode,
            block_matching_kwargs=block_matching_kwargs,
            filter_kwargs=filter_kwargs,
        )

    binned_sinos_orig = [sino_star]

    # Bin horizontally
    for i in range(k):
        binned_sinos_orig.append(downscale_2d_horizontal(binned_sinos_orig[-1], 2))

    # Multi-scale denoising
    for i in range(k, -1, -1):
        logging.info(f"Processing binned sinogram {i + 1} of {k + 1}")

        # compute the adjusted parameters
        scale_factor = int(scale_factor_base ** (i / 2))
        adjusted_block_matching_kwargs = get_scale_adjusted_blockmatching_params(
            block_matching_kwargs, scale_factor
        )
        adjusted_filter_kwargs = filter_kwargs.copy()
        adjusted_filter_kwargs["shrinkage_factor"] = (
            filter_kwargs["shrinkage_factor"] / scale_factor
        )

        # Denoise binned sinogram
        denoised_sino = bm3d_ring_artifact_removal(
            binned_sinos_orig[i],
            mode=mode,
            block_matching_kwargs=adjusted_block_matching_kwargs,
            filter_kwargs=adjusted_filter_kwargs,
        )

        # For iterations except the last, create the next noisy image with a finer scale residual
        if i > 0:
            # Calculate the noise at current scale
            noise_at_scale_i = binned_sinos_orig[i] - denoised_sino

            # Upscale the noise to the next finer scale
            upscaled_noise = upscale_2d_horizontal(
                noise_at_scale_i,
                2,
                original_width=binned_sinos_orig[i - 1].shape[1],
                use_iterative_refinement=use_iterative_refinement,
                refinement_iterations=refinement_iterations,
            )

            # Remove the upscaled noise from the finer scale
            # NOTE: The subtraction of noise will also be upscaled in the next iteration, therefore
            #       propagating the noise removal from coarser to finer scales
            binned_sinos_orig[i - 1] -= upscaled_noise

    return binned_sinos_orig[0]
