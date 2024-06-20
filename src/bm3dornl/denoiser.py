#!/usr/bin/env python
"""Denoiser module for BM3D-ORNL."""

import logging
import numpy as np
from typing import Tuple, Optional
from scipy.signal import medfilt2d
from bm3dornl.aggregation import aggregate_patches
from bm3dornl.block_matching import PatchManager
from bm3dornl.gpu_utils import (
    wiener_hadamard,
    memory_cleanup,
)
from bm3dornl.utils import (
    horizontal_binning,
    horizontal_debinning,
    estimate_noise_free_sinogram,
)


def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    background_threshold: Optional[float] = None,
    cut_off_distance: Tuple[int, int] = (64, 64),
    intensity_diff_threshold: float = 0.2,
    num_patches_per_group: int = 400,
    padding_mode: str = "circular",
    gaussian_noise_estimate: float = 5.0,
    wiener_filter_strength: float = 1.0,
) -> np.ndarray:
    """Remove ring artifact source in the sinogram using BM3D.

    Parameters
    ----------
    sinogram : np.ndarray
        The input sinogram to be denoised.
    patch_size : tuple[int, int], optional
        The size of the patches, by default (8, 8)
    stride:
        Steps when generating blocks with sliding window.
    background_threshold: float
        Estimated background intensity threshold, default to None.
    cut_off_distance : tuple
        Maximum spatial distance in terms of row and column indices for patches in the same block.
    intensity_diff_threshold : float, optional
        The threshold for patch similarity, by default 0.2
    num_patches_per_group : int
        The number of patch in each block.
    padding_mode : str, optional
        The padding mode for the hyper block, by default "circular"
    gaussian_noise_estimate : float, optional
        The estimated standard deviation of the Gaussian noise, by default 5.0
    wiener_filter_strength : float, optional
        The strength of the Wiener filter, by default 1.0. Higher means more aggressive filtering, but may
        remove some details.

    Returns
    -------
    np.ndarray
        The denoised sinogram.
    """
    # estimate the background threshold if not provided
    if background_threshold is None:
        background_threshold = np.percentile(sinogram, 2)

    # Step 1: estimate the noise free representation via simple fft filtering
    logging.info("Estimating noise free sinogram...")
    # cache dynamic range
    sino_min = np.min(sinogram)
    sino_max = np.max(sinogram)
    # rescale the sinogram to [0, 1]
    sinogram_noise_free_estimate = (sinogram - sino_min) / (sino_max - sino_min)
    # apply fft filtering
    sinogram_noise_free_estimate = estimate_noise_free_sinogram(
        sinogram=sinogram_noise_free_estimate,
        background_estimate=background_threshold,
        sigma_gaussian=gaussian_noise_estimate,
    )
    # scale back to original dynamic range
    sinogram_noise_free_estimate = (
        sinogram_noise_free_estimate * (sino_max - sino_min) + sino_min
    )

    # Step 2: re-filtering using the noise free estimate
    logging.info("Re-filtering using BM3D...")
    patch_manager = PatchManager(
        image=sinogram_noise_free_estimate,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
    )
    # patch_manager.image = sinogram_noise_free_estimate
    # re-compute the group of blocks
    patch_manager.group_signal_patches(
        cut_off_distance=cut_off_distance,
        intensity_diff_threshold=intensity_diff_threshold,
    )
    # estimate the noise level
    sigma_squared = np.var(sinogram - sinogram_noise_free_estimate)
    # get hyper blocks
    block, positions = patch_manager.get_hyper_block(
        num_patches_per_group=num_patches_per_group,
        padding_mode=padding_mode,
        alternative_source=sinogram,
    )
    # apply wiener-hadamard filtering
    block = wiener_hadamard(
        block,
        sigma_squared * wiener_filter_strength,
    )
    memory_cleanup()  # manual release of memory
    # aggregate the patches
    accumulator = np.zeros_like(sinogram)
    weights = np.zeros_like(sinogram)
    aggregate_patches(
        estimate_denoised_image=accumulator,
        weights=weights,
        hyper_block=block,
        hyper_block_index=positions,
    )
    # normalize the image
    sinogram_denoised = accumulator / np.maximum(weights, 1)
    return sinogram_denoised


def bm3d_ring_artifact_removal_ms1(
    sinogram: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    background_threshold: float = 0.0,
    cut_off_distance: Tuple[int, int] = (64, 64),
    intensity_diff_threshold: float = 0.1,
    num_patches_per_group: int = 400,
    padding_mode: str = "circular",
    gaussian_noise_estimate: float = 5.0,
    wiener_filter_strength: float = 1.0,
    k: int = 4,
) -> np.ndarray:
    """Multiscale BM3D for streak removal

    Parameters
    ----------
    sinogram : np.ndarray
        The input sinogram to be denoised.
    patch_size : tuple[int, int], optional
        The size of the patches, by default (8, 8)
    stride:
        Steps when generating blocks with sliding window.
    background_threshold: float
        Estimated background intensity threshold, default to 0.0.
    cut_off_distance : tuple
        Maximum spatial distance in terms of row and column indices for patches in the same block.
    intensity_diff_threshold : float, optional
        The threshold for patch similarity, by default 0.01
    num_patches_per_group : int
        The number of patch in each block.
    padding_mode : str, optional
        The padding mode for the hyper block, by default "circular".
    gaussian_noise_estimate : float, optional
        The estimated standard deviation of the Gaussian noise, by default 5.0.
    wiener_filter_strength : float, optional
        The strength of the Wiener filter, by default 1.0. Higher means more aggressive filtering, but may
        remove some details.
    k : int, optional
        The number of iterations for horizontal binning, by default 3

    Returns
    -------
    np.ndarray
        The denoised sinogram.

    References
    ----------
    [1] ref: `Collaborative Filtering of Correlated Noise <https://doi.org/10.1109/TIP.2020.3014721>`_
    [2] ref: `Ring artifact reduction via multiscale nonlocal collaborative filtering of spatially correlated noise <https://doi.org/10.1107/S1600577521001910>`_
    """
    # step 0: median filter the sinogram
    sinogram = medfilt2d(sinogram, kernel_size=3)
    sino_star = sinogram

    kwargs = {
        "patch_size": patch_size,
        "stride": stride,
        "background_threshold": background_threshold,
        "cut_off_distance": cut_off_distance,
        "intensity_diff_threshold": intensity_diff_threshold,
        "num_patches_per_group": num_patches_per_group,
        "padding_mode": padding_mode,
        "gaussian_noise_estimate": gaussian_noise_estimate,
        "wiener_filter_strength": wiener_filter_strength,
    }

    if k == 0:
        # single pass
        return bm3d_ring_artifact_removal(sino_star, **kwargs)

    # step 1: create a list of binned sinograms
    binned_sinos = horizontal_binning(sinogram, k=k)
    # reverse the list
    binned_sinos = binned_sinos[::-1]

    # step 2: estimate the noise level from the coarsest sinogram, then working back to the original sinogram
    noise_estimate = None
    for i in range(len(binned_sinos)):
        logging.info(f"Processing binned sinogram {i+1} of {len(binned_sinos)}")
        sino = binned_sinos[i]
        sino_star = (
            sino if i == 0 else sino - horizontal_debinning(noise_estimate, sino)
        )

        if i < len(binned_sinos) - 1:
            noise_estimate = sino - bm3d_ring_artifact_removal(sino_star, **kwargs)

    return sino_star


def bm3d_ring_artifact_removal_ms(
    sinogram: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    background_threshold: float = 0.0,
    cut_off_distance: Tuple[int, int] = (64, 64),
    intensity_diff_threshold: float = 0.1,
    num_patches_per_group: int = 400,
    padding_mode: str = "circular",
    gaussian_noise_estimate: float = 5.0,
    wiener_filter_strength: float = 1.0,
    k: int = 4,
) -> np.ndarray:
    """Multiscale BM3D for streak removal

    Parameters
    ----------
    sinogram : np.ndarray
        The input sinogram to be denoised.
    patch_size : tuple[int, int], optional
        The size of the patches, by default (8, 8)
    stride:
        Steps when generating blocks with sliding window.
    background_threshold: float
        Estimated background intensity threshold, default to 0.0.
    cut_off_distance : tuple
        Maximum spatial distance in terms of row and column indices for patches in the same block.
    intensity_diff_threshold : float, optional
        The threshold for patch similarity, by default 0.01
    num_patches_per_group : int
        The number of patch in each block.
    padding_mode : str, optional
        The padding mode for the hyper block, by default "circular".
    gaussian_noise_estimate : float, optional
        The estimated standard deviation of the Gaussian noise, by default 5.0.
    wiener_filter_strength : float, optional
        The strength of the Wiener filter, by default 1.0. Higher means more aggressive filtering, but may
        remove some details.
    k : int, optional
        The number of iterations for horizontal binning, by default 3

    Returns
    -------
    np.ndarray
        The denoised sinogram.

    References
    ----------
    [1] ref: `Collaborative Filtering of Correlated Noise <https://doi.org/10.1109/TIP.2020.3014721>`_
    [2] ref: `Ring artifact reduction via multiscale nonlocal collaborative filtering of spatially correlated noise <https://doi.org/10.1107/S1600577521001910>`_
    """
    # step 0: median filter the sinogram
    sinogram = medfilt2d(sinogram, kernel_size=3)
    sino_star = sinogram

    kwargs = {
        "patch_size": patch_size,
        "stride": stride,
        "background_threshold": background_threshold,
        "cut_off_distance": cut_off_distance,
        "intensity_diff_threshold": intensity_diff_threshold,
        "num_patches_per_group": num_patches_per_group,
        "padding_mode": padding_mode,
        "gaussian_noise_estimate": gaussian_noise_estimate,
        "wiener_filter_strength": wiener_filter_strength,
    }

    if k == 0:
        # single pass
        return bm3d_ring_artifact_removal(sino_star, **kwargs)

    denoised_sino = None
    # Make a copy of an original sinogram
    sino_orig = horizontal_binning(sino_star, 1, dim=0)
    binned_sinos_orig = [np.copy(sino_orig)]

    # Contains upscaled denoised sinograms
    binned_sinos = [np.zeros(0)]

    # Bin horizontally
    for i in range(0, k):
        binned_sinos_orig.append(horizontal_binning(binned_sinos_orig[-1], 2, dim=1))
        binned_sinos.append(np.zeros(0))

    binned_sinos[-1] = binned_sinos_orig[-1]

    for i in range(k, -1, -1):
        logging.info(f"Processing binned sinogram {i + 1} of {k}")
        # Denoise binned sinogram
        denoised_sino = bm3d_ring_artifact_removal(binned_sinos[i], **kwargs)
        # For iterations except the last, create the next noisy image with a finer scale residual
        if i > 0:
            debinned_sino = horizontal_debinning(denoised_sino - binned_sinos_orig[i], binned_sinos_orig[i - 1].shape[1], 2, 30, dim=1)
            binned_sinos[i - 1] = binned_sinos_orig[i - 1] + debinned_sino

    # residual
    sino_star = sino_star + horizontal_debinning(denoised_sino - sino_orig, sino_star.shape[0], 1, 30, dim=0)

    return sino_star
