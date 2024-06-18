#!/usr/bin/env python
"""Denoiser module for BM3D-ORNL."""

import logging
import numpy as np
from typing import Tuple, Optional
from scipy.signal import medfilt2d
from scipy.signal import convolve, convolve2d
from scipy.interpolate import interp1d
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

def _create_arr_for_bin(base_arr: np.ndarray, h: int, dim: int):
    """
    Create a padded and convolved array used in both binning and debinning.
    :param base_arr: Input array
    :param h: bin count
    :param dim: bin dimension (0 or 1)
    :return: resulting array
    """
    mod_pad = h - ((base_arr.shape[dim] - 1) % h) - 1
    if dim == 0:
        pads = ((0, mod_pad), (0, 0))
        kernel = np.ones((h, 1))
    else:
        pads = ((0, 0), (0, mod_pad))
        kernel = np.ones((1, h))

    return convolve2d(np.pad(base_arr, pads, 'symmetric'), kernel, 'same', 'fill')

def _bin_in_1d(z: np.ndarray, h: int = 2, dim: int = 1) -> np.ndarray:
    """
    Bin a 2-D array across a dimension
    :param z: input data (2-D)
    :param h: divisor for binning
    :param dim: dimension for binning (0 or 1)
    :return: binned data
    """

    if h > 1:
        h_half = h // 2
        z_bin = _create_arr_for_bin(z, h, dim)

        # get coordinates of bin centres
        if dim == 0:
            z_bin = z_bin[h_half + ((h % 2) == 1): z_bin.shape[dim] - h_half + 1: h, :]
        else:
            z_bin = z_bin[:, h_half + ((h % 2) == 1): z_bin.shape[dim] - h_half + 1: h]

        return z_bin

    return z


def _debin_in_1d(z: np.ndarray, size: int, h: int, n_iter: int, dim: int = 1) -> np.ndarray:
    """
    De-bin a 2-D array across a dimension
    :param z: input data (2-D)
    :param size: target size (original size before binning) for the second dimension
    :param h: binning factor (original divisor)
    :param n_iter: number of iterations
    :param dim: dimension for binning (0 or 1)
    :return: debinned image, size of "size" across bin dimension
    """
    if h <= 1:
        return np.copy(z)

    h_half = h // 2

    if dim == 0:
        base_arr = np.ones((size, 1))
    else:
        base_arr = np.ones((1, size))

    n_counter = _create_arr_for_bin(base_arr, h, dim)

    # coordinates of bin counts
    x1c = np.arange(h_half + ((h % 2) == 1), (z.shape[dim]) * h, h)
    x1 = np.arange(h_half + 1 - ((h % 2) == 0) / 2, (z.shape[dim]) * h, h)

    # coordinates of image pixels
    ix1 = np.arange(1, size + 1)

    y_j = 0

    for jj in range(max(1, n_iter)):
        # residual
        if jj > 0:
            r_j = z - _bin_in_1d(y_j, h, dim)
        else:
            r_j = z

        # interpolation
        if dim == 0:
            interp = interp1d(x1, r_j / n_counter[x1c, :], kind='cubic', fill_value='extrapolate', axis=0)
        else:
            interp = interp1d(x1, r_j / n_counter[:, x1c], kind='cubic', fill_value='extrapolate')
        y_j = y_j + interp(ix1)

    return y_j

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

    den = None
    max_bin_main = k
    bin_size_second = 32
    z = sino_star
    # Shrink along the first dimension
    shrunk_y = _bin_in_1d(z, bin_size_second, dim=0)
    zbins_orig = [np.copy(shrunk_y)]

    # Contains upscaled denoised sinograms
    zbins = [np.zeros(0)]

    # Bin horizontally
    for i in range(0, max_bin_main):
        zbins_orig.append(_bin_in_1d(zbins_orig[-1], 2, dim=1))
        zbins.append(np.zeros(0))

    zbins[-1] = zbins_orig[-1]

    for ii in range(max_bin_main, -1, -1):
        print("k:", ii)
        img = zbins[ii]

        # Denoise binned sinogram, if slice size is smaller than img, split before denoising and PSD est.
        #den = _denoise_in_slices(img, pro, psd_shapes[ii], slice_sizes[ii], slice_step_sizes[ii])
        den = bm3d_ring_artifact_removal(img, **kwargs)
        # For iterations except the last, create the next noisy image with a finer scale residual
        if ii > 0:
            debin = _debin_in_1d(den - zbins_orig[ii], zbins_orig[ii - 1].shape[1], 2, 30, dim=1)
            zbins[ii - 1] = zbins_orig[ii - 1] + debin

    # Vertical upscaling + residual
    den = z + _debin_in_1d(den - shrunk_y, z.shape[0], bin_size_second, 30, dim=0)
    sino_star = den

    return sino_star
