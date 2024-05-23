#!/usr/bin/env python
"""Denoiser module for BM3D-ORNL."""

import logging
import numpy as np
from typing import Tuple
from scipy.signal import medfilt2d
from bm3dornl.aggregation import aggregate_patches
from bm3dornl.block_matching import PatchManager
from bm3dornl.gpu_utils import (
    hard_thresholding,
    wiener_hadamard,
    memory_cleanup,
)
from bm3dornl.utils import (
    horizontal_binning,
    horizontal_debinning,
    estimate_noise_std,
    estimate_noise_free_sinogram,
)


class BM3D:
    def __init__(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int] = (8, 8),
        stride: int = 3,
        background_threshold: float = 0.1,
    ):
        """
        Initialize the BM3D class with an image and configuration parameters for patch management and denoising.

        Parameters
        ----------
        image : np.ndarray
            The sinogram or image to be denoised.
        patch_size : tuple
            Dimensions (height, width) of each patch. Default is (8, 8).
        stride : int
            The stride with which to slide the window across the image. Default is 3.
        background_threshold : float
            The mean intensity threshold below which a patch is considered a background patch.
        """
        self.image = np.asarray(image)
        self.estimate_denoised_image = np.zeros_like(self.image, dtype=float)
        self.final_denoised_image = np.zeros_like(self.image, dtype=float)
        # Initialize the PatchManager
        self.patch_manager = PatchManager(
            self.image,
            patch_size=patch_size,
            stride=stride,
            background_threshold=background_threshold,
        )
        # record input parameters
        self.patch_size = patch_size
        self.stride = stride
        self.background_threshold = background_threshold

    def group_signal_patches(
        self, cut_off_distance: Tuple[int, int], intensity_diff_threshold: float
    ):
        """
        Group signal patches into blocks based on spatial and intensity distance thresholds.

        Parameters:
        ----------
        cut_off_distance : tuple
            Maximum spatial distance in terms of row and column indices for patches in the same block.
        intensity_diff_threshold : float
            Maximum Euclidean distance in intensity for patches to be considered similar.
        """
        self.patch_manager.group_signal_patches(
            cut_off_distance, intensity_diff_threshold
        )
        logging.info(
            f"Total number of signal patches: {len(self.patch_manager.signal_patches_pos)}"
        )

    def thresholding(
        self,
        cut_off_distance: Tuple[int, int],
        intensity_diff_threshold: float,
        num_patches_per_group: int,
        threshold: float,
    ) -> np.ndarray:
        """
        Perform the denoising process using the specified configuration.

        Parameters:
        ----------
        cut_off_distance : tuple
            Maximum spatial distance in terms of row and column indices for patches in the same block.
        intensity_diff_threshold : float
            Maximum Euclidean distance in intensity for patches to be considered similar.
        num_patches_per_group : int
            The number of patchs in each block.
        threshold : float
            The threshold value for hard thresholding during the first pass.

        Returns:
        -------
        np.ndarray
            The denoised image estimate.
        """
        self.group_signal_patches(cut_off_distance, intensity_diff_threshold)

        weights = np.zeros_like(self.image, dtype=float)

        logging.info("Block matching for 1st pass...")
        block, positions = self.patch_manager.get_hyper_block(
            num_patches_per_group=num_patches_per_group, padding_mode="circular"
        )

        logging.info("Applying shrinkage...")
        block = hard_thresholding(block, threshold)

        # manual release of memory
        memory_cleanup()

        # Aggreation
        # NOTE: this part needs optimization (numba or parallel or both)
        logging.info("Aggregating...")
        aggregate_patches(
            estimate_denoised_image=self.estimate_denoised_image,
            weights=weights,
            hyper_block=block,
            hyper_block_index=positions,
        )

        # Normalize by the weights to compute the average
        self.estimate_denoised_image /= np.maximum(weights, 1)

    def re_filtering(
        self,
        cut_off_distance: Tuple[int, int],
        intensity_diff_threshold: float,
        num_patches_per_group: int,
    ):
        """
        Perform the second step for BM3D, re-filter using estimates as reference noisy free image.

        Parameters
        ----------
        cut_off_distance : tuple
            Maximum spatial distance in terms of row and column indices for patches in the same block.
        intensity_diff_threshold : float
            Maximum Euclidean distance in intensity for patches to be considered similar.
        num_patches_per_group : int
            The number of patch in each block.
        """
        # assume the patch manager has been update to use the estimate_denoised_image
        # NOTE: this should give us better blocks as we are using a noise reduced image as reference
        self.group_signal_patches(cut_off_distance, intensity_diff_threshold)

        weights = np.zeros_like(self.image, dtype=np.float64)

        # estimate the noise
        sigma_squared = estimate_noise_std(
            noisy_image=self.image, noise_free_image=self.estimate_denoised_image
        )

        logging.info("Block matching for 2nd pass...")
        block, positions = self.patch_manager.get_hyper_block(
            num_patches_per_group=num_patches_per_group,
            padding_mode="circular",
            alternative_source=self.image,  # use the original image
        )

        logging.info("Wiener-Hadamard filtering...")
        block = wiener_hadamard(block, sigma_squared)  # why does this work?

        # manual release of memory
        memory_cleanup()

        # Aggreation
        # NOTE: this part needs optimization (numba or parallel or both)
        logging.info("Aggregating...")
        aggregate_patches(
            estimate_denoised_image=self.final_denoised_image,
            weights=weights,
            hyper_block=block,
            hyper_block_index=positions,
        )

        # Normalize by the weights to compute the average
        self.final_denoised_image /= np.maximum(weights, 1)

    def denoise(
        self,
        cut_off_distance: Tuple[int, int],
        intensity_diff_threshold: float,
        num_patches_per_group: int,
        threshold: float,
        fast_estimate: bool = True,
    ):
        """
        Perform the BM3D denoising process on the input image.

        Parameters:
        ----------
        cut_off_distance : tuple
            Maximum spatial distance in terms of row and column indices for patches in the same block.
        intensity_diff_threshold : float
            Maximum Euclidean distance in intensity for patches to be considered similar.
        num_patches_per_group : int
            The number of patch in each block.
        threshold : float
            The threshold value for hard thresholding during the first pass.
        fast_estimate : bool
            Whether to use a fast estimate for the denoised image. Default is True.
        """
        # step 1: estimate the noise free image
        logging.info("Estimating noise free image...")
        if fast_estimate:
            logging.info("Using fast estimate")
            self.estimate_denoised_image = estimate_noise_free_sinogram(
                sinogram=self.image,
                background_estimate=self.background_threshold,
            )
        else:
            logging.info("Using block-matching with hard thresholding")
            self.thresholding(
                cut_off_distance=cut_off_distance,
                intensity_diff_threshold=intensity_diff_threshold,
                num_patches_per_group=num_patches_per_group,
                threshold=threshold,
            )

        # step 2: update patch manager with the estimate_denoised_image
        self.patch_manager.image = self.estimate_denoised_image

        # step 3: re-filtering
        logging.info("Second pass: Re-filtering")
        self.re_filtering(
            cut_off_distance=cut_off_distance,
            intensity_diff_threshold=intensity_diff_threshold,
            num_patches_per_group=num_patches_per_group,
        )


def bm3d_streak_removal(
    sinogram: np.ndarray,
    background_threshold: float = 0.1,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    cut_off_distance: Tuple[int, int] = (64, 64),
    intensity_diff_threshold: float = 0.1,
    num_patches_per_group: int = 400,
    shrinkage_threshold: float = 0.1,
    k: int = 4,
    fast_estimate: bool = True,
) -> np.ndarray:
    """Multiscale BM3D for streak removal

    Parameters
    ----------
    sinogram : np.ndarray
        The input sinogram to be denoised.
    background_threshold: float
        Estimated background intensity threshold, default to 0.1.
    patch_size : tuple[int, int], optional
        The size of the patches, by default (8, 8)
    stride:
        Steps when generating blocks with sliding window.
    cut_off_distance : tuple
        Maximum spatial distance in terms of row and column indices for patches in the same block.
    intensity_diff_threshold : float, optional
        The threshold for patch similarity, by default 0.01
    num_patches_per_group : int
        The number of patch in each block.
    shrinkage_threshold : float, optional
        The threshold for hard thresholding, by default 0.2
    k : int, optional
        The number of iterations for horizontal binning, by default 3
    fast_estimate : bool, optional
        Whether to use a fast estimate for the denoised image, by default True

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

    if k == 0:
        # direct without multi-scale
        worker = BM3D(
            image=sino_star,
            patch_size=patch_size,
            stride=stride,
            background_threshold=background_threshold,
        )
        worker.denoise(
            cut_off_distance=cut_off_distance,
            intensity_diff_threshold=intensity_diff_threshold,
            num_patches_per_group=num_patches_per_group,
            threshold=shrinkage_threshold,
            fast_estimate=fast_estimate,
        )
        return worker.final_denoised_image

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
            worker = BM3D(
                image=sino_star,
                patch_size=patch_size,
                stride=stride,
                background_threshold=background_threshold,
            )
            worker.denoise(
                cut_off_distance=cut_off_distance,
                intensity_diff_threshold=intensity_diff_threshold,
                num_patches_per_group=num_patches_per_group,
                threshold=shrinkage_threshold,
                fast_estimate=fast_estimate,
            )
            noise_estimate = sino - worker.final_denoised_image

    return sino_star
