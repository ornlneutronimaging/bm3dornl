#!/usr/bin/env python3
"""Block matching to build hyper block from single sinogram."""

import numpy as np
from typing import Tuple, Optional
from bm3dornl.utils import (
    compute_hyper_block,
    compute_signal_blocks_matrix,
    get_patch_numba,
    get_signal_patch_positions,
)


class PatchManager:
    def __init__(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int] = (8, 8),
        stride: int = 1,
        background_threshold: float = 0.1,
    ):
        """
        Initialize the PatchManager with an image, patch configuration, and background threshold
        for distinguishing between signal and background patches.

        Parameters
        ----------
        image : np.ndarray
            The image from which patches will be managed.
        patch_size : tuple
            Dimensions (height, width) of each patch. Default is (8, 8).
        stride : int
            The stride with which to slide the window across the image. Default is 1.
        background_threshold : float
            The mean intensity threshold below which a patch is considered a background patch.
        """
        self._image = image
        self.patch_size = patch_size
        self.stride = stride
        self.background_threshold = background_threshold
        self.signal_patches_pos = []
        self.signal_blocks_matrix = []
        self._generate_patch_positions()

    def _generate_patch_positions(self):
        """Generate the positions of signal and background patches in the image."""
        self.signal_patches_pos = get_signal_patch_positions(
            self._image, self.patch_size, self.stride, self.background_threshold
        )

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self._generate_patch_positions()

    def get_patch(
        self, position: tuple, source_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Retreive a patch from the image at the specified position.

        Parameters:
        ----------
        position : tuple
            The row and column indices of the top-left corner of the patch.
        source_image : np.ndarray

        Returns:
        -------
        np.ndarray
            The patch extracted from the image.
        """
        source_image = self._image if source_image is None else source_image
        return get_patch_numba(source_image, position, self.patch_size)

    def group_signal_patches(
        self, cut_off_distance: tuple, intensity_diff_threshold: float
    ):
        """
        Group signal patches into blocks based on spatial and intensity distance thresholds.

        Parameters:
        ----------
        cut_off_distance : tuple
            Maximum spatial distance in terms of row and column indices for patches in the same block, Manhattan distance (taxi cab distance).
        intensity_diff_threshold : float
            Maximum Euclidean distance in intensity for patches to be considered similar.
        """
        num_patches = len(self.signal_patches_pos)
        # Initialize the signal blocks matrix
        # note:
        # - the matrix is symmetric
        # - the zero values means the patches are not similar
        # - the non-zero values are the Euclidean distance between the patches, i.e smaller values means smaller distance, higher similarity
        self.signal_blocks_matrix = np.zeros((num_patches, num_patches), dtype=float)

        # Cache patches as views
        cached_patches = np.array(
            [self.get_patch(pos) for pos in self.signal_patches_pos]
        )

        # Compute signal blocks matrix using Numba JIT for speed
        compute_signal_blocks_matrix(
            self.signal_blocks_matrix,
            cached_patches,
            np.array(self.signal_patches_pos),
            np.array(cut_off_distance),
            intensity_diff_threshold,
        )

    def get_hyper_block(
        self,
        num_patches_per_group: int,
        padding_mode="circular",
        alternative_source: np.ndarray = None,
    ):
        """
        Return groups of similar patches as 4D arrays with each group having a fixed number of patches.

        Parameters:
        ----------
        num_patches_per_group : int
            Number of patches in each group.
        padding_mode : str
            Mode for padding the patch IDs when the number of candidates is less than `num_patches_per_group`.
            Options are 'first', 'repeat_sequence', 'circular', 'mirror', 'random'.
        alternative_source : cp.ndarray
            An alternative source image to extract patches from. Default is None.

        Returns:
        -------
        tuple
            A tuple containing the 4D array of patch groups and the corresponding positions.
        """
        source_image = self._image if alternative_source is None else alternative_source
        block, positions = compute_hyper_block(
            self.signal_blocks_matrix,
            np.array(self.signal_patches_pos),
            self.patch_size,
            num_patches_per_group,
            source_image,
            padding_mode,
        )
        return block, positions
