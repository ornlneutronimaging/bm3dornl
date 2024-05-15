#!/usr/bin/env python3
"""Block matching to build hyper block from single sinogram."""

import numpy as np
from typing import Tuple, Optional
from bm3dornl.utils import (
    get_signal_patch_positions,
    find_candidate_patch_ids,
    is_within_threshold,
    pad_patch_ids,
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
        i, j = position
        return source_image[i : i + self.patch_size[0], j : j + self.patch_size[1]]

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
        self.signal_blocks_matrix = np.eye(num_patches, dtype=bool)

        # Cache patches as views
        cached_patches = [self.get_patch(pos) for pos in self.signal_patches_pos]

        for ref_patch_id in range(num_patches):
            ref_patch = cached_patches[ref_patch_id]
            candidate_patch_ids = find_candidate_patch_ids(
                self.signal_patches_pos, ref_patch_id, cut_off_distance
            )
            # iterate over the candidate patches
            for neightbor_patch_id in candidate_patch_ids:
                if is_within_threshold(
                    ref_patch,
                    cached_patches[neightbor_patch_id],
                    intensity_diff_threshold,
                ):
                    self.signal_blocks_matrix[ref_patch_id, neightbor_patch_id] = True
                    self.signal_blocks_matrix[neightbor_patch_id, ref_patch_id] = True

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

        TODO:
        -----
        - use multi-processing to further improve the speed of block building
        """
        group_size = len(self.signal_blocks_matrix)
        block = np.empty(
            (group_size, num_patches_per_group, *self.patch_size), dtype=np.float32
        )
        positions = np.empty((group_size, num_patches_per_group, 2), dtype=np.int32)

        for i, row in enumerate(self.signal_blocks_matrix):
            candidate_patch_ids = np.where(row)[0]
            padded_patch_ids = pad_patch_ids(
                candidate_patch_ids, num_patches_per_group, mode=padding_mode
            )
            # update block and positions
            block[i] = np.array(
                [
                    self.get_patch(self.signal_patches_pos[idx], alternative_source)
                    for idx in padded_patch_ids
                ]
            )
            positions[i] = np.array(self.signal_patches_pos[padded_patch_ids])

        return block, positions
