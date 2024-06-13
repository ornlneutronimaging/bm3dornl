#!/usr/bin/env python3
"""Block matching to build hyper block from single sinogram."""

import numpy as np
from typing import Tuple
from numba import njit, prange
from bm3dornl.utils import (
    pad_patch_ids,
    find_candidate_patch_ids,
)


@njit
def get_signal_patch_positions(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    background_threshold: float = 0.1,
) -> np.ndarray:
    """Segment an image into signal patches.

    Parameters
    ----------
    image : np.ndarray
        The input image to be segmented into patches.
    patch_size : Tuple[int, int]
        The size of the patches to be extracted.
    stride : int
        The stride for patch extraction.
    background_threshold : float
        The threshold for determining background patches.

    Returns
    -------
    signal_patches : np.ndarray
        An array of positions of signal patches.

    NOTE
    ----
    Numba has issues with return a Tuple of np.ndarray, and since we only operates on signal patches,
    we will ignore the background patches for now.
    """
    i_height, i_width = image.shape
    p_height, p_width = patch_size

    signal_patches = []

    for r in range(0, i_height - p_height + 1, stride):
        for c in range(0, i_width - p_width + 1, stride):
            patch = image[r : r + p_height, c : c + p_width]
            patch_max = np.max(patch)
            if patch_max >= background_threshold:
                signal_patches.append((r, c))

    # deal with empty list
    # Note: raise error when couldn't find a single signal patch from the entire
    #       sinogram, which usually indicating a bad background estimation.
    if len(signal_patches) == 0:
        raise ValueError(
            "Couldn't find any signal patches in the image! Please check the background threshold."
        )

    return np.array(signal_patches)


@njit
def get_patch_numba(
    image: np.ndarray, position: Tuple[int, int], patch_size: Tuple[int, int]
) -> np.ndarray:
    """
    Retrieve a patch from the image at the specified position.

    Parameters
    ----------
    image : np.ndarray
        The image from which the patch is to be extracted.
    position : tuple
        The row and column indices of the top-left corner of the patch.
    patch_size : tuple
        The size of the patch to be extracted.

    Returns
    -------
    np.ndarray
        The patch extracted from the image.
    """
    i, j = position
    return image[i : i + patch_size[0], j : j + patch_size[1]]


@njit(parallel=True)
def compute_variance_weighted_distance_matrix(
    transformed_patches: np.ndarray,
    variances: np.ndarray,
    patch_positions: np.ndarray,
    cut_off_distance: Tuple[int, int],
) -> np.ndarray:
    """
    Compute the distance matrix for the given signal patches in the frequency domain, considering noise variances.

    Parameters
    ----------
    transformed_patches : np.ndarray
        The transformed (frequency domain) patches for the signal blocks.
    variances : np.ndarray
        The variances for the transformed patches.
    patch_positions : np.ndarray
        The positions of the signal patches.
    cut_off_distance : tuple
        The maximum Manhattan distance for considering candidate patches.

    Returns
    -------
    np.ndarray
        The computed distance matrix.
    """
    num_patches = len(patch_positions)
    distance_matrix = np.zeros((num_patches, num_patches), dtype=float)

    for ref_patch_id in prange(num_patches):
        ref_patch = transformed_patches[ref_patch_id]
        ref_variance = variances[ref_patch_id]
        # note:
        # find_candidate_patch_ids will add the ref_patch_id to the list of candidate patch ids
        candidate_patch_ids = find_candidate_patch_ids(
            patch_positions, ref_patch_id, cut_off_distance
        )
        for neighbor_patch_id in candidate_patch_ids:
            neighbor_patch_id = int(neighbor_patch_id)
            neighbor_patch = transformed_patches[neighbor_patch_id]
            neighbor_variance = variances[neighbor_patch_id]
            # Compute weighted L2 norm distance between patches in the frequency domain
            val_diff = 0.0
            for i in range(ref_patch.shape[0]):
                for j in range(ref_patch.shape[1]):
                    variance_avg = (ref_variance[i, j] + neighbor_variance[i, j]) / 2
                    val_diff += (
                        (ref_patch[i, j] - neighbor_patch[i, j]) ** 2
                    ) / variance_avg
            val_diff = val_diff**0.5
            val_diff = max(val_diff, 1e-8)
            distance_matrix[ref_patch_id, neighbor_patch_id] = val_diff
            distance_matrix[neighbor_patch_id, ref_patch_id] = val_diff

    return distance_matrix


@njit(parallel=True)
def form_hyper_blocks_from_distance_matrix(
    distance_matrix: np.ndarray,
    patch_positions: np.ndarray,
    patch_size: Tuple[int, int],
    num_patches_per_group: int,
    image: np.ndarray,
    variances: np.ndarray,
    padding_mode: str = "circular",
):
    """
    Form hyper blocks (groups of patches) from the distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix containing distances between patches.
    patch_positions : np.ndarray
        The positions of the patches.
    patch_size : tuple
        The size of each patch (height, width).
    num_patches_per_group : int
        The number of patches in each group.
    image : np.ndarray
        The input image from which patches are extracted.
    variances : np.ndarray
        The variances of the patches in the transform domain.
    padding_mode : str, optional
        The padding mode for the hyper block, by default "circular".

    Returns
    -------
    block : np.ndarray
        The hyper blocks of patches.
    positions : np.ndarray
        The positions of the patches in the hyper blocks.
    variance_blocks : np.ndarray
        The variances of the patches in the hyper blocks.
    """
    group_size = len(distance_matrix)
    block = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )
    positions = np.empty((group_size, num_patches_per_group, 2), dtype=np.int32)
    variance_blocks = np.empty(
        (group_size, num_patches_per_group, patch_size[0], patch_size[1]),
        dtype=np.float32,
    )

    for i in prange(group_size):
        row = distance_matrix[i]
        candidate_patch_ids = np.where(row > 0)[0]
        candidate_patch_val = row[candidate_patch_ids]
        candidate_patch_ids = candidate_patch_ids[np.argsort(candidate_patch_val)]
        padded_patch_ids = pad_patch_ids(
            candidate_patch_ids, num_patches_per_group, mode=padding_mode
        )

        for j in range(num_patches_per_group):
            idx = padded_patch_ids[j]
            patch = get_patch_numba(image, patch_positions[idx], patch_size)
            block[i, j] = patch
            positions[i, j] = patch_positions[idx]
            variance_blocks[i, j] = variances[idx]

    return block, positions, variance_blocks


# def image_to_hyper_blocks(
#     image: np.ndarray,
#     patch_size: Tuple[int, int] = (8, 8),
#     stride: int = 3,
#     background_threshold: float = 1e-3,
#     cut_off_distance: Tuple[int, int] = (64, 64),
#     intensity_diff_threshold: float = 0.1,
#     num_patches_per_group: int = 100,
#     padding_mode: str = "circular",
#     alternative_source: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Generate hyper block from single image for block matching.

#     Parameters
#     ----------
#     image : np.ndarray
#         The image from which to extract patches.
#     patch_size : tuple
#         The size of the patches to extract.
#     stride : int
#         The stride with which to slide the patch window.
#     background_threshold : float
#         The threshold below which a patch is considered background.
#     cut_off_distance : tuple
#         The maximum distance between patches in a block.
#     intensity_diff_threshold : float
#         The maximum intensity difference between patches.
#     num_patches_per_group : int
#         The number of patches in each group.
#     padding_mode : str
#         The mode for padding the patch IDs.
#     alternative_source : np.ndarray
#         An alternative source image to extract patches from.

#     Returns
#     -------
#     tuple
#         A tuple containing the hyper block and the corresponding positions.
#     """
#     # get the patch positions, upper left corner of each patch
#     signal_patches_pos = get_signal_patch_positions(
#         image, patch_size, stride, background_threshold
#     )

#     # prepare distance matrix
#     num_patches = len(signal_patches_pos)
#     signal_blocks_matrix = np.zeros((num_patches, num_patches), dtype=float)

#     # cache the patches
#     cached_patches = np.array([get_patch_numba(image, pos, patch_size) for pos in signal_patches_pos])

#     # fill the matrix
#     # TODO: improve ordering function to consider noise variance
#     compute_signal_blocks_matrix(
#         signal_blocks_matrix=signal_blocks_matrix,
#         cached_patches=cached_patches,
#         signal_patches_pos=signal_patches_pos,
#         cut_off_distance=cut_off_distance,
#         intensity_diff_threshold=intensity_diff_threshold,
#     )

#     # compute hyper blocks and index
#     block, positions = compute_hyper_block(
#         signal_blocks_matrix=signal_blocks_matrix,
#         signal_patches_pos=signal_patches_pos,
#         patch_size=patch_size,
#         num_patches_per_group=num_patches_per_group,
#         image=image if alternative_source is None else alternative_source,
#         padding_mode=padding_mode,
#     )

#     return block, positions


# class PatchManager:
#     def __init__(
#         self,
#         image: np.ndarray,
#         patch_size: Tuple[int, int] = (8, 8),
#         stride: int = 1,
#         background_threshold: float = 0.1,
#     ):
#         """
#         Initialize the PatchManager with an image, patch configuration, and background threshold
#         for distinguishing between signal and background patches.

#         Parameters
#         ----------
#         image : np.ndarray
#             The image from which patches will be managed.
#         patch_size : tuple
#             Dimensions (height, width) of each patch. Default is (8, 8).
#         stride : int
#             The stride with which to slide the window across the image. Default is 1.
#         background_threshold : float
#             The mean intensity threshold below which a patch is considered a background patch.
#         """
#         self._image = image
#         self.patch_size = patch_size
#         self.stride = stride
#         self.background_threshold = background_threshold
#         self.signal_patches_pos = []
#         self.signal_blocks_matrix = []
#         self._generate_patch_positions()

#     def _generate_patch_positions(self):
#         """Generate the positions of signal and background patches in the image."""
#         self.signal_patches_pos = get_signal_patch_positions(
#             self._image, self.patch_size, self.stride, self.background_threshold
#         )

#     @property
#     def image(self):
#         return self._image

#     @image.setter
#     def image(self, value):
#         self._image = value
#         self._generate_patch_positions()

#     def get_patch(
#         self, position: tuple, source_image: Optional[np.ndarray] = None
#     ) -> np.ndarray:
#         """Retreive a patch from the image at the specified position.

#         Parameters:
#         ----------
#         position : tuple
#             The row and column indices of the top-left corner of the patch.
#         source_image : np.ndarray

#         Returns:
#         -------
#         np.ndarray
#             The patch extracted from the image.
#         """
#         source_image = self._image if source_image is None else source_image
#         return get_patch_numba(source_image, position, self.patch_size)

#     def group_signal_patches(
#         self, cut_off_distance: tuple, intensity_diff_threshold: float
#     ):
#         """
#         Group signal patches into blocks based on spatial and intensity distance thresholds.

#         Parameters:
#         ----------
#         cut_off_distance : tuple
#             Maximum spatial distance in terms of row and column indices for patches in the same block, Manhattan distance (taxi cab distance).
#         intensity_diff_threshold : float
#             Maximum Euclidean distance in intensity for patches to be considered similar.
#         """
#         num_patches = len(self.signal_patches_pos)
#         # Initialize the signal blocks matrix
#         # note:
#         # - the matrix is symmetric
#         # - the zero values means the patches are not similar
#         # - the non-zero values are the Euclidean distance between the patches, i.e smaller values means smaller distance, higher similarity
#         self.signal_blocks_matrix = np.zeros((num_patches, num_patches), dtype=float)

#         # Cache patches as views
#         cached_patches = np.array(
#             [self.get_patch(pos) for pos in self.signal_patches_pos]
#         )

#         # Compute signal blocks matrix using Numba JIT for speed
#         compute_signal_blocks_matrix(
#             self.signal_blocks_matrix,
#             cached_patches,
#             np.array(self.signal_patches_pos),
#             np.array(cut_off_distance),
#             intensity_diff_threshold,
#         )

#     def get_hyper_block(
#         self,
#         num_patches_per_group: int,
#         padding_mode="circular",
#         alternative_source: np.ndarray = None,
#     ):
#         """
#         Return groups of similar patches as 4D arrays with each group having a fixed number of patches.

#         Parameters:
#         ----------
#         num_patches_per_group : int
#             Number of patches in each group.
#         padding_mode : str
#             Mode for padding the patch IDs when the number of candidates is less than `num_patches_per_group`.
#             Options are 'first', 'repeat_sequence', 'circular', 'mirror', 'random'.
#         alternative_source : cp.ndarray
#             An alternative source image to extract patches from. Default is None.

#         Returns:
#         -------
#         tuple
#             A tuple containing the 4D array of patch groups and the corresponding positions.
#         """
#         source_image = self._image if alternative_source is None else alternative_source
#         block, positions = compute_hyper_block(
#             self.signal_blocks_matrix,
#             np.array(self.signal_patches_pos),
#             self.patch_size,
#             num_patches_per_group,
#             source_image,
#             padding_mode,
#         )
#         return block, positions
