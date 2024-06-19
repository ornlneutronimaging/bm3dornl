import numpy as np
import pytest
from bm3dornl.block_matching import (
    find_candidate_patch_ids,
    get_signal_patch_positions,
    pad_patch_ids,
    get_patch_numba,
    compute_variance_weighted_distance_matrix,
    compute_distance_matrix_no_variance,
    form_hyper_blocks_from_distance_matrix,
    form_hyper_blocks_from_two_images,
)


@pytest.fixture
def sample_image():
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def sample_patches(sample_image):
    patch_size = (8, 8)
    stride = 3
    background_threshold = 0.1
    return get_signal_patch_positions(
        sample_image, patch_size, stride, background_threshold
    )


def test_find_candidate_patch_ids(sample_patches):
    cut_off_distance = (5, 5)
    candidate_patch_ids = find_candidate_patch_ids(sample_patches, 0, cut_off_distance)
    assert isinstance(candidate_patch_ids, list)
    assert len(candidate_patch_ids) > 0


def test_get_signal_patch_positions(sample_image):
    patch_size = (8, 8)
    stride = 3
    background_threshold = 0.1
    signal_patches = get_signal_patch_positions(
        sample_image, patch_size, stride, background_threshold
    )
    assert signal_patches.ndim == 2
    assert signal_patches.shape[1] == 2


def test_pad_patch_ids():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 5
    padded_ids_first = pad_patch_ids(candidate_patch_ids, num_patches, mode="first")
    padded_ids_repeat = pad_patch_ids(
        candidate_patch_ids, num_patches, mode="repeat_sequence"
    )
    padded_ids_circular = pad_patch_ids(
        candidate_patch_ids, num_patches, mode="circular"
    )
    padded_ids_mirror = pad_patch_ids(candidate_patch_ids, num_patches, mode="mirror")
    padded_ids_random = pad_patch_ids(candidate_patch_ids, num_patches, mode="random")

    assert len(padded_ids_first) == num_patches
    assert len(padded_ids_repeat) == num_patches
    assert len(padded_ids_circular) == num_patches
    assert len(padded_ids_mirror) == num_patches
    assert len(padded_ids_random) == num_patches


def test_get_patch_numba(sample_image):
    position = (5, 5)
    patch_size = (8, 8)
    patch = get_patch_numba(sample_image, position, patch_size)
    assert patch.shape == patch_size


def test_compute_variance_weighted_distance_matrix(sample_patches):
    num_patches = len(sample_patches)
    transformed_patches = np.random.rand(num_patches, 8, 8).astype(np.float32)
    variances = np.random.rand(num_patches, 8, 8).astype(np.float32)
    cut_off_distance = (5, 5)
    distance_matrix = compute_variance_weighted_distance_matrix(
        transformed_patches, variances, sample_patches, cut_off_distance
    )
    assert distance_matrix.shape == (num_patches, num_patches)
    assert np.all(distance_matrix >= 0)


def test_compute_distance_matrix_no_variance(sample_patches):
    num_patches = len(sample_patches)
    transformed_patches = np.random.rand(num_patches, 8, 8).astype(np.float32)
    cut_off_distance = (5, 5)
    distance_matrix = compute_distance_matrix_no_variance(
        transformed_patches, sample_patches, cut_off_distance
    )
    assert distance_matrix.shape == (num_patches, num_patches)
    assert np.all(distance_matrix >= 0)


def test_form_hyper_blocks_from_distance_matrix(sample_image, sample_patches):
    num_patches = len(sample_patches)
    patch_size = (8, 8)
    num_patches_per_group = 5
    transformed_patches = np.random.rand(
        num_patches, patch_size[0], patch_size[1]
    ).astype(np.float32)
    variances = np.random.rand(num_patches, patch_size[0], patch_size[1]).astype(
        np.float32
    )
    cut_off_distance = (5, 5)
    distance_matrix = compute_variance_weighted_distance_matrix(
        transformed_patches, variances, sample_patches, cut_off_distance
    )
    blocks, positions, blocks_var = form_hyper_blocks_from_distance_matrix(
        distance_matrix,
        sample_patches,
        patch_size,
        num_patches_per_group,
        sample_image,
        variances,
    )

    assert blocks.shape == (
        num_patches,
        num_patches_per_group,
        patch_size[0],
        patch_size[1],
    )
    assert positions.shape == (num_patches, num_patches_per_group, 2)
    assert blocks_var.shape == (
        num_patches,
        num_patches_per_group,
        patch_size[0],
        patch_size[1],
    )


def test_form_hyper_blocks_from_two_images(sample_image, sample_patches):
    num_patches = len(sample_patches)
    patch_size = (8, 8)
    num_patches_per_group = 5
    transformed_patches = np.random.rand(
        num_patches, patch_size[0], patch_size[1]
    ).astype(np.float32)
    variances = np.random.rand(num_patches, patch_size[0], patch_size[1]).astype(
        np.float32
    )
    cut_off_distance = (5, 5)
    distance_matrix = compute_variance_weighted_distance_matrix(
        transformed_patches, variances, sample_patches, cut_off_distance
    )
    blocks1, blocks2, positions, blocks_var1 = form_hyper_blocks_from_two_images(
        distance_matrix,
        sample_patches,
        patch_size,
        num_patches_per_group,
        sample_image,
        sample_image,
        variances,
    )

    assert blocks1.shape == (
        num_patches,
        num_patches_per_group,
        patch_size[0],
        patch_size[1],
    )
    assert blocks2.shape == (
        num_patches,
        num_patches_per_group,
        patch_size[0],
        patch_size[1],
    )
    assert positions.shape == (num_patches, num_patches_per_group, 2)
    assert blocks_var1.shape == (
        num_patches,
        num_patches_per_group,
        patch_size[0],
        patch_size[1],
    )


if __name__ == "__main__":
    pytest.main([__file__])
