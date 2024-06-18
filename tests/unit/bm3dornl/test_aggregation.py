import numpy as np
import pytest
from bm3dornl.aggregation import (
    aggregate_block_to_image,
    aggregate_denoised_block_to_image,
)


@pytest.fixture
def sample_data():
    image_shape = (64, 64)
    patch_size = (8, 8)
    num_blocks = 10
    num_patches_per_block = 5

    hyper_blocks = np.random.rand(
        num_blocks, num_patches_per_block, patch_size[0], patch_size[1]
    ).astype(np.float32)
    variance_blocks = np.random.rand(
        num_blocks, num_patches_per_block, patch_size[0], patch_size[1]
    ).astype(np.float32)
    hyper_block_indices = np.random.randint(
        0, image_shape[0] - patch_size[0], size=(num_blocks, num_patches_per_block, 2)
    ).astype(np.int32)

    denoised_patches = np.random.rand(
        num_blocks, num_patches_per_block, patch_size[0], patch_size[1]
    ).astype(np.float32)
    patch_positions = hyper_block_indices

    return (
        image_shape,
        hyper_blocks,
        hyper_block_indices,
        variance_blocks,
        denoised_patches,
        patch_positions,
    )


def test_aggregate_block_to_image(sample_data):
    image_shape, hyper_blocks, hyper_block_indices, variance_blocks, _, _ = sample_data

    denoised_image = aggregate_block_to_image(
        image_shape, hyper_blocks, hyper_block_indices, variance_blocks
    )

    assert denoised_image.shape == image_shape
    assert np.all(denoised_image >= 0)
    assert np.all(np.isfinite(denoised_image))


def test_aggregate_denoised_block_to_image(sample_data):
    image_shape, _, _, _, denoised_patches, patch_positions = sample_data

    final_denoised_image = aggregate_denoised_block_to_image(
        image_shape, denoised_patches, patch_positions
    )

    assert final_denoised_image.shape == image_shape
    assert np.all(final_denoised_image >= 0)
    assert np.all(np.isfinite(final_denoised_image))


def test_aggregate_block_to_image_zero_variance():
    image_shape = (16, 16)
    patch_size = (4, 4)
    num_blocks = 2
    num_patches_per_block = 3

    hyper_blocks = np.random.rand(
        num_blocks, num_patches_per_block, patch_size[0], patch_size[1]
    ).astype(np.float32)
    variance_blocks = np.zeros(
        (num_blocks, num_patches_per_block, patch_size[0], patch_size[1])
    ).astype(np.float32)
    hyper_block_indices = np.random.randint(
        0, image_shape[0] - patch_size[0], size=(num_blocks, num_patches_per_block, 2)
    ).astype(np.int32)

    denoised_image = aggregate_block_to_image(
        image_shape, hyper_blocks, hyper_block_indices, variance_blocks
    )

    assert denoised_image.shape == image_shape


def test_aggregate_denoised_block_to_image_uniform_weights():
    image_shape = (16, 16)
    patch_size = (4, 4)
    num_blocks = 2
    num_patches_per_block = 3

    denoised_patches = np.ones(
        (num_blocks, num_patches_per_block, patch_size[0], patch_size[1])
    ).astype(np.float32)
    patch_positions = np.random.randint(
        0, image_shape[0] - patch_size[0], size=(num_blocks, num_patches_per_block, 2)
    ).astype(np.int32)

    final_denoised_image = aggregate_denoised_block_to_image(
        image_shape, denoised_patches, patch_positions
    )

    assert final_denoised_image.shape == image_shape
    assert np.all(final_denoised_image >= 0)
    assert np.all(np.isfinite(final_denoised_image))


def test_aggregate_denoised_block_to_image_non_overlapping():
    image_shape = (16, 16)

    denoised_patches = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )
    patch_positions = np.array([[[0, 0]]], dtype=np.int32)

    final_denoised_image = aggregate_denoised_block_to_image(
        image_shape, denoised_patches, patch_positions
    )

    expected_image = np.zeros(image_shape, dtype=np.float32)
    expected_image[:4, :4] = denoised_patches[0, 0]

    assert final_denoised_image.shape == image_shape
    assert np.allclose(final_denoised_image, expected_image)


if __name__ == "__main__":
    pytest.main([__file__])
