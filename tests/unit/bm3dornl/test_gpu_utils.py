#!/usr/env/bin python3

"""Unit test for cupy utility module."""

import pytest
import numpy as np
import cupy as cp
from bm3dornl.gpu_utils import (
    hard_thresholding,
    wiener_hadamard,
    memory_cleanup,
)


@pytest.mark.cuda_required
def test_hard_thresholding():
    # Setup the patch block
    patch_block = np.random.rand(2, 5, 8, 8)  # Random block of patches on GPU
    threshold_quantile = 0.5  # Threshold for hard thresholding

    # Apply shrinkage
    denoised_block = hard_thresholding(patch_block, threshold_quantile)

    # Convert back to frequency domain to check thresholding
    dct_block_check = cp.fft.rfft2(cp.asarray(denoised_block), axes=(1, 2, 3)).get()
    threshold = np.quantile(np.abs(dct_block_check), threshold_quantile)

    # Test if all values in the DCT domain are either zero or above the threshold
    # Allow a small tolerance for floating point arithmetic issues
    tolerance = 1e-5
    assert np.all(
        (np.abs(dct_block_check) >= threshold - tolerance)
        | (np.abs(dct_block_check) < tolerance)
    ), "DCT coefficients are not correctly thresholded"

    # Check the shape is maintained
    assert (
        patch_block.shape == denoised_block.shape
    ), "Output shape does not match input shape"

    # Check for any values that should not have been zeroed out
    original_dct_block = cp.fft.rfft2(cp.asarray(patch_block), axes=(1, 2, 3)).get()
    threshold = np.quantile(np.abs(original_dct_block), threshold_quantile)
    should_not_change = np.abs(original_dct_block) >= threshold
    assert np.allclose(
        dct_block_check[should_not_change],
        original_dct_block[should_not_change],
        atol=tolerance,
    ), "Values that should not have been zeroed out have changed"

    # Cleanup GPU memory
    memory_cleanup()


@pytest.mark.cuda_required
def test_wiener_hadamard_3d_input():
    # Prepare a 3D patch block
    patch_block = np.random.rand(1000, 8, 8)  # 1000 patches of 8x8 pixels
    sigma_squared = 0.1

    # Apply the Wiener-Hadamard filter
    denoised_block = wiener_hadamard(patch_block, sigma_squared)

    # Check if the output dimensions match the input
    assert (
        patch_block.shape == denoised_block.shape
    ), "Output dimensions should match input dimensions"

    # Ensure changes were made to the patch block
    assert not cp.allclose(
        patch_block, denoised_block, atol=1e-3
    ), "No changes detected in the patch block after filtering"

    # Cleanup GPU memory
    memory_cleanup()


@pytest.mark.cuda_required
def test_wiener_hadamard_4d_input():
    # Prepare a 4D patch block
    patch_block = np.random.rand(
        4, 1000, 8, 8
    )  # 4 batches, 1000 patches each, of 8x8 pixels
    sigma_squared = 0.1

    # Apply the Wiener-Hadamard filter
    denoised_block = wiener_hadamard(patch_block, sigma_squared)

    # Check if the output dimensions match the input
    assert (
        patch_block.shape == denoised_block.shape
    ), "Output dimensions should match input dimensions"

    # Ensure changes were made to the patch block
    assert not np.allclose(
        patch_block, denoised_block, atol=1e-3
    ), "No changes detected in the patch block after filtering"

    # Cleanup GPU memory
    memory_cleanup()


@pytest.mark.cuda_required
def test_memory_cleanup(mocker):
    # Create mock objects for the method chains
    mock_free_all_blocks = mocker.Mock()
    mock_free_all_blocks_pinned = mocker.Mock()
    mock_synchronize = mocker.Mock()

    # Mock the chain calls
    mock_memory_pool = mocker.patch(
        "cupy.get_default_memory_pool", return_value=mock_free_all_blocks
    )
    mock_memory_pool().free_all_blocks = mock_free_all_blocks

    mock_pinned_memory_pool = mocker.patch(
        "cupy.get_default_pinned_memory_pool", return_value=mock_free_all_blocks_pinned
    )
    mock_pinned_memory_pool().free_all_blocks = mock_free_all_blocks_pinned

    mocker.patch("cupy.cuda.Stream.null.synchronize", mock_synchronize)

    # Call the function
    memory_cleanup()

    # Check if the functions were called
    mock_free_all_blocks.assert_called_once()
    mock_free_all_blocks_pinned.assert_called_once()
    mock_synchronize.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])