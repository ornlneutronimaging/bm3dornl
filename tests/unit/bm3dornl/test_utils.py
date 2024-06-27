#!/usr/bin/env python3

"""Unit tests for the utility module."""

import pytest
import numpy as np
from bm3dornl.utils import (
    estimate_background_intensity,
    is_within_threshold,
    horizontal_binning,
    horizontal_debinning,
)


def test_is_within_threshold():
    # Setup the patches
    ref_patch = np.array([1, 2, 3], dtype=float)
    cmp_patch_same = np.array([1, 2, 3], dtype=float)
    cmp_patch_different = np.array([4, 5, 6], dtype=float)
    cmp_patch_close = np.array([1, 2, 4], dtype=float)

    # Test case 1: Same patches, zero distance
    threshold = 0
    result = is_within_threshold(ref_patch, cmp_patch_same, threshold)
    assert result, "Failed: Same patches should be within zero distance"

    # Test case 2: Different patches, threshold less than actual distance
    threshold = 2
    result = is_within_threshold(ref_patch, cmp_patch_different, threshold)
    assert not result, "Failed: Different patches should not be within distance of 2"

    # Test case 3: Different patches, threshold greater than actual distance
    threshold = 6
    result = is_within_threshold(ref_patch, cmp_patch_different, threshold)
    assert result, "Failed: Different patches should be within distance of 6"

    # Test case 4: Slightly different patches, small threshold
    threshold = 2
    result = is_within_threshold(ref_patch, cmp_patch_close, threshold)
    assert result, "Failed: Slightly different patches should be within distance of 2"

    # Test case 5: Slightly different patches, very small threshold
    threshold = 0.1
    result = is_within_threshold(ref_patch, cmp_patch_close, threshold)
    assert not result, "Failed: Slightly different patches should not be within very small distance of 0.1"


def test_horizontal_binning():
    # Initial setup: Create a test image
    Z = np.random.rand(64, 64)

    # Number of binning iterations
    k = 3

    # Perform the binning
    binned_images = horizontal_binning(Z, k)

    # Assert we have the correct number of images
    assert len(binned_images) == k + 1, "Incorrect number of binned images returned"

    # Assert that each image has the correct dimensions
    expected_width = 64
    for i, img in enumerate(binned_images):
        assert img.shape[0] == 64, f"Height of image {i} is incorrect"
        assert img.shape[1] == expected_width, f"Width of image {i} is incorrect"
        expected_width = (expected_width + 1) // 2  # Calculate the next expected width


def test_horizontal_binning_k_zero():
    Z = np.random.rand(64, 64)
    binned_images = horizontal_binning(Z, 0)
    assert len(binned_images) == 1 and np.array_equal(
        binned_images[0], Z
    ), "Binning with k=0 should return only the original image"


def test_horizontal_binning_large_k():
    Z = np.random.rand(64, 64)
    binned_images = horizontal_binning(Z, 6)
    assert len(binned_images) == 7, "Incorrect number of images for large k"
    assert binned_images[-1].shape[1] == 1, "Final image width should be 1 for large k"


@pytest.mark.parametrize(
    "original_width, target_width", [(32, 64), (64, 128), (128, 256)]
)
def test_horizontal_debinning_scaling(original_width, target_width):
    original_image = np.random.rand(64, original_width)
    target_shape = (64, target_width)
    debinned_image = horizontal_debinning(original_image, np.empty(target_shape))
    assert (
        debinned_image.shape == target_shape
    ), f"Failed to scale from {original_width} to {target_width}"


def test_estimate_background_intensity():
    # Create a sample 3D tomostack
    tomostack = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            [[2, 4, 6], [8, 10, 12], [14, 16, 18]],
        ]
    )

    # Calculate the expected background intensity (5% quantile of the mean along axis 1)
    expected_intensity = np.quantile(np.mean(tomostack, axis=1), 0.05)

    # Call the function
    result = estimate_background_intensity(tomostack, quantile=0.05)

    # Assert the result is as expected
    assert np.isclose(
        result, expected_intensity
    ), f"Expected {expected_intensity}, but got {result}"


if __name__ == "__main__":
    pytest.main([__file__])
