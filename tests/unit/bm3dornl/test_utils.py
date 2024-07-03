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
    size_x, size_y = 64, 64
    k = 6
    # Initial setup: Create a test image
    Z = np.random.rand(size_y, size_y)

    # Assert that each image has the correct dimensions
    expected_width = size_x
    for i in range(k):
        # Perform the binning
        expected_width = (expected_width + 1) // 2  # Calculate the next expected width
        binned_image = horizontal_binning(Z, fac=2)
        assert binned_image.shape[0] == 64, f"Height of image {i} is incorrect"
        assert (
            binned_image.shape[1] == expected_width
        ), f"Width of image {i} is incorrect"
        Z = binned_image


@pytest.mark.parametrize(
    "original_width, target_width", [(32, 64), (64, 128), (128, 256)]
)
def test_horizontal_debinning_scaling(original_width, target_width):
    original_image = np.random.rand(64, original_width)
    target_shape = (64, target_width)
    debinned_image = horizontal_debinning(
        original_image, target_width, fac=2, dim=1, n_iter=1
    )
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
