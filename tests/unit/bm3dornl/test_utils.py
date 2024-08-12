#!/usr/bin/env python3

"""Unit tests for the utility module."""

import pytest
import numpy as np
from bm3dornl.utils import (
    estimate_background_intensity,
    is_within_threshold,
    downscale_2d_horizontal,
    upscale_2d_horizontal,
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


def test_upscale_basic():
    # Test a basic upscaling
    array = np.array([[1, 2], [3, 4]])
    scale_factor = 2
    original_width = 4
    result = upscale_2d_horizontal(array, scale_factor, original_width)

    assert result.shape == (2, 4)  # Ensure upscaled shape is correct


def test_upscale_with_refinement():
    # Test upscaling with iterative refinement
    array = np.array([[1, 2], [3, 4]])
    scale_factor = 2
    original_width = 4
    result = upscale_2d_horizontal(
        array, scale_factor, original_width, use_iterative_refinement=True
    )

    assert result.shape == (2, 4)  # Ensure upscaled shape is correct


def test_upscale_recover_original():
    # Test upscaling to recover the original array
    original_array = np.random.rand(8, 8)
    scale_factor = 2
    downscaled_array = downscale_2d_horizontal(original_array, scale_factor)
    upscaled_array = upscale_2d_horizontal(
        downscaled_array, scale_factor, original_array.shape[1]
    )

    # ensure the shape is correct
    assert upscaled_array.shape == original_array.shape


def test_upscale_invalid_input():
    # Test with invalid input (e.g., scale_factor = 0)
    array = np.array([[1, 2], [3, 4]])
    scale_factor = 0
    original_width = 4
    with pytest.raises(ValueError):
        upscale_2d_horizontal(array, scale_factor, original_width)


if __name__ == "__main__":
    pytest.main([__file__])
