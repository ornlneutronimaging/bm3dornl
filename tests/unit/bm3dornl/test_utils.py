#!/usr/bin/env python3

"""Unit tests for the utility module."""

import pytest
import numpy as np
from bm3dornl.utils import (
    find_candidate_patch_ids,
    is_within_threshold,
    get_signal_patch_positions,
    pad_patch_ids,
    horizontal_binning,
    horizontal_debinning,
)


def test_find_candidate_patch_ids():
    # Setup the signal patches and test various reference indices and cut-off distances
    signal_patches = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 2], [3, 3]])

    # Test case 1
    ref_index = 0
    cut_off_distance = (1, 1)
    expected = [
        1,
        3,
        4,
    ]  # Only patches within 1 unit from (0, 0) in both dimensions and are after index 0
    result = find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance)
    assert result == expected, "Test case 1 failed"

    # Test case 2
    ref_index = 2
    cut_off_distance = (2, 2)
    expected = [3, 4, 5]  # Indices that are within 2 units from (0, 2)
    result = find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance)
    assert result == expected, "Test case 2 failed"

    # Test case 3
    ref_index = 4
    cut_off_distance = (3, 3)
    expected = [5, 6]  # Indices that are within 3 units from (1, 1)
    result = find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance)
    assert result == expected, "Test case 3 failed"

    # Test case 4
    ref_index = 0
    cut_off_distance = (0, 0)
    expected = []  # No patch within 0 distance from (0, 0) except itself
    result = find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance)
    assert result == expected, "Test case 4 failed"


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


def test_get_signal_patch_positions():
    # Create a synthetic image with a signal patch in the center
    image = np.zeros((10, 10), dtype=float)
    image[4:6, 4:6] = 1.0  # Making the center bright

    # Define the patch size, stride, and background threshold
    patch_size = (3, 3)
    stride = 1
    background_threshold = 0.5

    # Call the function
    result = get_signal_patch_positions(
        image=image,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
    )

    # Check that the function correctly identified the signal patch
    assert (
        [4, 4] in result.tolist()
    ), "The signal patch at position (4, 4) was not identified correctly"


def test_get_signal_patch_positions_no_signal_error():
    # Create an image with all values below the threshold
    image = np.zeros((10, 10), dtype=float)
    patch_size = (3, 3)
    stride = 1
    background_threshold = 0.5

    # Check for ValueError when no signal patches are found
    with pytest.raises(ValueError) as excinfo:
        get_signal_patch_positions(
            image=image,
            patch_size=patch_size,
            stride=stride,
            background_threshold=background_threshold,
        )
    assert "Couldn't find any signal patches in the image" in str(
        excinfo.value
    ), "Expected ValueError for no signal patches was not raised"


def test_pad_patch_ids_first():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 5
    padded = pad_patch_ids(candidate_patch_ids, num_patches, mode="first")
    assert np.array_equal(
        padded, np.array([1, 2, 3, 1, 1])
    ), "Padding with the first element failed"


def test_pad_patch_ids_repeat_sequence():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 7
    padded = pad_patch_ids(candidate_patch_ids, num_patches, mode="repeat_sequence")
    assert np.array_equal(
        padded, np.array([1, 2, 3, 1, 2, 3, 1])
    ), "Repeating sequence padding failed"


def test_pad_patch_ids_circular():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 6
    padded = pad_patch_ids(candidate_patch_ids, num_patches, mode="circular")
    assert np.array_equal(
        padded, np.array([1, 2, 3, 1, 2, 3])
    ), "Circular padding failed"


def test_pad_patch_ids_mirror():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 6
    padded = pad_patch_ids(candidate_patch_ids, num_patches, mode="mirror")
    assert np.array_equal(padded, np.array([1, 2, 3, 3, 2, 1])), "Mirror padding failed"


def test_pad_patch_ids_random():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 5
    padded = pad_patch_ids(candidate_patch_ids, num_patches, mode="random")
    # Check that all elements in padded are from candidate_patch_ids
    assert all(item in candidate_patch_ids for item in padded), "Random padding failed"


def test_pad_patch_ids_unknown_mode():
    candidate_patch_ids = np.array([1, 2, 3])
    num_patches = 5
    with pytest.raises(ValueError) as excinfo:
        pad_patch_ids(candidate_patch_ids, num_patches, mode="unknown")
    assert "Unknown padding mode specified" in str(
        excinfo.value
    ), "Error not raised for unknown mode"


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


if __name__ == "__main__":
    pytest.main([__file__])
