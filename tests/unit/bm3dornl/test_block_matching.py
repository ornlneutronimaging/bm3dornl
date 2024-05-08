#!/usr/bin/env python3
"""Unit test for block matching functions."""

import pytest
import numpy as np
from bm3dornl.block_matching import PatchManager


@pytest.fixture
def patch_manager():
    image = np.ones(400).reshape(20, 20).astype(float)  # Simple uniform image
    patch_size = (5, 5)
    stride = 5
    background_threshold = (
        0.1  # All patches are signal since threshold is lower than image values
    )
    manager = PatchManager(image, patch_size, stride, background_threshold)
    return manager


def test_generate_patch_positions(patch_manager):
    expected_number_of_patches = (20 // 5) * (
        20 // 5
    )  # As stride equals the patch size
    assert (
        len(patch_manager.signal_patches_pos) == expected_number_of_patches
    ), "Incorrect number of signal patches generated."


def test_get_patch(patch_manager):
    expected_patch = patch_manager.image[0:5, 0:5]
    retrieved_patch = patch_manager.get_patch((0, 0))
    np.testing.assert_array_equal(
        retrieved_patch, expected_patch, "Patch retrieved incorrectly."
    )


def test_group_signal_patches_geometric(patch_manager):
    cut_off_distance = (100, 100)  # Larger than image dimensions
    intensity_diff_threshold = 0.5  # Irrelevant due to uniform image
    patch_manager.group_signal_patches(cut_off_distance, intensity_diff_threshold)
    expected_blocks = np.ones(
        (len(patch_manager.signal_patches_pos), len(patch_manager.signal_patches_pos)),
        dtype=bool,
    )
    np.testing.assert_array_equal(
        patch_manager.signal_blocks_matrix,
        expected_blocks,
        "Signal blocks grouped incorrectly.",
    )


def test_get_4d_patch_groups(patch_manager):
    cut_off_distance = (100, 100)  # Larger than image dimensions
    intensity_diff_threshold = 0.5  # Irrelevant due to uniform image
    patch_manager.group_signal_patches(cut_off_distance, intensity_diff_threshold)
    num_patches_per_group = 4
    padding_mode = "circular"
    patch_groups, positions = patch_manager.get_4d_patch_groups(
        num_patches_per_group, padding_mode
    )
    assert patch_groups.shape == (
        len(patch_manager.signal_patches_pos),
        num_patches_per_group,
        5,
        5,
    ), "Incorrect shape of patch groups."
    assert positions.shape == (
        len(patch_manager.signal_patches_pos),
        num_patches_per_group,
        2,
    ), "Incorrect shape of positions."


if __name__ == "__main__":
    pytest.main([__file__])
