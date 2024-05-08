#!/usr/bin/env python3

"""Unit test for patch aggregation functions."""

import pytest
import numpy as np
from bm3dornl.aggregation import aggregate_patches


def test_aggregate_patches():
    # Setup
    # ph, pw = 2, 2  # patch height and width
    # num_blocks = 1
    # num_patches_per_block = 2

    # Create a simple hyper block with known values
    hyper_block = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

    # Index positions where patches will be placed
    hyper_block_index = np.array(
        [
            [
                [0, 0],  # First patch at top-left corner
                [0, 0],  # Second patch also starts at top-left for overlap
            ]
        ]
    )

    # Initial image and weights matrices sized 2x2
    estimate_denoised_image = np.zeros((2, 2), dtype=float)
    weights = np.zeros((2, 2), dtype=float)

    # Expected outputs
    expected_image = np.array(
        [
            [6, 8],  # Both patches contribute to the first row
            [10, 12],  # Both patches contribute to the second row
        ]
    )
    expected_weights = np.array(
        [
            [2, 2],  # Both patches contribute to each position
            [2, 2],
        ]
    )

    # Invoke the function under test
    aggregate_patches(estimate_denoised_image, weights, hyper_block, hyper_block_index)

    # Assertions
    np.testing.assert_array_almost_equal(
        estimate_denoised_image,
        expected_image,
        err_msg="Image aggregation did not match expected",
    )
    np.testing.assert_array_equal(
        weights, expected_weights, err_msg="Weights update did not match expected"
    )


if __name__ == "__main__":
    pytest.main([__file__])
