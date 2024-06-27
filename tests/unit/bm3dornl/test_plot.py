#!/usr/bin/env python3
"""Unit tests for the plotting utilities."""

import numpy as np
import pytest
from bm3dornl.plot import compute_cdf


def test_compute_cdf():
    # Create a test image
    test_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32)

    # Compute the CDF using the function
    cdf_sorted, p = compute_cdf(test_image)

    # Expected CDF sorted values and probabilities
    expected_cdf_sorted = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    expected_p = np.linspace(0, 1, len(expected_cdf_sorted))

    # Assert that the computed CDF values match the expected values
    np.testing.assert_array_equal(
        cdf_sorted,
        expected_cdf_sorted,
        err_msg="CDF sorted values do not match the expected values.",
    )
    np.testing.assert_array_almost_equal(
        p,
        expected_p,
        decimal=6,
        err_msg="CDF probabilities do not match the expected values.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
