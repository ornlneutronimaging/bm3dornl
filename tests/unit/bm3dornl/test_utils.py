#!/usr/bin/env python3

import numpy as np
import pytest
from bm3dornl.utils import compute_cdf


def test_compute_cdf():
    test_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32)

    cdf_sorted, p = compute_cdf(test_image)

    expected_cdf_sorted = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    expected_p = np.linspace(0, 1, len(expected_cdf_sorted))

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
