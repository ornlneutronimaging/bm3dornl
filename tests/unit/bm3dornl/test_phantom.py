#!/usr/bin/env python3

"""Unit test for phantom module."""

import pytest
import numpy as np
from bm3dornl.phantom import (
    shepp_logan_phantom,
    generate_sinogram,
    simulate_detector_gain_error,
)


def test_shepp_logan_phantom():
    """Test the shepp_logan_phantom function."""
    size = 256
    contrast_factor = 2.0
    phantom = shepp_logan_phantom(size=size, contrast_factor=contrast_factor)

    # Check the shape
    assert phantom.shape == (size, size), "Phantom shape mismatch"

    # Check that all values are between 0 and 1
    assert phantom.min() >= 0, "Phantom values should be >= 0"
    assert phantom.max() <= 1, "Phantom values should be <= 1"

    # Check that the phantom contains meaningful non-zero values
    assert np.any(phantom > 0), "Phantom should have non-zero values"


def test_generate_sinogram():
    """Test the generate_sinogram function."""
    input_size = 256
    scan_step = 1.0
    input_img = np.random.rand(input_size, input_size)

    sinogram, thetas_deg = generate_sinogram(input_img, scan_step)

    # Verify the shape of the sinogram
    expected_num_projections = int(360 / scan_step)
    assert sinogram.shape == (
        expected_num_projections,
        input_size,
    ), f"Sinogram shape mismatch, expected: {(expected_num_projections, input_size)}"

    # Verify the length of the angles array
    assert thetas_deg.shape == (
        expected_num_projections,
    ), f"Theta shape mismatch, expected: {(expected_num_projections,)}"

    # Ensure that the theta array spans the correct range
    assert thetas_deg.min() >= -180, "Minimum theta value should be -180 degrees"
    assert thetas_deg.max() < 180, "Maximum theta value should be less than 180 degrees"

    # Check for non-zero sinogram
    assert np.any(sinogram > 0), "The sinogram should contain non-zero values"


def test_simulate_detector_gain_error():
    """Test the simulate_detector_gain_error function."""
    # Define the parameters for the test
    sinogram_shape = (360, 256)
    detector_gain_range = (0.9, 1.1)
    detector_gain_error = 0.1

    # Create a random sinogram for testing
    sinogram = np.random.rand(*sinogram_shape)

    # Call the function to simulate gain error
    modified_sinogram, detector_gain = simulate_detector_gain_error(
        sinogram, detector_gain_range, detector_gain_error
    )

    # Ensure the output sinogram and detector gain have the same shape as the input
    assert (
        modified_sinogram.shape == sinogram_shape
    ), f"Output sinogram shape mismatch, expected: {sinogram_shape}"
    assert (
        detector_gain.shape == sinogram_shape
    ), f"Detector gain shape mismatch, expected: {sinogram_shape}"

    # Check that the sinogram is normalized to [0, 1]
    assert modified_sinogram.min() >= 0, "Sinogram values should be >= 0"
    assert modified_sinogram.max() <= 1, "Sinogram values should be <= 1"

    # Ensure that the output is of type float32
    assert (
        modified_sinogram.dtype == np.float32
    ), "Output sinogram should be of type float32"
    assert detector_gain.dtype == np.float32, "Detector gain should be of type float32"


if __name__ == "__main__":
    pytest.main([__file__])
