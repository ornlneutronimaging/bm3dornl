import numpy as np
import pytest
from bm3dornl.phantom import (
    shepp_logan_phantom,
    generate_sinogram,
    simulate_detector_gain_error,
    apply_convolution,
    get_synthetic_noise,
)


@pytest.fixture
def sample_image():
    return np.random.rand(256, 256).astype(float)


def test_shepp_logan_phantom():
    phantom = shepp_logan_phantom(size=256, contrast_factor=2.0)
    assert phantom.shape == (256, 256)
    assert phantom.dtype == float
    assert np.min(phantom) >= 0
    assert np.max(phantom) <= 1


def test_generate_sinogram(sample_image):
    sinogram, thetas_deg = generate_sinogram(sample_image, scan_step=1)
    assert sinogram.shape[0] == 360
    assert thetas_deg.shape == (360,)
    assert np.min(sinogram) >= 0.000
    assert np.max(sinogram) <= 1.001


def test_simulate_detector_gain_error(sample_image):
    sinogram, _ = generate_sinogram(sample_image, scan_step=1)
    sinogram_with_error, detector_gain = simulate_detector_gain_error(
        sinogram, detector_gain_range=(0.9, 1.1), detector_gain_error=0.1
    )
    assert sinogram_with_error.shape == sinogram.shape
    assert detector_gain.shape == sinogram.shape
    assert np.min(sinogram_with_error) >= 0
    assert np.max(sinogram_with_error) <= 1


def test_apply_convolution(sample_image):
    kernel = np.ones((3, 3))
    convolved_noise = apply_convolution(sample_image, kernel)
    assert convolved_noise.shape == sample_image.shape
    assert convolved_noise.dtype == float


def test_get_synthetic_noise():
    noise = get_synthetic_noise(
        image_size=(256, 256),
        streak_kernel_width=1,
        streak_kernel_length=200,
        white_noise_intensity=0.01,
        streak_noise_intensity=0.01,
    )
    assert noise.shape == (256, 256)
    assert noise.dtype == float
    assert np.min(noise) >= -0.5
    assert np.max(noise) <= 0.5


if __name__ == "__main__":
    pytest.main([__file__])
