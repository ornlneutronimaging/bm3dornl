import numpy as np
import pytest
from bm3dornl.noise_analysis import (
    estimate_noise_psd,
    get_exact_noise_variance_fft,
    get_exact_noise_variance,
    get_exact_noise_variance_hadamard,
)


@pytest.fixture
def sample_noisy_image():
    return np.random.rand(64, 64).astype(float)


@pytest.fixture
def sample_patches():
    return np.random.rand(10, 8, 8).astype(float)


def test_estimate_noise_psd(sample_noisy_image):
    noise_psd = estimate_noise_psd(sample_noisy_image)
    assert noise_psd.shape == sample_noisy_image.shape
    assert noise_psd.dtype == float
    assert np.all(noise_psd >= 0)


def test_get_exact_noise_variance_fft(sample_patches):
    noise_variance = get_exact_noise_variance_fft(sample_patches)
    assert noise_variance.shape == sample_patches.shape
    assert noise_variance.dtype == float
    assert np.all(noise_variance >= 0)


def test_get_exact_noise_variance(sample_patches):
    transformed_patches = np.fft.fftn(sample_patches, axes=(-2, -1))
    noise_variance = get_exact_noise_variance(transformed_patches)
    assert noise_variance.shape == sample_patches.shape
    assert noise_variance.dtype == float
    assert np.all(noise_variance >= 0)


def test_get_exact_noise_variance_hadamard(sample_patches):
    noise_variance = get_exact_noise_variance_hadamard(sample_patches)
    assert noise_variance.shape == sample_patches.shape
    assert noise_variance.dtype == float
    assert np.all(noise_variance >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
