import pytest
import numpy as np
from bm3dornl.bm3d import (
    global_fourier_thresholding,
    global_wiener_filtering,
    estimate_noise_free_sinogram,
)


def test_global_fourier_thresholding():
    noisy_image = np.random.rand(256, 256)
    noise_psd = np.random.rand(256, 256)
    estimated_image = np.random.rand(256, 256)

    result = global_fourier_thresholding(noisy_image, noise_psd, estimated_image)

    assert result is not None
    assert result.shape == noisy_image.shape
    assert np.all(np.isfinite(result))


def test_global_wiener_filtering():
    sinogram = np.random.rand(256, 256)

    result = global_wiener_filtering(sinogram)

    assert result is not None
    assert result.shape == sinogram.shape
    assert np.all(np.isfinite(result))
    assert np.min(result) >= 0
    assert np.max(result) <= 1


def test_estimate_noise_free_sinogram():
    sinogram = np.random.rand(256, 256)

    result = estimate_noise_free_sinogram(sinogram)

    assert result is not None
    assert result.shape == sinogram.shape
    assert np.all(np.isfinite(result))
    assert np.min(result) >= 0
    assert np.max(result) <= 1


if __name__ == "__main__":
    pytest.main([__file__])
