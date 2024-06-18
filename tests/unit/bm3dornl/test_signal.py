import numpy as np
import pytest
from bm3dornl.signal import (
    fft_transform,
    inverse_fft_transform,
    hadamard_transform,
    inverse_hadamard_transform,
)


@pytest.fixture
def sample_patches():
    return np.random.rand(8, 8, 8).astype(float)


def test_fft_transform(sample_patches):
    transformed = fft_transform(sample_patches)
    assert transformed.shape == sample_patches.shape
    assert transformed.dtype == float
    assert np.all(transformed >= 0)


def test_inverse_fft_transform(sample_patches):
    transformed = fft_transform(sample_patches)
    inverse_transformed = inverse_fft_transform(transformed)
    assert inverse_transformed.shape == sample_patches.shape
    assert inverse_transformed.dtype == float


def test_hadamard_transform(sample_patches):
    transformed = hadamard_transform(sample_patches)
    assert transformed.shape == sample_patches.shape
    assert transformed.dtype == float


def test_inverse_hadamard_transform(sample_patches):
    transformed = hadamard_transform(sample_patches)
    inverse_transformed = inverse_hadamard_transform(transformed)
    assert inverse_transformed.shape == sample_patches.shape
    assert inverse_transformed.dtype == float


if __name__ == "__main__":
    pytest.main([__file__])
