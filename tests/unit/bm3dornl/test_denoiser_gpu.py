import numpy as np
import pytest
from bm3dornl.denoiser_gpu import (
    memory_cleanup,
    shrinkage_fft,
    shrinkage_hadamard,
    collaborative_wiener_filtering,
    collaborative_hadamard_filtering,
)


@pytest.fixture
def sample_hyper_blocks():
    return np.random.rand(10, 5, 8, 8).astype(float)


@pytest.fixture
def sample_variance_blocks():
    return np.random.rand(10, 5, 8, 8).astype(float)


@pytest.fixture
def sample_noisy_hyper_blocks():
    return np.random.rand(10, 5, 8, 8).astype(float)


@pytest.mark.cuda_required
def test_memory_cleanup():
    # Test if the memory cleanup function runs without error
    memory_cleanup()


@pytest.mark.cuda_required
def test_shrinkage_fft(sample_hyper_blocks, sample_variance_blocks):
    threshold_factor = 3
    denoised_blocks = shrinkage_fft(
        sample_hyper_blocks, sample_variance_blocks, threshold_factor
    )
    assert denoised_blocks.shape == sample_hyper_blocks.shape
    assert denoised_blocks.dtype == float


@pytest.mark.cuda_required
def test_shrinkage_hadamard(sample_hyper_blocks, sample_variance_blocks):
    threshold_factor = 3
    denoised_blocks = shrinkage_hadamard(
        sample_hyper_blocks, sample_variance_blocks, threshold_factor
    )
    assert denoised_blocks.shape == sample_hyper_blocks.shape
    assert denoised_blocks.dtype == float


@pytest.mark.cuda_required
def test_collaborative_wiener_filtering(
    sample_hyper_blocks, sample_variance_blocks, sample_noisy_hyper_blocks
):
    denoised_blocks = collaborative_wiener_filtering(
        sample_hyper_blocks, sample_variance_blocks, sample_noisy_hyper_blocks
    )
    assert denoised_blocks.shape == sample_hyper_blocks.shape
    assert denoised_blocks.dtype == float


@pytest.mark.cuda_required
def test_collaborative_hadamard_filtering(
    sample_hyper_blocks, sample_variance_blocks, sample_noisy_hyper_blocks
):
    denoised_blocks = collaborative_hadamard_filtering(
        sample_hyper_blocks, sample_variance_blocks, sample_noisy_hyper_blocks
    )
    assert denoised_blocks.shape == sample_hyper_blocks.shape
    assert denoised_blocks.dtype == float


if __name__ == "__main__":
    pytest.main([__file__])
