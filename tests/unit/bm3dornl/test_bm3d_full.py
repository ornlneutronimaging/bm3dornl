import pytest
from unittest.mock import patch
import numpy as np
from bm3dornl.bm3d import bm3d_full


@pytest.fixture
def setup_sinogram():
    return np.random.rand(256, 256)


@patch("bm3dornl.bm3d.get_patch_numba")
@patch("bm3dornl.bm3d.global_fourier_thresholding")
@patch("bm3dornl.bm3d.collaborative_filtering")
@patch("bm3dornl.bm3d.shrinkage_via_hardthresholding")
@patch("bm3dornl.bm3d.get_signal_patch_positions")
@patch("bm3dornl.bm3d.estimate_noise_psd")
@patch("bm3dornl.bm3d.get_exact_noise_variance")
@patch("bm3dornl.bm3d.fft_transform")
def test_bm3d_full(
    mock_fft_transform,
    mock_get_exact_noise_variance,
    mock_estimate_noise_psd,
    mock_get_signal_patch_positions,
    mock_shrinkage_via_hardthresholding,
    mock_collaborative_filtering,
    mock_global_fourier_thresholding,
    mock_get_patch_numba,
    setup_sinogram,
):
    sinogram = setup_sinogram
    mock_get_signal_patch_positions.return_value = np.random.randint(0, 256, (100, 2))
    mock_fft_transform.return_value = np.random.rand(100, 8, 8)
    mock_get_exact_noise_variance.return_value = np.random.rand(100, 8, 8)
    mock_shrinkage_via_hardthresholding.return_value = np.random.rand(256, 256)
    mock_collaborative_filtering.return_value = np.random.rand(256, 256)
    mock_global_fourier_thresholding.return_value = np.random.rand(256, 256)
    mock_estimate_noise_psd.return_value = np.random.rand(256, 256)
    mock_get_patch_numba.return_value = np.random.rand(8, 8)

    result = bm3d_full(sinogram)

    assert result is not None
    mock_get_signal_patch_positions.assert_called()
    mock_fft_transform.assert_called()
    mock_get_exact_noise_variance.assert_called()
    mock_shrinkage_via_hardthresholding.assert_called()
    mock_collaborative_filtering.assert_called()
    mock_global_fourier_thresholding.assert_called()
    mock_estimate_noise_psd.assert_called()
    mock_get_patch_numba.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
