import pytest
from unittest.mock import patch
import numpy as np
from bm3dornl.bm3d import bm3d_lite


@pytest.fixture
def setup_sinogram():
    return np.random.rand(256, 256)


@patch("bm3dornl.bm3d.get_patch_numba")
@patch("bm3dornl.bm3d.global_fourier_thresholding")
@patch("bm3dornl.bm3d.collaborative_filtering")
@patch("bm3dornl.bm3d.estimate_noise_free_sinogram")
@patch("bm3dornl.bm3d.get_signal_patch_positions")
def test_bm3d_lite(
    mock_get_signal_patch_positions,
    mock_estimate_noise_free_sinogram,
    mock_collaborative_filtering,
    mock_global_fourier_thresholding,
    mock_get_patch_numba,
    setup_sinogram,
):
    sinogram = setup_sinogram
    mock_get_signal_patch_positions.return_value = np.random.randint(0, 256, (100, 2))
    mock_estimate_noise_free_sinogram.return_value = np.random.rand(256, 256)
    mock_collaborative_filtering.return_value = np.random.rand(256, 256)
    mock_global_fourier_thresholding.return_value = np.random.rand(256, 256)
    mock_get_patch_numba.return_value = np.random.rand(8, 8)

    result = bm3d_lite(sinogram)

    assert result is not None
    mock_get_signal_patch_positions.assert_called_once()
    mock_estimate_noise_free_sinogram.assert_called_once()
    mock_collaborative_filtering.assert_called()
    mock_global_fourier_thresholding.assert_called()
    mock_get_patch_numba.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
