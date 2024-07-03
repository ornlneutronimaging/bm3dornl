import pytest
from unittest.mock import patch
import numpy as np
from bm3dornl.bm3d import bm3d_ring_artifact_removal_ms


size_x = 256
size_y = 256


@pytest.fixture
def setup_sinogram():
    return np.random.rand(size_x, size_y)


def test_bm3d_ring_artifact_removal_ms(
    setup_sinogram,
):
    sinogram = setup_sinogram

    result = bm3d_ring_artifact_removal_ms(sinogram, k=4)

    assert result is not None

    r, c = result.shape

    assert c == size_x
    assert r == size_y

    result_single_pass = bm3d_ring_artifact_removal_ms(sinogram, k=0)
    assert result_single_pass is not None

    r, c = result_single_pass.shape

    assert c == size_x
    assert r == size_y


@pytest.mark.skip(reason="Change function signature")
@patch("bm3dornl.bm3d.horizontal_binning")
@patch("bm3dornl.bm3d.horizontal_debinning")
@patch("bm3dornl.bm3d.bm3d_ring_artifact_removal")
def test_bm3d_ring_artifact_removal_ms(
    mock_bm3d_ring_artifact_removal,
    mock_horizontal_debinning,
    mock_horizontal_binning,
    setup_sinogram,
):
    sinogram = setup_sinogram
    binned_sinos = [np.random.rand(64, 64) for _ in range(4)]
    mock_horizontal_binning.return_value = binned_sinos
    binned_sinos = [np.random.rand(64, 64) for _ in range(4)]
    mock_bm3d_ring_artifact_removal.return_value = binned_sinos
    binned_sinos = [np.random.rand(64, 64) for _ in range(4)]
    mock_horizontal_debinning.return_value = binned_sinos

    result = bm3d_ring_artifact_removal_ms(sinogram, k=4)

    assert result is not None
    mock_horizontal_binning.assert_called_once_with(sinogram, k=4)
    assert mock_bm3d_ring_artifact_removal.call_count == 3
    assert mock_horizontal_debinning.call_count == 3

    result_single_pass = bm3d_ring_artifact_removal_ms(sinogram, k=0)
    assert result_single_pass is not None
    mock_bm3d_ring_artifact_removal.assert_called_with(
        sinogram,
        mode="simple",
        block_matching_kwargs={
            "patch_size": (8, 8),
            "stride": 3,
            "background_threshold": 0.0,
            "cut_off_distance": (64, 64),
            "num_patches_per_group": 32,
            "padding_mode": "circular",
        },
        filter_kwargs={
            "filter_function": "fft",
            "shrinkage_factor": 3e-2,
        },
    )
