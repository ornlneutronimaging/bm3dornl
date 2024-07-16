import pytest
from unittest.mock import patch
import numpy as np
from bm3dornl.bm3d import bm3d_ring_artifact_removal, default_block_matching_kwargs, default_filter_kwargs


@pytest.fixture
def setup_sinogram():
    return np.random.rand(256, 256)


@patch("bm3dornl.bm3d.bm3d_full")
@patch("bm3dornl.bm3d.bm3d_lite")
def test_bm3d_ring_artifact_removal(mock_bm3d_lite, mock_bm3d_full, setup_sinogram):
    sinogram = setup_sinogram
    mock_bm3d_lite.return_value = np.random.rand(256, 256)
    mock_bm3d_full.return_value = np.random.rand(256, 256)

    result_simple = bm3d_ring_artifact_removal(sinogram, mode="simple")
    result_express = bm3d_ring_artifact_removal(sinogram, mode="express")
    result_full = bm3d_ring_artifact_removal(sinogram, mode="full")

    assert result_simple is not None
    assert result_express is not None
    assert result_full is not None

    mock_bm3d_lite.assert_any_call(
        sinogram=sinogram,
        block_matching_kwargs=default_block_matching_kwargs,
        filter_kwargs=default_filter_kwargs,
        use_refiltering=True,
    )
    mock_bm3d_lite.assert_any_call(
        sinogram=sinogram,
        block_matching_kwargs=default_block_matching_kwargs,
        filter_kwargs=default_filter_kwargs,
        use_refiltering=False,
    )
    mock_bm3d_full.assert_called_once_with(
        sinogram=sinogram,
        block_matching_kwargs=default_block_matching_kwargs,
        filter_kwargs=default_filter_kwargs,
    )


if __name__ == "__main__":
    pytest.main([__file__])
