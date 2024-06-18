import pytest
from unittest.mock import patch
import numpy as np
from bm3dornl.bm3d import collaborative_filtering


@pytest.fixture
def setup_data():
    sinogram = np.random.rand(256, 256)
    denoised_sinogram = np.random.rand(256, 256)
    patch_size = (8, 8)
    num_patches_per_group = 32
    padding_mode = "circular"
    noise_variance = np.random.rand(100, 32, 8, 8)
    patch_positions = np.random.randint(0, 256, (100, 2))
    cut_off_distance = (64, 64)
    return (
        sinogram,
        denoised_sinogram,
        patch_size,
        num_patches_per_group,
        padding_mode,
        noise_variance,
        patch_positions,
        cut_off_distance,
    )


@patch("bm3dornl.bm3d.aggregate_denoised_block_to_image")
@patch("bm3dornl.bm3d.collaborative_wiener_filtering")
@patch("bm3dornl.bm3d.form_hyper_blocks_from_two_images")
@patch("bm3dornl.bm3d.compute_distance_matrix_no_variance")
@patch("bm3dornl.bm3d.get_patch_numba")
def test_collaborative_filtering(
    mock_get_patch_numba,
    mock_distance_matrix,
    mock_form_hyper_blocks,
    mock_collaborative_filtering,
    mock_aggregate_block,
    setup_data,
):
    (
        sinogram,
        denoised_sinogram,
        patch_size,
        num_patches_per_group,
        padding_mode,
        noise_variance,
        patch_positions,
        cut_off_distance,
    ) = setup_data

    mock_get_patch_numba.return_value = np.random.rand(8, 8)
    mock_distance_matrix.return_value = np.random.rand(100, 100)
    mock_form_hyper_blocks.return_value = (
        np.random.rand(100, 32, 8, 8),
        np.random.rand(100, 32, 8, 8),
        np.random.randint(0, 256, (100, 32, 2)),
        np.random.rand(100, 32, 8, 8),
    )
    mock_collaborative_filtering.return_value = np.random.rand(100, 32, 8, 8)
    mock_aggregate_block.return_value = np.random.rand(256, 256)

    result = collaborative_filtering(
        sinogram,
        denoised_sinogram,
        patch_size,
        num_patches_per_group,
        padding_mode,
        noise_variance,
        patch_positions,
        cut_off_distance,
        lambda x: x,
        mock_collaborative_filtering,
    )

    assert result is not None
    assert mock_get_patch_numba.call_count == len(patch_positions)
    mock_distance_matrix.assert_called_once()
    mock_form_hyper_blocks.assert_called_once()
    mock_collaborative_filtering.assert_called_once()
    mock_aggregate_block.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
