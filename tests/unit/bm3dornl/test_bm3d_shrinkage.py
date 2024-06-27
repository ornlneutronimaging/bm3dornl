import pytest
from unittest.mock import patch
import numpy as np
from bm3dornl.bm3d import shrinkage_via_hardthresholding


@pytest.fixture
def setup_data():
    sinogram = np.random.rand(256, 256)
    patch_size = (8, 8)
    num_patches_per_group = 32
    padding_mode = "circular"
    transformed_patches = np.random.rand(100, 32, 8, 8)
    noise_variance = np.random.rand(100, 32, 8, 8)
    patch_positions = np.random.randint(0, 256, (100, 2))
    cut_off_distance = (64, 64)
    shrinkage_factor = 3e-2
    return (
        sinogram,
        patch_size,
        num_patches_per_group,
        padding_mode,
        transformed_patches,
        noise_variance,
        patch_positions,
        cut_off_distance,
        shrinkage_factor,
    )


@patch("bm3dornl.bm3d.aggregate_block_to_image")
@patch("bm3dornl.bm3d.shrinkage_fft")
@patch("bm3dornl.bm3d.form_hyper_blocks_from_distance_matrix")
@patch("bm3dornl.bm3d.compute_variance_weighted_distance_matrix")
def test_shrinkage_via_hardthresholding(
    mock_distance_matrix,
    mock_form_hyper_blocks,
    mock_shrinkage_fft,
    mock_aggregate_block,
    setup_data,
):
    (
        sinogram,
        patch_size,
        num_patches_per_group,
        padding_mode,
        transformed_patches,
        noise_variance,
        patch_positions,
        cut_off_distance,
        shrinkage_factor,
    ) = setup_data

    mock_distance_matrix.return_value = np.random.rand(100, 100)
    mock_form_hyper_blocks.return_value = (
        np.random.rand(100, 32, 8, 8),
        np.random.randint(0, 256, (100, 32, 2)),
        np.random.rand(100, 32, 8, 8),
    )
    mock_shrinkage_fft.return_value = np.random.rand(100, 32, 8, 8)
    mock_aggregate_block.return_value = np.random.rand(256, 256)

    result = shrinkage_via_hardthresholding(
        sinogram,
        patch_size,
        num_patches_per_group,
        padding_mode,
        transformed_patches,
        noise_variance,
        patch_positions,
        cut_off_distance,
        shrinkage_factor,
        mock_shrinkage_fft,
    )

    assert result is not None
    mock_distance_matrix.assert_called_once()
    mock_form_hyper_blocks.assert_called_once()
    mock_shrinkage_fft.assert_called_once()
    mock_aggregate_block.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
