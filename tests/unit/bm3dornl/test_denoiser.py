#!/usr/env/bin python3

"""Unit test for denoiser module."""

import pytest
import numpy as np
from unittest.mock import patch
from bm3dornl.denoiser import (
    BM3D,
    bm3d_streak_removal,
)


def test_bm3d_initialization():
    image = np.random.rand(64, 64)
    bm3d_instance = BM3D(image)
    assert np.array_equal(
        bm3d_instance.image, image
    ), "The images should be identical after initialization."
    assert (
        bm3d_instance.patch_manager is not None
    ), "PatchManager should be initialized."


def test_group_signal_patches():
    image = np.random.rand(64, 64)

    bm3d_instance = BM3D(image)
    with patch.object(
        bm3d_instance.patch_manager, "group_signal_patches"
    ) as mock_method:
        bm3d_instance.group_signal_patches((5, 5), 0.1)
        mock_method.assert_called_once_with((5, 5), 0.1)


@patch("bm3dornl.denoiser.memory_cleanup")
@patch("bm3dornl.denoiser.aggregate_patches")
@patch("bm3dornl.denoiser.hard_thresholding")
@patch("bm3dornl.denoiser.PatchManager", autospec=True)
def test_thresholding(
    mock_patch_manager,
    mock_hard_thresholding,
    mock_aggregate_patches,
    mock_memory_cleanup,
):
    # Call the method to be tested
    image = np.random.rand(64, 64)
    bm3d_instance = BM3D(image)

    mock_patch_manager.return_value.get_hyper_block.return_value = (
        np.random.rand(10, 8, 8),
        np.random.randint(0, 64, (10, 2)),
    )

    with patch.object(
        bm3d_instance, "group_signal_patches"
    ) as mock_group_signal_patches:
        bm3d_instance.thresholding((5, 5), 0.1, 10, 0.1)
        mock_group_signal_patches.assert_called_once_with((5, 5), 0.1)
        mock_hard_thresholding.assert_called_once()
        mock_aggregate_patches.assert_called_once()
        mock_memory_cleanup.assert_called_once()


@patch("bm3dornl.denoiser.wiener_hadamard")
@patch("bm3dornl.denoiser.aggregate_patches")
@patch("bm3dornl.denoiser.memory_cleanup")
@patch("bm3dornl.denoiser.PatchManager", autospec=True)
def test_re_filtering(
    mock_patch_manager,
    mock_memory_cleanup,
    mock_aggregate_patches,
    mock_wiener_hadamard,
):
    image = np.random.rand(64, 64)
    bm3d_instance = BM3D(image)
    mock_patch_manager.return_value.get_hyper_block.return_value = (
        np.random.rand(10, 8, 8),
        np.random.randint(0, 64, (10, 2)),
    )
    mock_wiener_hadamard.return_value = np.random.rand(10, 8, 8)

    with patch.object(
        bm3d_instance, "group_signal_patches"
    ) as mock_group_signal_patches:
        bm3d_instance.re_filtering((5, 5), 0.1, 10)
        mock_group_signal_patches.assert_called_once_with((5, 5), 0.1)

    mock_wiener_hadamard.assert_called_once()
    mock_aggregate_patches.assert_called_once()
    mock_memory_cleanup.assert_called_once()


@patch("bm3dornl.denoiser.BM3D", autospec=True)
@patch("bm3dornl.denoiser.horizontal_binning", return_value=np.random.rand(64, 64))
@patch("bm3dornl.denoiser.horizontal_debinning", return_value=np.random.rand(64, 64))
@patch("bm3dornl.denoiser.medfilt2d", return_value=np.random.rand(64, 64))
def test_bm3d_streak_removal(
    mock_medfilt2d, mock_horizontal_debinning, mock_horizontal_binning, mock_bm3d
):
    sinogram = np.random.rand(64, 64)
    mock_bm3d_instance = mock_bm3d.return_value
    mock_bm3d_instance.final_denoised_image = np.random.rand(
        64, 64
    )  # Set the final_denoised_image attribute

    result = bm3d_streak_removal(sinogram, k=1)

    assert result.shape == (64, 64), "The output should maintain the input dimensions."
    mock_medfilt2d.assert_called_once_with(sinogram, kernel_size=3)
    mock_horizontal_binning.assert_called()
    mock_bm3d.assert_called()
    mock_horizontal_debinning.assert_called()


@patch("bm3dornl.denoiser.BM3D", autospec=True)
@patch(
    "bm3dornl.denoiser.horizontal_binning",
    return_value=[np.random.rand(64, 64) for _ in range(3)],
)
@patch("bm3dornl.denoiser.horizontal_debinning", return_value=np.random.rand(64, 64))
@patch("bm3dornl.denoiser.medfilt2d", return_value=np.random.rand(64, 64))
def test_bm3d_streak_removal_multiscale(
    mock_medfilt2d, mock_horizontal_debinning, mock_horizontal_binning, mock_bm3d
):
    sinogram = np.random.rand(64, 64)
    mock_bm3d_instance = mock_bm3d.return_value
    mock_bm3d_instance.final_denoised_image = np.random.rand(
        64, 64
    )  # Set the final_denoised_image attribute

    result = bm3d_streak_removal(sinogram, k=3)

    assert result.shape == (64, 64), "The output should maintain the input dimensions."
    mock_medfilt2d.assert_called_once_with(sinogram, kernel_size=3)
    mock_horizontal_binning.assert_called()
    mock_bm3d.assert_called()
    mock_horizontal_debinning.assert_called()


@patch(
    "bm3dornl.denoiser.estimate_noise_free_sinogram",
    return_value=np.random.rand(64, 64),
)
def test_denoise(mock_estimate_noise_free_sinogram):
    image = np.random.rand(64, 64)
    bm3d_instance = BM3D(image)

    with patch.object(bm3d_instance, "re_filtering") as mock_re_filtering, patch.object(
        bm3d_instance, "thresholding"
    ) as mock_thresholding:
        mock_re_filtering.return_value = None
        # fast
        bm3d_instance.denoise((5, 5), 0.1, 10, 0.1, fast_estimate=True)
        mock_estimate_noise_free_sinogram.assert_called_once()
        mock_re_filtering.assert_called_once()
        # slow
        bm3d_instance.denoise((5, 5), 0.1, 10, 0.1, fast_estimate=False)
        mock_thresholding.assert_called_once()
        assert mock_re_filtering.call_count == 2  # Should be called twice now


if __name__ == "__main__":
    pytest.main([__file__])
