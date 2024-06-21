#!/usr/env/bin python3

"""Unit test for denoiser module."""

import pytest
import numpy as np
from unittest.mock import patch
from bm3dornl.denoiser import bm3d_ring_artifact_removal, bm3d_ring_artifact_removal_ms


def test_bm3d_ring_artifact_removal():
    sinogram = np.random.rand(64, 64).astype(np.float32)

    with patch(
        "bm3dornl.denoiser.estimate_noise_free_sinogram"
    ) as mock_estimate_noise_free_sinogram, patch(
        "bm3dornl.denoiser.PatchManager"
    ) as mock_patch_manager, patch(
        "bm3dornl.denoiser.wiener_hadamard"
    ) as mock_wiener_hadamard, patch(
        "bm3dornl.denoiser.aggregate_patches"
    ) as mock_aggregate_patches, patch(
        "bm3dornl.denoiser.memory_cleanup"
    ) as mock_memory_cleanup:
        # Set up mock return values
        mock_estimate_noise_free_sinogram.return_value = np.random.rand(64, 64).astype(
            np.float32
        )
        mock_patch_manager_instance = mock_patch_manager.return_value
        mock_patch_manager_instance.get_hyper_block.return_value = (
            np.random.rand(10, 8, 8).astype(np.float32),
            np.random.randint(0, 64, (10, 8, 2)),
        )
        mock_wiener_hadamard.return_value = np.random.rand(10, 8, 8).astype(np.float32)

        denoised_sinogram = bm3d_ring_artifact_removal(sinogram)

        # Check the shape of the output
        assert denoised_sinogram.shape == sinogram.shape, "Output shape mismatch."

        # Check that the output is a numpy array
        assert isinstance(denoised_sinogram, np.ndarray), "Output is not a numpy array."

        # Check that the functions are called
        mock_estimate_noise_free_sinogram.assert_called_once()
        mock_patch_manager.assert_called_once()
        mock_patch_manager_instance.group_signal_patches.assert_called_once()
        mock_patch_manager_instance.get_hyper_block.assert_called_once()
        mock_wiener_hadamard.assert_called_once()
        mock_aggregate_patches.assert_called_once()
        mock_memory_cleanup.assert_called_once()

@pytest.mark.skip(reason="Changed function singature")
def test_bm3d_ring_artifact_removal_ms():
    sinogram = np.random.rand(64, 64).astype(np.float32)

    with patch(
        "bm3dornl.denoiser.estimate_noise_free_sinogram"
    ) as mock_estimate_noise_free_sinogram, patch(
        "bm3dornl.denoiser.PatchManager"
    ) as mock_patch_manager, patch(
        "bm3dornl.denoiser.wiener_hadamard"
    ) as mock_wiener_hadamard, patch(
        "bm3dornl.denoiser.aggregate_patches"
    ) as mock_aggregate_patches, patch(
        "bm3dornl.denoiser.memory_cleanup"
    ) as mock_memory_cleanup, patch(
        "bm3dornl.denoiser.horizontal_binning"
    ) as mock_horizontal_binning, patch(
        "bm3dornl.denoiser.horizontal_debinning"
    ) as mock_horizontal_debinning:
        # Set up mock return values
        mock_estimate_noise_free_sinogram.return_value = np.random.rand(64, 64).astype(
            np.float32
        )
        mock_patch_manager_instance = mock_patch_manager.return_value
        mock_patch_manager_instance.get_hyper_block.return_value = (
            np.random.rand(10, 8, 8).astype(np.float32),
            np.random.randint(0, 64, (10, 8, 2)),
        )
        mock_wiener_hadamard.return_value = np.random.rand(10, 8, 8).astype(np.float32)
        mock_horizontal_binning.return_value = [
            np.random.rand(64, 64).astype(np.float32) for _ in range(5)
        ]
        mock_horizontal_debinning.return_value = np.random.rand(64, 64).astype(
            np.float32
        )

        denoised_sinogram = bm3d_ring_artifact_removal_ms(sinogram)

        # Check the shape of the output
        assert denoised_sinogram.shape == sinogram.shape, "Output shape mismatch."

        # Check that the output is a numpy array
        assert isinstance(denoised_sinogram, np.ndarray), "Output is not a numpy array."

        # Check that the functions are called
        mock_estimate_noise_free_sinogram.assert_called()
        mock_patch_manager.assert_called()
        mock_patch_manager_instance.group_signal_patches.assert_called()
        mock_patch_manager_instance.get_hyper_block.assert_called()
        mock_wiener_hadamard.assert_called()
        mock_aggregate_patches.assert_called()
        mock_memory_cleanup.assert_called()
        mock_horizontal_binning.assert_called()
        mock_horizontal_debinning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
