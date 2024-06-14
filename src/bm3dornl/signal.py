#!/usr/bin/env python3
"""Module for signal processing."""

import numpy as np
from scipy.linalg import hadamard


def fft_transform(patches: np.ndarray) -> np.ndarray:
    """Apply the FFT to a patch."""
    return np.abs(np.fft.fftn(patches, axes=(-2, -1)))


def inverse_fft_transform(transformed_patches: np.ndarray) -> np.ndarray:
    """Apply the inverse FFT to a transformed patch."""
    return np.fft.ifftn(transformed_patches, axes=(-2, -1)).real


def hadamard_transform(patches: np.ndarray) -> np.ndarray:
    """Apply the Hadamard transform to a patch."""
    H = hadamard(patches.shape[-1])
    patches = np.einsum("ij,...jl->...il", H, patches)
    patches = np.einsum("...ij,jl->...il", patches, H)
    return patches


def inverse_hadamard_transform(transformed_patches: np.ndarray) -> np.ndarray:
    """Apply the inverse Hadamard transform to a transformed patch."""
    H = hadamard(transformed_patches.shape[-1])
    transformed_patches = np.einsum("ij,...jl->...il", H, transformed_patches)
    transformed_patches = np.einsum("...ij,jl->...il", transformed_patches, H)
    return transformed_patches
