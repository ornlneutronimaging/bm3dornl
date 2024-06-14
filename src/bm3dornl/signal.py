#!/usr/bin/env python3
"""Module for signal processing."""

import numpy as np
from scipy.linalg import hadamard


def hadamard_transform(patch: np.ndarray) -> np.ndarray:
    """Apply the Hadamard transform to a patch."""
    H = hadamard(patch.shape[0])
    return np.dot(H, np.dot(patch, H))


def inverse_hadamard_transform(transformed_patch: np.ndarray) -> np.ndarray:
    """Apply the inverse Hadamard transform to a transformed patch."""
    H = hadamard(transformed_patch.shape[0])
    return np.dot(H, np.dot(transformed_patch, H)) / transformed_patch.size
