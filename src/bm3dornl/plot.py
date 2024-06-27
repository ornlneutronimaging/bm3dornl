#!/usr/bin/env python3
"""Plotting utilities for BM3D and ORNL."""

import numpy as np
from typing import Tuple


def compute_cdf(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the cumulative distribution function of an image.

    Parameters
    ----------
    img : np.ndarray
        The input image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The sorted CDF values and the corresponding probabilities.
    """
    cdf_org_sorted = np.sort(img.flatten())
    p_org = 1.0 * np.arange(len(cdf_org_sorted)) / (len(cdf_org_sorted) - 1)
    return cdf_org_sorted, p_org
