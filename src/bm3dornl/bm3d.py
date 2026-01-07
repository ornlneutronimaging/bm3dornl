#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from . import bm3d_rust

# Default parameters (simplified for Rust backend)
default_block_matching_kwargs = {
    "patch_size": (7, 7),
    "stride": 2,
    "cut_off_distance": (40, 40),
    "num_patches_per_group": 64,
}
# Legacy filter kwargs preserved just for signature compatibility if needed
default_filter_kwargs = {}

def estimate_streak_profile(sinogram, sigma_smooth=3.0):
    """
    Estimate the static vertical streak profile using a data-driven residual median approach.
    
    Assumption: Ring artifacts appear as constant vertical lines in the sinogram.
    Method:
    1. Estimate the object structure (Low Frequency) using strong smoothing.
    2. Compute Residual = Original - Object.
    3. Compute Column-wise Median of Residuals (Robust estimate of the static vertical offset).
    """
    # 1. Estimate Object (Smooth heavily to remove streaks and noise, keep shape)
    # Allows anisotropic sigma: (sigma_y, sigma_x)
    z_smooth = gaussian_filter(sinogram, sigma_smooth)
    
    # 2. Residual (Contains Noise + Streaks + Fine Details)
    residual = sinogram - z_smooth
    
    # 3. Streak Profile (Median along vertical axis)
    # This isolates the constant vertical component, rejecting random noise and moving object features.
    streak_profile = np.median(residual, axis=0)
    
    # 4. Refine Profile (Optional: slight smoothing to prevent 1px jaggedness)
    # sigma=1.0 is gentle enough to keep sharp streak edges but kill single-pixel outliers
    streak_profile = gaussian_filter1d(streak_profile, 1.0)
    
    return streak_profile

def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    mode: str = "generic",  # "generic" or "streak"
    sigma: float = 0.1,    
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
) -> np.ndarray:
    """Remove ring artifacts (streaks) from a sinogram.
    
    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram (2D float array).
    mode : str
        "generic": Standard BM3D (assume white noise).
        "streak": Additive Streak Removal (Residual Median) + Standard BM3D.
    sigma : float
        Noise standard deviation.
    """
    # Unpack parameters
    patch_size = block_matching_kwargs.get("patch_size", (7, 7))
    if isinstance(patch_size, int): patch_size_dim = patch_size
    else: patch_size_dim = patch_size[0]

    step_size = block_matching_kwargs.get("stride", 2)
    cut_off = block_matching_kwargs.get("cut_off_distance", (40, 40))[0]
    num_patches = block_matching_kwargs.get("num_patches_per_group", 64)
    threshold_hard = 2.7
    
    logger = logging.getLogger(__name__)
    
    # Ensure float32 and normalize
    z = sinogram.astype(np.float32)
    d_min = z.min()
    d_max = z.max()
    z_norm = z
    if d_max > d_min:
        z_norm = (z - d_min) / (d_max - d_min)

    # --- Mode Support ---
    sigma_psd = np.zeros((1, 1), dtype=np.float32) # Default dummy
    
    if mode == "streak":
        # Data-Driven Streak Removal
        # Instead of FFT Filtering, we estimate the streak vector and subtract it.
        
        # Default smooth is 3.0. Use larger (e.g. 15.0) to capture wider streaks in Low SNR.
        sigma_smooth = filter_kwargs.get("sigma_smooth", 3.0)
        
        # 1. Estimate the additive streak profile
        streak_profile = estimate_streak_profile(z_norm, sigma_smooth=sigma_smooth)
        
        # 2. Subtract streaks
        # Broadcast 1D profile to 2D
        correction = np.tile(streak_profile, (z_norm.shape[0], 1))
        z_norm = z_norm - correction
        
        # 3. Post-Correction Clipping?
        # Subtraction might push values < 0 slightly. 
        # But BM3D handles floats fine.
        
    # --- Generic BM3D Execution ---

    # --- Generic BM3D Execution ---
    # Generic Mode (Scalar Sigma)
    sigma_norm = sigma
    
    # Step 1: Hard Thresholding
    yhat_ht = bm3d_rust.bm3d_hard_thresholding(
        z_norm, z_norm, sigma_psd, sigma_norm, threshold_hard,
        patch_size_dim, step_size, cut_off, num_patches
    )
    
    z_gft_ht = yhat_ht 
    yhat_ht_gft = bm3d_rust.bm3d_hard_thresholding(
        z_norm, z_gft_ht, sigma_psd, sigma_norm, threshold_hard,
        patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Step 2: Wiener Filtering
    yhat_wie = bm3d_rust.bm3d_wiener_filtering(
         z_norm, yhat_ht_gft, sigma_psd, sigma_norm,
         patch_size_dim, step_size, cut_off, num_patches
    )
    
    z_gft_wie = yhat_wie
    yhat_final = bm3d_rust.bm3d_wiener_filtering(
         z_norm, z_gft_wie, sigma_psd, sigma_norm,
         patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Denormalize and Return
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    return yhat_final
