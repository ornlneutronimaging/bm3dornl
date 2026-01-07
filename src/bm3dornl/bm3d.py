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

    # --- Spatially Adaptive BM3D Setup ---
    # 1. Estimate Streak MAP (Spatially Varying Variance)
    sigma_map = filter_kwargs.get("sigma_map", None)
    
    # If not provided, estimate from the single image (Data-Driven)
    if sigma_map is None:
        # Use simple gradient-based or residual-based estimator
        # We assume streaks are vertical.
        # High-pass filter along X (rows) to find vertical edges, filter along Y (cols) to find long structures.
        
        # Re-use our robust estimator logic:
        # Residual = Sinogram - Smooth(Sinogram)
        # StreakProfile = Median(Residual)
        # SigmaMap = Abs(StreakProfile) broadcasted.
        
        # We reuse the logic from `mode="streak"` loop below, or pre-calc it?
        # Let's pre-calc it effectively.
        sigma_smooth_est = 3.0
        z_smooth_est = gaussian_filter(z_norm, (sigma_smooth_est, sigma_smooth_est))
        residual_est = z_norm - z_smooth_est
        profile_est = np.median(residual_est, axis=0) # (W,)
        
        # Refine profile (remove DC offset of object)
        profile_smooth = gaussian_filter1d(profile_est, 50.0)
        streak_signal = profile_est - profile_smooth
        
        # Map is the Magnitude of the Streak Signal (Assuming Noise ~ Signal strength for these artifacts)
        # We broadcast to (H, W)
        sigma_streak_1d = np.abs(streak_signal).astype(np.float32)
        # Amplify? User complained of "insufficient cleaning".
        # Let's scale it slightly to be aggressive on streaks.
        sigma_map = np.tile(sigma_streak_1d, (z_norm.shape[0], 1))
        
    sigma_map = np.ascontiguousarray(sigma_map, dtype=np.float32)
    
    # 2. Sigma Random (White Noise Floor)
    sigma_random = float(sigma)

    # --- Mode Support (Streak Subtraction) ---
    sigma_psd = np.zeros((1, 1), dtype=np.float32) 
    
    if mode == "streak":
        # Additional Mean Subtraction (Shift).
        # We keep this as it works well for the DC component.
        sigma_smooth = filter_kwargs.get("sigma_smooth", 3.0)
        iterations = filter_kwargs.get("streak_iterations", 1)
        
        for i in range(iterations):
            streak_profile = estimate_streak_profile(z_norm, sigma_smooth=sigma_smooth)
            correction = np.tile(streak_profile, (z_norm.shape[0], 1))
            z_norm = z_norm - correction
        
        # Construct PSD for Vertical Streaks
        # Vertical streaks (constant in Y) corresponds to energy in the first row of the 2D transform (freq Y = 0)
        # We assume 2D FFT (DFT).
        if patch_size_dim > 0:
            sigma_psd = np.zeros((patch_size_dim, patch_size_dim), dtype=np.float32)
            # Assign weight to Row 0 (DC along Y, all freqs along X)
            # We assume uniform distribution along X logic for the streak profile "noise"
            # The magnitude is controlled by sigma_map.
            sigma_psd[0, :] = 1.0
            # Optional: Add small weight to row 1 for smoothness/leakage
            # sigma_psd[1, :] = 0.5 

    
    # --- BM3D Execution (Adaptive) ---
    
    # Step 1: Hard Thresholding
    # Note: We now pass sigma_map and sigma_random
    yhat_ht = bm3d_rust.bm3d_hard_thresholding(
        z_norm, z_norm, sigma_psd, 
        sigma_map, sigma_random, 
        threshold_hard,
        patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Step 2: Wiener Filtering
    yhat_final = bm3d_rust.bm3d_wiener_filtering(
         z_norm, yhat_ht, sigma_psd, 
         sigma_map, sigma_random,
         patch_size_dim, step_size, cut_off, num_patches
    )
    
    # Denormalize and Return
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    return yhat_final
